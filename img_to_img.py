import sys
import os
sys.path.append(".")
sys.path.append("CLIP")
sys.path.append("unidiffuser")

# os.chdir("unidiffuser")
print(os.getcwd())
print("Importing libraries...")
import ml_collections
import torch
import random
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image, make_grid
import torchvision.transforms as standard_transforms
import numpy as np
from CLIP import clip
from PIL import Image

from libs.uvit_multi_post_ln_v1 import UViT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
nnet = UViT(
    img_size=64,
    in_chans=4,
    patch_size=2,
    embed_dim=1536,
    depth=30,
    num_heads=24,
    text_dim=64,
    num_text_tokens=77,
    clip_img_dim=512,
    use_checkpoint=True
)
nnet.to(device)
nnet.load_state_dict(torch.load('unidiffuser/models/uvit_v1.pth', map_location='cpu'))
nnet.eval()


from libs.caption_decoder import CaptionDecoder
caption_decoder = CaptionDecoder(device=device, pretrained_path="unidiffuser/models/caption_decoder.pth", hidden_dim=64)

clip_text_model = libs.clip.FrozenCLIPEmbedder(device=device)
clip_text_model.eval()
clip_text_model.to(device)

autoencoder = libs.autoencoder.get_model(pretrained_path='unidiffuser/models/autoencoder_kl.pth')
autoencoder.to(device)

clip_img_model, clip_img_model_preprocess = clip.load("ViT-B/32", device=device, jit=False)

@torch.cuda.amp.autocast()
def encode(_batch):
    return autoencoder.encode(_batch)

@torch.cuda.amp.autocast()
def decode(_batch):
    return autoencoder.decode(_batch)

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()
_betas = stable_diffusion_beta_schedule()
N = len(_betas)

def split(x):
    C, H, W = 4, 64, 64
    z_dim = C * H * W
    z, clip_img = x.split([z_dim, 512], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=512)
    return z, clip_img

def combine(z, clip_img):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    return torch.concat([z, clip_img], dim=-1)

def combine_joint(z, clip_img, text):
    z = einops.rearrange(z, 'B C H W -> B (C H W)')
    clip_img = einops.rearrange(clip_img, 'B L D -> B (L D)')
    text = einops.rearrange(text, 'B L D -> B (L D)')
    return torch.concat([z, clip_img, text], dim=-1)

def split_joint(x):
    C, H, W = 4, 64, 64
    z_dim = C * H * W
    z, clip_img, text = x.split([z_dim, 512, 77 * 64], dim=1)
    z = einops.rearrange(z, 'B (C H W) -> B C H W', C=C, H=H, W=W)
    clip_img = einops.rearrange(clip_img, 'B (L D) -> B L D', L=1, D=512)
    text = einops.rearrange(text, 'B (L D) -> B L D', L=77, D=64)
    return z, clip_img, text

def unpreprocess(v):  # to B C H W and [0, 1]
    v = 0.5 * (v + 1.)
    v.clamp_(0., 1.)
    return v

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watermarking(save_path):
    img_pre = Image.open(save_path)
    img_pos = utils.add_water(img_pre)
    img_pos.save(save_path)

def t2i_nnet(x, timesteps, text):  # text is the low dimension version of the text clip embedding
    """
    1. calculate the conditional model output
    2. calculate unconditional model output
        config.sample.t2i_cfg_mode == 'empty_token': using the original cfg with the empty string
        config.sample.t2i_cfg_mode == 'true_uncond: using the unconditional model learned by our method
    3. return linear combination of conditional output and unconditional output
    """
    z, clip_img = split(x)

    t_text = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

    z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                        data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + data_type)
    x_out = combine(z_out, clip_img_out)

    if cfg_scale == 0.:
        return x_out

    text_N = torch.randn_like(text)  # 3 other possible choices
    z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z, clip_img, text=text_N, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                                  data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + data_type)
    x_out_uncond = combine(z_out_uncond, clip_img_out_uncond)


    return x_out + cfg_scale * (x_out - x_out_uncond)

def i_nnet(x, timesteps):
    z, clip_img = split(x)
    text = torch.randn(x.size(0), 77, 64, device=device)
    t_text = torch.ones_like(timesteps) * N
    z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=t_text,
                          data_type=torch.zeros_like(t_text, device=device, dtype=torch.int) + data_type)
    x_out = combine(z_out, clip_img_out)
    return x_out

def t_nnet(x, timesteps):
    z = torch.randn(x.size(0), *[4, 64, 64], device=device)
    clip_img = torch.randn(x.size(0), 1, 512, device=device)
    z_out, clip_img_out, text_out = nnet(z, clip_img, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                        data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)
    return text_out

def i2t_nnet(x, timesteps, z, clip_img):
    """
    1. calculate the conditional model output
    2. calculate unconditional model output
    3. return linear combination of conditional output and unconditional output
    """
    t_img = torch.zeros(timesteps.size(0), dtype=torch.int, device=device)

    z_out, clip_img_out, text_out = nnet(z, clip_img, text=x, t_img=t_img, t_text=timesteps,
                        data_type=torch.zeros_like(t_img, device=device, dtype=torch.int) + data_type)

    if cfg_scale == 0.:
        return text_out

    z_N = torch.randn_like(z)  # 3 other possible choices
    clip_img_N = torch.randn_like(clip_img)
    z_out_uncond, clip_img_out_uncond, text_out_uncond = nnet(z_N, clip_img_N, text=x, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                                    data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)

    return text_out + cfg_scale * (text_out - text_out_uncond)

def joint_nnet(x, timesteps):
    z, clip_img, text = split_joint(x)
    z_out, clip_img_out, text_out = nnet(z, clip_img, text=text, t_img=timesteps, t_text=timesteps,
                        data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)
    x_out = combine_joint(z_out, clip_img_out, text_out)

    if cfg_scale == 0.:
        return x_out

    z_noise = torch.randn(x.size(0), *(4, 64, 64), device=device)
    clip_img_noise = torch.randn(x.size(0), 1, 512, device=device)
    text_noise = torch.randn(x.size(0), 77, 64, device=device)

    _, _, text_out_uncond = nnet(z_noise, clip_img_noise, text=text, t_img=torch.ones_like(timesteps) * N, t_text=timesteps,
                      data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)
    z_out_uncond, clip_img_out_uncond, _ = nnet(z, clip_img, text=text_noise, t_img=timesteps, t_text=torch.ones_like(timesteps) * N,
                            data_type=torch.zeros_like(timesteps, device=device, dtype=torch.int) + data_type)

    x_out_uncond = combine_joint(z_out_uncond, clip_img_out_uncond, text_out_uncond)

    return x_out + cfg_scale * (x_out - x_out_uncond)


def sample_fn(mode, **kwargs):

    _z_init = torch.randn(n_samples, *(4, 64, 64), device=device)
    _clip_img_init = torch.randn(n_samples, 1, 512, device=device)
    _text_init = torch.randn(n_samples, 77, 64, device=device)
    if mode == 'joint':
        _x_init = combine_joint(_z_init, _clip_img_init, _text_init)
    elif mode in ['t2i', 'i']:
        _x_init = combine(_z_init, _clip_img_init)
    elif mode in ['i2t', 't']:
        _x_init = _text_init
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    def model_fn(x, t_continuous):
        t = t_continuous * N
        if mode == 'joint':
            return joint_nnet(x, t)
        elif mode == 't2i':
            return t2i_nnet(x, t, **kwargs)
        elif mode == 'i2t':
            return i2t_nnet(x, t, **kwargs)
        elif mode == 'i':
            return i_nnet(x, t)
        elif mode == 't':
            return t_nnet(x, t)

    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    with torch.no_grad():
        with torch.autocast(device_type=device):
            x = dpm_solver.sample(_x_init, steps=steps, eps=1. / N, T=1.)

    # os.makedirs(output_path, exist_ok=True)
    if mode == 'joint':
        _z, _clip_img, _text = split_joint(x)
        return _z, _clip_img, _text
    elif mode in ['t2i', 'i']:
        _z, _clip_img = split(x)
        return _z, _clip_img
    elif mode in ['i2t', 't']:
        return x


def get_img_feature(image):
    image = np.array(image).astype(np.uint8)
    image = utils.center_crop(512, 512, image)
    clip_img_feature = clip_img_model.encode_image(clip_img_model_preprocess(Image.fromarray(image)).unsqueeze(0).to(device))
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = einops.rearrange(image, 'h w c -> 1 c h w')
    image = torch.tensor(image, device=device)
    moments = autoencoder.encode_moments(image)
    return clip_img_feature, moments

# __main__

import os
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

mode = 'i2t2i'

# Define the root input directory containing all the classes
input_root_directory = '/home/raza.imam/Documents/Spac/Spac/FG_dataset'
output_root_directory = f'/home/raza.imam/Documents/Spac/Spac/Augmented_{mode}_dataset'
n_samples = 2
target_size = (512, 512)

# Loop through the classes (subdirectories)
for class_name in os.listdir(input_root_directory):
    class_directory = os.path.join(input_root_directory, class_name)
    print(class_directory)
    # Skip non-directory entries
    if not os.path.isdir(class_directory):
        continue

    # Create a corresponding output directory for the class
    output_class_directory = os.path.join(output_root_directory, class_name)
    os.makedirs(output_class_directory, exist_ok=True)

    image_paths = [os.path.join(class_directory, filename) for filename in os.listdir(class_directory) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in tqdm(image_paths):
        assert mode in ['t2i', 'i2t', 'joint', 't', 'i', 't2i2t', 'i2t2i']
        prompt = "an elephant under the sea"
        img = image_path
        seed = np.random.randint(0, 1000001)
        steps = 50
        cfg_scale = 8
        n_samples = 4
        nrow = 2
        data_type = 1

        if mode == 't2i' or mode == 't2i2t':
            prompts = [prompt] * n_samples
            contexts = clip_text_model.encode(prompts)
            contexts_low_dim = caption_decoder.encode_prefix(contexts)
        elif mode == 'i2t' or mode == 'i2t2i':
            img_contexts = []
            clip_imgs = []

        image = Image.open(img).convert('RGB')
        clip_img, img_context = get_img_feature(image)

        img_contexts.append(img_context)
        clip_imgs.append(clip_img)
        img_contexts = img_contexts * n_samples
        clip_imgs = clip_imgs * n_samples

        img_contexts = torch.concat(img_contexts, dim=0)
        z_img = autoencoder.sample(img_contexts)
        clip_imgs = torch.stack(clip_imgs, dim=0)

        set_seed(seed)
        if mode == 't2i': #text-to-image generation
            _z, _clip_img = sample_fn(mode, text=contexts_low_dim)  # conditioned on the text embedding
        elif mode == 'i2t2i': #image-to-image generation
            _text = sample_fn('i2t', z=z_img, clip_img=clip_imgs)  # conditioned on the image embedding
            _z, _clip_img = sample_fn('t2i', text=_text)
        samples = unpreprocess(decode(_z))
        base_filename = os.path.splitext(os.path.basename(img))[0]

        # Resize the input image
        image = image.resize(target_size, Image.ANTIALIAS)

        # Save the input image with its original filename to the class-specific output directory
        input_image_output_path = os.path.join(output_class_directory, f'{base_filename}.png')
        save_image(transforms.ToTensor()(image), input_image_output_path)

        for idx, sample in enumerate(samples):
            save_path = os.path.join(output_class_directory, f'{base_filename}_{idx}.png')
            save_image(sample, save_path)

print("Variations generation and saving completed.")
