import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torchvision.models as models
import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import numpy as np
# from clip import clip, tokenize
from torch.optim.lr_scheduler import StepLR


# Load the CLIP model
gpu_id = 2  # Replace this with the GPU ID you want to switch to
device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a common size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

dataset = torchvision.datasets.ImageFolder(root='/home/mohammed.alam/Documents/Tortnake/PlanetX/HR_space_data_aug', transform=transform)
train_size = int(0.7 * len(dataset))
valid_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - valid_size

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, valid_size, test_size])
 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_classes = 8

# Replace the final fully connected layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

model.to(device)

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=1, gamma=2, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha  # Weighting factor for class imbalance
#         self.gamma = gamma  # Focusing parameter
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # Compute softmax over the inputs
#         softmax = torch.nn.functional.softmax(inputs, dim=1)

#         # Calculate log-softmax probabilities
#         log_softmax = torch.nn.functional.log_softmax(inputs, dim=1)

#         # Gather the log probabilities at the true class labels
#         log_probs = log_softmax.gather(1, targets.view(-1, 1))

#         # Calculate the focal loss
#         loss = -self.alpha * (1 - softmax.gather(1, targets.view(-1, 1)))**self.gamma * log_probs

#         if self.reduction == 'none':
#             return loss
#         elif self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
            
# focal_loss = FocalLoss(alpha=1, gamma=2, reduction='mean')

# Create empty lists to store training and validation losses
train_losses = []
valid_losses = []
valid_accuracies = []
valid_f1_scores = []
confusion_matrices = []

# Initialize variables for early stopping
best_valid_loss = float("inf")
patience = 50  # Number of epochs with no improvement after which to stop
wait = 0

num_epochs = 50

# Set the directory where you want to save the model checkpoints
save_dir = "model_checkpoints_resnet18/"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Initialize variables for confusion matrix
# num_classes = 8  # Replace with the actual number of classes
all_preds = []
all_labels = []

# Initialize variables for saving checkpoints
save_interval = 5
epoch_counter = 0

# Define the step learning rate scheduler
step_lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model in training mode
    running_loss = 0.0

    # Use tqdm for a progress bar
    with tqdm(train_loader, unit="batch") as t:
        for inputs, labels in t:
            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.shape)
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss = focal_loss(outputs, labels)
            loss.backward()
            optimizer.step()  # Update model weights

            running_loss += loss.item()

            t.set_postfix(loss=running_loss / len(t))  # Display loss in the progress bar
    
    # Step the learning rate
    step_lr_scheduler.step()

    # Calculate the average training loss for this epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()  # Set the model in evaluation mode
    validation_loss = 0.0

    # Use no_grad() to prevent gradient computation during validation
    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # loss = focal_loss(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for the confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        validation_loss /= len(valid_loader)
        valid_losses.append(validation_loss)
        accuracy = 100 * correct / total
        valid_accuracies.append(accuracy)

        f1 = f1_score(all_labels, all_preds, average='weighted')
        valid_f1_scores.append(f1)

        confusion = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
        confusion_matrices.append(confusion)

        print(f"Epoch {epoch + 1}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation F1 Score: {f1:.4f}")

    # Early stopping logic
    if validation_loss < best_valid_loss:
        best_valid_loss = validation_loss
        wait = 0
        # Save the model weights when validation loss improves
        checkpoint_path = os.path.join(save_dir, f"model_epoch{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    # #  Save the model weights every 'save_interval' epochs
    # if (epoch_counter + 1) % save_interval == 0:
    #     checkpoint_path = os.path.join(save_dir, f"model_epoch{epoch + 1}.pt")
    #     torch.save(model.state_dict(), checkpoint_path)
    #     epoch_counter += 1    

print("Training complete.")