import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

import numpy as np
import cv2
import os
import time

import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.io as sio

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
print(f"Using device: {device_name}")

path_dataset = "./data/SVHN_mat"
norm_mean = [0.4377, 0.4438, 0.4728]
norm_std = [0.1980, 0.2010, 0.1970]

class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        data = sio.loadmat(mat_file)
        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y'].flatten()
        self.labels[self.labels == 10] = 0
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label

transform = A.Compose([
    A.RandomResizedCrop(32, 32),
    A.Rotate(limit=30),
    A.Normalize(mean=norm_mean, std=norm_std),
    ToTensorV2()
])

train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"), transform=transform)
test_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "test_32x32.mat"), transform=transform)
extra_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "extra_32x32.mat"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
extra_loader = DataLoader(Subset(extra_dataset, indices=list(range(30000))), batch_size=64, shuffle=False)

# def unnormalize(img, mean, std):
#     """Revert the normalization for visualization."""
#     img = img * std + mean
#     return np.clip(img, 0, 1)
#
# # Plotting multiple images in a grid
# grid_rows, grid_cols = 1, 6
#
# fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(6, 6))
#
# peak_index = random.randint(0, train_dataset.__len__()-1)
#
# for i in range(grid_cols):
#     img_tensor, label = train_dataset.__getitem__(peak_index)
#     img = img_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
#     img = unnormalize(img, norm_mean, norm_std)
#
#     ax = axes[i]  # Get subplot axis
#     ax.imshow(img)
#     ax.set_title(f"Label: {label}")
#
# plt.tight_layout()
# plt.show()

# print(f"Peaking data from training set of index {peak_index}.\nImage Tnesor Size:{train_dataset.__getitem__(peak_index)[0].shape}")


class SmallVGG(nn.Module):
    def __init__(self, frame_size=32):
        super(SmallVGG, self).__init__()
        self.frame_size = frame_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(frame_size * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

num_epochs = 30
learning_rate = 0.001
model = SmallVGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_and_evaluate(model,
                      train_loader,
                      test_loader,
                      criterion,
                      optimizer,
                      num_epochs=100):
    # Record Losses to plot
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(images)
        train_losses.append(running_loss / len(train_loader))

        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * len(images)

        test_losses.append(test_loss / len(test_loader))
        print(f"Epoch[{epoch+1}/{num_epochs}], Train Loss:{train_losses[-1]:.4f}, Test Loss:{test_losses[-1]:.4f}")

    return train_losses, test_losses

train_losses, test_losses = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), f"./models/small_vgg_ne-{num_epochs}_lr-{learning_rate:.0e}.pth")

from sklearn.metrics import (confusion_matrix, accuracy_score,
                            precision_score, recall_score,
                            f1_score, roc_auc_score,
                            roc_curve, precision_recall_curve,
                            average_precision_score)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize


def get_predictions(model_path, extra_loader):
    model_state = torch.load(model_path)
    model = SmallVGG()
    model.load_state_dict(model_state)

    model.to(device)
    model.eval()

    pred_scores = []  # Prob. of predictions
    true_labels = []  # Ground Truth
    pred_labels = []  # Label of prediction, i.e., argmax(softmax(pred_scores))

    with torch.no_grad():
        for images, labels in tqdm(extra_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            pred_scores_batch = nn.functional.softmax(outputs, dim=-1)

            pred_scores.extend(pred_scores_batch.cpu().tolist())
            pred_labels.extend(outputs.argmax(dim=1).tolist())
            true_labels.extend(labels.cpu().tolist())

    return pred_scores, true_labels, pred_labels

pred_scores, true_labels_cpu, pred_labels_cpu = get_predictions("./models/small_vgg_ne-30_lr-1e-03.pth", extra_loader)
print("First 100 true labels:")
[print(num, end=" ") for num in true_labels_cpu[:100]]
print("...\n")

print("First 100 predictions:")
[print(num, end=" ") for num in pred_labels_cpu[:100]]
print("...\n")

print("Prediction Probabilities:")
[print(arr) for arr in pred_scores[:5]]
print("...")

def display_el_curve(train_losses, test_losses):
    plt.figure(figsize=(3,3))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Curves")
    plt.legend()
    plt.show()

display_el_curve(train_losses, test_losses)

def display_cm(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(0,10))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

display_cm(true_labels_cpu, pred_labels_cpu)

true_labels_bin = label_binarize(true_labels_cpu, classes=range(0,10))
true_labels_bin

pred_labels_bin = label_binarize(pred_labels_cpu, classes=range(0,10))
pred_labels_bin

def get_metrics(true_labels, pred_labels):
    accuracy = accuracy_score(true_labels_cpu, pred_labels_cpu)
    precision = precision_score(true_labels_cpu, pred_labels_cpu, average=None, labels=range(0, 10))
    recall = recall_score(true_labels_cpu, pred_labels_cpu, average=None, labels=range(0, 10))
    f1 = f1_score(true_labels_cpu, pred_labels_cpu, average=None, labels=range(0, 10))

    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = get_metrics(true_labels_cpu, pred_labels_cpu)
print(f"Accuracy:{accuracy:.2f}")
for i in range(10):
    print(f"Class {i}: Prec:{precision[i]:.2f}, Recall:{recall[i]:.2f}, F_1 Score:{f1[i]:.2f}")


def display_pr_curve(true_labels_bin, pred_scores):
    for i in range(0, 10):
        precision_i, recall_i, _ = precision_recall_curve(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        average_precision = average_precision_score(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        plt.step(recall_i, precision_i, where="post", label=f"Class {i} AP={average_precision:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.show()

display_pr_curve(true_labels_bin, pred_scores)