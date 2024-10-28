import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Tuple, List, Any

import numpy as np
import cv2
import os
import time

import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import scipy.io as sio

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)


# path_dataset = "../data/SVHN_mat"
# norm_mean = [0.4377, 0.4438, 0.4728]
# norm_std = [0.1980, 0.2010, 0.1970]


class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform_component=None):
        data = sio.loadmat(mat_file)
        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y'].flatten()
        self.labels[self.labels == 10] = 0
        self.transform = transform_component

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


class SmallVGG(nn.Module):
    def __init__(self, frame_size=32):
        super(SmallVGG, self).__init__()
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


def _anti_normalize(img, mean, std):
    """Revert the normalization for visualization."""
    img = img * std + mean
    return np.clip(img, 0, 1)


# Plotting multiple images in a grid
def plot_transformed_img_in_grid(train_dataset: SVHNDataset,
                                 norm_mean: list,
                                 norm_std: list) -> None:
    grid_rows, grid_cols = 1, 6

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(6, 6))

    peak_index = random.randint(0, train_dataset.__len__() - 1)

    for i in range(grid_cols):
        img_tensor, label = train_dataset.__getitem__(peak_index)
        img = img_tensor.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        img = _anti_normalize(img, norm_mean, norm_std)

        ax = axes[i]  # Get subplot axis
        ax.imshow(img)
        ax.set_title(f"Label: {label}")

    plt.tight_layout()
    plt.show()

    print(f"Peaking data from training set of index {peak_index}.\n"
          f"Image Tensor Size:{train_dataset.__getitem__(peak_index)[0].shape}")


def display_epochs_loss_curve(train_losses: List[float],
                              test_losses: List[float]) -> None:
    plt.figure(figsize=(3, 3))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Curves")
    plt.legend()
    plt.show()


def display_confusion_matrix(true_labels: List[int],
                             pred_labels: List[int]) -> None:
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(0, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.show()


def get_metrics(true_labels: List[int],
                pred_labels: List[int]) -> Tuple[float, np.array, np.array, np.array]:
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=1, average=None, labels=range(0, 10))
    recall = recall_score(true_labels, pred_labels, zero_division=1, average=None, labels=range(0, 10))
    f1 = f1_score(true_labels, pred_labels, zero_division=0, average=None, labels=range(0, 10))

    return accuracy, precision, recall, f1


def display_precision_recall_curve(true_labels_bin: np.array,
                                   pred_scores: List[List[float]]) -> None:
    for i in range(0, 10):
        precision_i, recall_i, _ = precision_recall_curve(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        average_precision = average_precision_score(true_labels_bin[:, i], np.array(pred_scores)[:, i])
        plt.step(recall_i, precision_i, where="post", label=f"Class {i} AP={average_precision:.2f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.show()

def train_and_evaluate(model: SmallVGG,
                       train_loader: DataLoader,
                       test_loader: DataLoader,
                       criterion: nn.CrossEntropyLoss,
                       optimizer: optim.Optimizer,
                       num_epochs=100) -> Tuple[List[float], List[float]]:
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
        print(f"Epoch[{epoch + 1}/{num_epochs}], Train Loss:{train_losses[-1]:.4f}, Test Loss:{test_losses[-1]:.4f}")

    return train_losses, test_losses