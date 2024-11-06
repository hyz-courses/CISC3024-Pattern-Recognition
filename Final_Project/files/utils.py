import copy
import random
from typing import Tuple, List, Union, OrderedDict as TypingOrderedDict
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, precision_recall_curve,
                             average_precision_score)
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dstruct import (SVHNDataset, SmallVGG)

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)


# ============================== Split ============================= #
def split_train_valid(train_dataset, train_ratio):
    ori_len = len(train_dataset)
    train_size = int(train_ratio * ori_len)
    valid_size = ori_len - train_size

    # These are subsets!! Don't directly use them, or you will spend 2 hours solving for it.
    train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

    # Re-construct two SVHNDataset object from indices
    train_dataset_ = train_dataset.overwrite(indices=train_subset.indices)
    valid_dataset_ = train_dataset.overwrite(indices=valid_subset.indices)

    return train_dataset_, valid_dataset_


# ============================== For Plotting ============================= #
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


# ================================ main.py ================================ #
def train_and_evaluate(model,
                       train_loader,
                       valid_loader,
                       criterion,
                       optimizer,
                       num_epochs=100,
                       stop_early_params=None):
    # Record Losses to plot
    train_losses = []
    valid_losses = []

    # Early stop params
    current_optimized_model = None
    current_min_valid_loss = np.inf
    num_overfit_epochs = 0

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
        valid_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * len(images)

        valid_losses.append(valid_loss / len(valid_loader))
        print(
            f"Epoch[{epoch + 1}/{num_epochs}], Train Loss:{train_losses[-1]:.4f}, Validation Loss:{valid_losses[-1]:.4f}")

        # Early Stop?
        if stop_early_params is None:
            continue

        if current_min_valid_loss - stop_early_params["min_delta"] > valid_losses[-1]:  # Validation loss decreases
            current_min_valid_loss = valid_losses[-1]
            current_optimized_model = copy.deepcopy(model)
            num_overfit_epochs = (num_overfit_epochs - 1) if num_overfit_epochs > 0 else 0
        else:  # Validation loss increases
            num_overfit_epochs += 1

        if num_overfit_epochs > stop_early_params["patience"]:
            print(f"Early stopping at epoch {epoch + 1}.")
            model = current_optimized_model
            break

    return train_losses, valid_losses


# ============================= experiment3.py ============================= #

def contrast(data: np.array,
             factor: Union[float, Tuple[float, float]],
             seed=114514) -> np.array:
    random.seed(seed)
    if isinstance(factor, tuple):
        factor_min = factor[0]
        factor_max = factor[1]
    else:
        factor_min = 1 / factor
        factor_max = factor

    _dtype = data.dtype

    data = data.astype(np.float64)

    for i in range(len(data)):
        contrast_factor = random.uniform(factor_min, factor_max)
        img = data[i] * contrast_factor
        data[i] = np.clip(img, 0, 255)  # apply contrast enhancement

    return data.astype(_dtype)


# ============================= experiment4.py ============================= #

def mix_seq_and_act(seq: Tuple[TypingOrderedDict, TypingOrderedDict],
                    activation_func: nn.Module) -> Tuple[nn.Sequential, nn.Sequential]:
    """
    replace all layers whose names start with '*' to the selected activation function
    """
    conv_seq = seq[0].copy()
    for name, module in conv_seq.items():
        if name.startswith('*'):
            conv_seq[name] = activation_func

    fc_seq = seq[1].copy()
    for name, module in fc_seq.items():
        if name.startswith('*'):
            fc_seq[name] = activation_func

    return nn.Sequential(conv_seq), nn.Sequential(fc_seq)
