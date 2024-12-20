import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sympy.physics.units.definitions.dimension_definitions import angle
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Tuple, List, Any, Union, Dict
from PIL import Image

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

from utils import (plot_transformed_img_in_grid, display_epochs_loss_curve,
                   display_confusion_matrix, get_metrics, display_precision_recall_curve,
                   train_and_evaluate)
import dvalue
from dstruct import (SVHNDataset, SmallVGG, AddBiasTransform)

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)
print(f"Using device: {device_name}")

path_dataset = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"
norm_mean = [0.4377, 0.4438, 0.4728]
norm_std = [0.1980, 0.2010, 0.1970]

transform = A.Compose([
    # A.RandomResizedCrop(32, 32),
    A.RandomResizedCrop(32, 32, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
    A.Rotate(limit=45),
    A.Normalize(mean=norm_mean, std=norm_std),
    ToTensorV2()
])

train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"), transform=transform)
test_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "test_32x32.mat"), transform=transform)
extra_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "extra_32x32.mat"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
extra_loader = DataLoader(Subset(extra_dataset, indices=list(range(30000))), batch_size=64, shuffle=False)

print(f"Train Size:{train_dataset.__len__()}\nTest Size:{test_dataset.__len__()}\nExtra Size:{extra_dataset.__len__()}")

plot_transformed_img_in_grid(train_dataset, norm_mean, norm_std)  # utils

num_epochs = 3
learning_rate = 0.001
model = SmallVGG().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, test_losses = train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs)
torch.save(model.state_dict(), f"../models/small_vgg_ne-{num_epochs}_lr-{learning_rate:.0e}.pth")


def get_predictions(model_path: str, extra_loader: DataLoader) -> Tuple[List[List[float]], List[int], List[int]]:
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


pred_scores, true_labels_cpu, pred_labels_cpu = get_predictions(
    f"../models/small_vgg_ne-{num_epochs}_lr-{learning_rate:.0e}.pth", extra_loader)

display_epochs_loss_curve(train_losses, test_losses)  # utils

display_confusion_matrix(true_labels_cpu, pred_labels_cpu)  # utils

true_labels_bin = label_binarize(true_labels_cpu, classes=range(0, 10))

pred_labels_bin = label_binarize(pred_labels_cpu, classes=range(0, 10))

accuracy, precision, recall, f1 = get_metrics(true_labels_cpu, pred_labels_cpu)  # utils
print(f"Accuracy:{accuracy:.2f}")
for i in range(10):
    print(f"Class {i}: Precision:{precision[i]:.2f}, Recall:{recall[i]:.2f}, F_1 Score:{f1[i]:.2f}")

display_precision_recall_curve(true_labels_bin, pred_scores)  # utils
