import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sympy.physics.units.definitions.dimension_definitions import angle
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from typing import Tuple, List, Any, Union
from PIL import Image

import numpy as np
import cv2
import os
import time

import random
import scipy.io as sio

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

from utils import (SmallVGG, SVHNDataset, plot_transformed_img_in_grid, display_epochs_loss_curve,
                   display_confusion_matrix, get_metrics, display_precision_recall_curve,
                   train_and_evaluate, add_bias)

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)

path_dataset = "../data/SVHN_mat"
norm_mean = [0.4377, 0.4438, 0.4728]
norm_std = [0.1980, 0.2010, 0.1970]

# ==================================================================== #

# Group 1
candidate_angles = [15, 30, 45, 60]
candidate_crops = [0.08, 0.24, 0.40, 0.60]  # Left Boundary
exp3_1_hyperparams = dict(num_epochs=15, lr=0.001)


def run_exp3_1(angles: List[float], crops: List[float], hyperparams):
    experiments = []
    cnt = 1
    for _angle in angles:
        for _crop in crops:
            print(f"Experiment {cnt}. Running experiment on angle: {_angle} with crop size: {_crop}")
            cnt += 1

            _transform = transforms.Compose([
                A.RandomResizedCrop(32, 32, scale=(_crop, 1.0)),
                A.Rotate(limit=_angle),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2()
            ])

            train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"),
                                        transform_component=_transform)
            test_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "test_32x32.mat"),
                                       transform_component=_transform)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            num_epochs = hyperparams['num_epochs']
            learning_rate = hyperparams['lr']
            exp3_1_model = SmallVGG().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(exp3_1_model.parameters(), lr=learning_rate)

            train_losses, test_losses = train_and_evaluate(exp3_1_model, train_loader, test_loader, criterion,
                                                           optimizer,
                                                           num_epochs)
            experiments.append({
                "angle": _angle,
                "crop": _crop,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "model_state_dict": exp3_1_model.state_dict()
            })

            del exp3_1_model, criterion, optimizer
            del train_dataset, test_dataset, train_loader, test_loader
            torch.cuda.empty_cache()

        return experiments


# exp3_1 = run_exp3_1(candidate_angles, candidate_crops, exp3_1_hyperparams)
# time_str = str(time.time()).replace(".", "")
# torch.save(exp3_1, f"./models/exp3_1_{time_str}.pth")

# ==================================================================== #
# Group 2
candidate_ratios = [0.25, 0.42, 0.58, 0.75]  # Left Boundary
candidate_channel_biases = [0, 32, 64, 128]

a = sio.loadmat("../data/SVHN_mat/train_32x32.mat")['X']
b = sio.loadmat("../data/SVHN_mat/test_32x32.mat")['X']
c = sio.loadmat("../data/SVHN_mat/extra_32x32.mat")['X']
abc = np.concatenate((a, b, c), axis=3)
abc = np.transpose(abc, (3, 0, 1, 2)).astype(np.float32)

norm_mean_bias = []
norm_std_bias = []

for bias in tqdm(candidate_channel_biases):
    abc_bias = np.transpose(add_bias(abc.copy(), bias=bias), (3, 0, 1, 2))

    norm_mean_bias.append([np.mean(abc_bias[0]), np.mean(abc_bias[1]), np.mean(abc_bias[2])])
    norm_std_bias.append([np.std(abc_bias[0], ddof=-1), np.std(abc_bias[1], ddof=-1), np.std(abc_bias[2], ddof=-1)])

for (x, y) in zip(norm_mean_bias, norm_std_bias):
    print(x)
    print(y)
    print("\n")

# exp3_2_hyperparams = dict(num_epochs=15, lr=0.001, angle=45, crop=0.6)  # TODO
#
#
# def run_exp3_2(ratios: List[float], biases: List[int], hyperparams):
#     experiments = []
#     cnt = 1
#     for _ratio in ratios:
#         for _bias in biases:
#             print(f"Experiment {cnt}. Running experiment on ratio: {_ratio} with bias: {_bias}")
#             cnt += 1
#
#             _transform = transforms.Compose([
#                 A.RandomResizedCrop(32, 32, scale=(hyperparams['crop'], 1.0), ratio=(_ratio, 1 / _ratio)),
#                 A.Rotate(limit=hyperparams['angle']),
#                 A.Normalize(mean=norm_mean, std=norm_std),
#                 ToTensorV2()
#             ])
#
#             train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"),
#                                         transform_component=_transform)
#             test_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "test_32x32.mat"),
#                                        transform_component=_transform)
#             train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
#             test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
#
#             num_epochs = hyperparams['num_epochs']
#             learning_rate = hyperparams['lr']
#             exp3_1_model = SmallVGG().to(device)
#             criterion = nn.CrossEntropyLoss()
#             optimizer = optim.Adam(exp3_1_model.parameters(), lr=learning_rate)
#
#             train_losses, test_losses = train_and_evaluate(exp3_1_model, train_loader, test_loader, criterion,
#                                                            optimizer,
#                                                            num_epochs)
#             experiments.append({
#                 "angle": angles,
#                 "crop": crops,
#                 "train_losses": train_losses,
#                 "test_losses": test_losses,
#                 "model_state_dict": exp3_1_model.state_dict()
#             })
#
#             del exp3_1_model, criterion, optimizer, train_dataset, test_dataset, train_loader, test_loader
#             torch.cuda.empty_cache()
#
#         return experiments

# ==================================================================== #
