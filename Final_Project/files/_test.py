import os
import time
from typing import List, Union, Dict

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import dvalue
from dstruct import (SVHNDataset, SmallVGG, AddBiasTransform)
from utils import (train_and_evaluate)

path_dataset = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"

# ================= #
