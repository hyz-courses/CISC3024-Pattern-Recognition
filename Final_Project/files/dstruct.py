import copy
import random
from collections import OrderedDict
from typing import Tuple, Union, List, OrderedDict as TypingOrderedDict, Optional

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        data = sio.loadmat(mat_file)

        self.images = np.transpose(data['X'], (3, 0, 1, 2))
        self.labels = data['y'].flatten()
        self.labels[self.labels == 10] = 0
        self.transform = transform  # Allow postponed injection of transform.

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # There should always be a transform.
        # It converts image to float, and permutes it from (32, 32, 3) to Tensor([3, 32, 32]).
        # ...which is important!!
        if self.transform is None:
            raise ValueError(
                "CISC3024 Custom Error: The transform should not be None when this object is passed into a DataLoader.")

        image = self.transform(image=image)['image']
        return image, label

    def get_meanstd(self, contrast_factor=None):
        if contrast_factor is not None:
            random_cf = random.uniform(1 / contrast_factor, contrast_factor)
            images_ = []
            for i in range(len(self.images)):
                image = self.images[i]
                image = np.clip(image.astype(np.float32) * random_cf, 0.0, 255.0)
                images_.append(image.astype(np.uint8))
            images_ = np.array(images_)
        else:
            images_ = self.images

        images_ = images_.astype(np.float32) / 255.0

        # mean = np.mean(images_, axis=(0, 1, 2))
        # std = np.std(images_, axis=(0, 1, 2), ddof=0)

        images_ = np.transpose(images_, (3, 0, 1, 2))
        mean = [np.mean(x) for x in images_]
        std = [np.std(x, ddof=0) for x in images_]

        return mean, std
        # return mean.tolist(), std.tolist()

    def overwrite(self, indices: Union[list, np.ndarray]):
        """
        Create a deep copy of the mother dataset instance and only keep the wanted
        data samples, controlled by indices.
        """
        if any(index < 0 or index >= len(self.labels) for index in indices):
            raise IndexError("CISC3024 Custom Error: One or more indices are out of bounds.")

        new_dataset = copy.deepcopy(self)
        new_dataset.images = self.images[indices]
        new_dataset.labels = self.labels[indices]
        return new_dataset


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


# class AddBiasTransform:
#     def __init__(self, bias: Union[int, Tuple[int, int]]) -> None:
#         if isinstance(bias, tuple):
#             self.bias1 = bias[0]
#             self.bias2 = bias[1]
#         else:
#             self.bias1 = 0
#             self.bias2 = bias
#
#     def __call__(self, img: np.ndarray) -> np.ndarray:
#         _dtype = img.dtype
#         bias_value = random.randint(self.bias1, self.bias2)
#         img = (img.astype(np.int16) + bias_value) % 256
#         return img.astype(_dtype)


class ContrastEnhanceTransform:
    def __init__(self, factor: Union[float, Tuple[float, float]]) -> None:
        if isinstance(factor, tuple):
            self.factor_min = factor[0]
            self.factor_max = factor[1]
        else:
            self.factor_min = 1 / factor  # TODO
            self.factor_max = factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _dtype = img.dtype
        contrast_factor = random.uniform(self.factor_min, self.factor_max)
        img = np.clip(img * contrast_factor, 0, 255)  # apply contrast enhancement
        return img.astype(_dtype)


# ==================================================================== #

class Inception(nn.Module):
    def __init__(self, in_channels: int, ch1x1: int, ch3x3_reduce: int, ch3x3: int,
                 ch5x5_reduce: int, ch5x5: int, pool_proj: int):
        super(Inception, self).__init__()

        # 1x1 conv batch
        self.branch1x1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3 conv batch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5 conv batch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        # 3x3 pool -> 1x1 conv batch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], 1)
        return outputs


candidate_seq: List[Tuple[TypingOrderedDict[str, Optional[nn.Module]], TypingOrderedDict[str, Optional[nn.Module]]]] = [
    (OrderedDict([  # first struct: SmallVGG
        ('conv1', nn.Conv2d(3, 8, kernel_size=3, padding=1)),
        ('*1', None),
        ('conv2', nn.Conv2d(8, 16, kernel_size=3, padding=1)),
        ('*2', None),
        ('max1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16

        ('conv3', nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        ('*3', None),
        ('conv4', nn.Conv2d(32, 48, kernel_size=3, padding=1)),
        ('*4', None),
        ('max2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8

        ('conv5', nn.Conv2d(48, 56, kernel_size=3, padding=1)),
        ('*5', None),
        ('conv6', nn.Conv2d(56, 64, kernel_size=3, padding=1)),
        ('*6', None),
        ('max3', nn.MaxPool2d(kernel_size=2, stride=2))  # 4x4
    ]), OrderedDict([
        ('fc1', nn.Linear(64 * 4 * 4, 512)),
        ('*1', None),
        ('fc2', nn.Linear(512, 256)),
        ('*2', None),
        ('fc3', nn.Linear(256, 10))
    ])),

    (OrderedDict([  # second struct: LeNet-5
        ('conv1', nn.Conv2d(3, 12, kernel_size=5, stride=1, padding=2)),
        ('*1', None),
        ('avg1', nn.AvgPool2d(kernel_size=2, stride=2)),

        ('conv2', nn.Conv2d(12, 32, kernel_size=5)),
        ('*2', None),
        ('avg2', nn.AvgPool2d(kernel_size=2, stride=2)),
    ]), OrderedDict([
        ('fc1', nn.Linear(32 * 6 * 6, 256)),
        ('*3', None),
        ('fc2', nn.Linear(256, 128)),
        ('*4', None),
        ('fc3', nn.Linear(128, 10))
    ])),

    (OrderedDict([  # third struct: 2012AlexNet
        ('conv1', nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)),  # 32x32
        ('*1', None),
        ('max1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16

        ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
        ('*2', None),
        ('max2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8

        ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
        ('*3', None),

        ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
        ('*4', None),

        ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ('*5', None),
        ('max3', nn.MaxPool2d(kernel_size=2, stride=2))  # 4x4
    ]), OrderedDict([
        ('fc1', nn.Linear(256 * 4 * 4, 4096)),  # 256 * 4 * 4 = 4096
        ('*6', None),
        ('dropout1', nn.Dropout()),

        ('fc2', nn.Linear(4096, 4096)),
        ('*7', None),
        ('dropout2', nn.Dropout()),

        ('fc3', nn.Linear(4096, 10))
    ])),

    (OrderedDict([  # fourth struct: 2014GoogLeNet
        ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)),
        ('max1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('conv2', nn.Conv2d(64, 64, kernel_size=1)),
        ('conv3', nn.Conv2d(64, 192, kernel_size=3, padding=1)),
        ('max2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        # Inception 模块
        ('inception3a', Inception(192, 64, 96, 128, 16, 32, 32)),
        ('inception3b', Inception(256, 128, 128, 192, 32, 96, 64)),
        ('max3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('inception4a', Inception(480, 192, 96, 208, 16, 48, 64)),
        ('inception4b', Inception(512, 160, 112, 224, 24, 64, 64)),
        ('inception4c', Inception(512, 128, 128, 256, 24, 64, 64)),
        ('inception4d', Inception(512, 112, 144, 288, 32, 64, 64)),
        ('inception4e', Inception(528, 256, 160, 320, 32, 128, 128)),
        ('max4', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ('inception5a', Inception(832, 256, 160, 320, 32, 128, 128)),
        ('inception5b', Inception(832, 384, 192, 384, 48, 128, 128)),

        ('avg1', nn.AdaptiveAvgPool2d((1, 1))),
        ('dropout1', nn.Dropout(0.4)),
    ]), OrderedDict([
        ('fc1', nn.Linear(1024, 10))
    ]))
]
candidate_seq_name = ['SmallVGG', 'LeNet-5', '2012AlexNet', '2014GoogLeNet', ]

candidate_activation_func: List[nn.Module] = [nn.ReLU(), nn.ELU(), nn.LeakyReLU(), nn.SiLU()]
