import random
from collections import OrderedDict
from modulefinder import Module
from typing import Tuple, Union, List, OrderedDict as TypingOrderedDict

import numpy as np
import scipy.io as sio
import torch.nn as nn
from torch.utils.data import Dataset


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
    def __init__(self, frame_size=32, struct: Tuple[nn.Sequential, nn.Sequential] = None, act_func: Module = None):
        super(SmallVGG, self).__init__()
        if act_func is not None:
            self.activation = act_func
        else:
            self.activation = nn.ReLU()

        if struct is not None:
            self.conv_layers = struct[0]
            self.fc_layers = struct[1]
        else:
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


class AddBiasTransform:
    def __init__(self, bias: Union[int, Tuple[int, int]]) -> None:
        if isinstance(bias, tuple):
            self.bias1 = bias[0]
            self.bias2 = bias[1]
        else:
            self.bias1 = 0
            self.bias2 = bias

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _dtype = img.dtype
        bias_value = random.randint(self.bias1, self.bias2)
        img = (img.astype(np.int16) + bias_value) % 256
        return img.astype(_dtype)


# ==================================================================== #

candidate_seq: List[Tuple[Union[TypingOrderedDict[str, None], TypingOrderedDict[str, nn.Module]],
                          Union[TypingOrderedDict[str, None], TypingOrderedDict[str, nn.Module]]]] = [
    (OrderedDict([  # first struct
        ('conv1', nn.Conv2d(3, 8, kernel_size=3, padding=1)),
        ('*1', None),
        ('conv2', nn.Conv2d(8, 16, kernel_size=3, padding=1)),
        ('*2', None),
        ('max1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16

        ('conv3', nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        ('*3', None),
        ('conv4', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*4', None),
        ('max2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8

        ('conv5', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*5', None),
        ('conv6', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*6', None),
        ('max3', nn.MaxPool2d(kernel_size=2, stride=2))  # 4x4
    ]), OrderedDict([
        ('fc1', nn.Linear(32 * 4 * 4, 256)),
        ('*1', None),
        ('fc2', nn.Linear(256, 10))
    ])),
    (OrderedDict([  # second struct
        ('conv1', nn.Conv2d(3, 8, kernel_size=3, padding=1)),
        ('*1', None),
        ('conv2', nn.Conv2d(8, 16, kernel_size=3, padding=1)),
        ('*2', None),
        ('max1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16

        ('conv3', nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        ('*3', None),
        ('conv4', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*4', None),
        ('max2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8

        ('conv5', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*5', None),
        ('conv6', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*6', None),
        ('max3', nn.MaxPool2d(kernel_size=2, stride=2))  # 4x4
    ]), OrderedDict([
        ('fc1', nn.Linear(32 * 4 * 4, 256)),
        ('*1', None),
        ('fc2', nn.Linear(256, 10))
    ])),
    (OrderedDict([  # third struct
        ('conv1', nn.Conv2d(3, 8, kernel_size=3, padding=1)),
        ('*1', None),
        ('conv2', nn.Conv2d(8, 16, kernel_size=3, padding=1)),
        ('*2', None),
        ('max1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16

        ('conv3', nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        ('*3', None),
        ('conv4', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*4', None),
        ('max2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8

        ('conv5', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*5', None),
        ('conv6', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*6', None),
        ('max3', nn.MaxPool2d(kernel_size=2, stride=2))  # 4x4
    ]), OrderedDict([
        ('fc1', nn.Linear(32 * 4 * 4, 256)),
        ('*1', None),
        ('fc2', nn.Linear(256, 10))
    ])),
    (OrderedDict([  # fourth struct
        ('conv1', nn.Conv2d(3, 8, kernel_size=3, padding=1)),
        ('*1', None),
        ('conv2', nn.Conv2d(8, 16, kernel_size=3, padding=1)),
        ('*2', None),
        ('max1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16

        ('conv3', nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        ('*3', None),
        ('conv4', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*4', None),
        ('max2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8

        ('conv5', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*5', None),
        ('conv6', nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        ('*6', None),
        ('max3', nn.MaxPool2d(kernel_size=2, stride=2))  # 4x4
    ]), OrderedDict([
        ('fc1', nn.Linear(32 * 4 * 4, 256)),
        ('*1', None),
        ('fc2', nn.Linear(256, 10))
    ]))
]

candidate_activation_func: List[nn.Module] = [nn.ReLU(), nn.ELU(), nn.LeakyReLU(), nn.SiLU()]
