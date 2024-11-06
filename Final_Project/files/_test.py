import os
import random
from typing import Union, Tuple

import numpy as np

from Final_Project.files.dstruct import SVHNDataset

path_dataset = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"
tmp = SVHNDataset(os.path.join(path_dataset, "train_32x32.mat"), transform=None)


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


a = contrast(tmp.images, 3, 114514)

b = contrast(tmp.images, 3, 114514)

print(a)
print(b)
