import os

import torch.nn as nn
from collections import OrderedDict
from typing import OrderedDict as TypingOrderedDict
import random
from collections import OrderedDict
from modulefinder import Module
from typing import Tuple, Union, List, OrderedDict as TypingOrderedDict

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset

# from Final_Project.files import utils, dstruct
#
# selected_seq = dstruct.candidate_seq[0]
# selected_act = dstruct.candidate_activation_func[0]
#
# final_seq = utils.mix_seq_and_act(selected_seq, selected_act)
#
# print(final_seq)


dir_path = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"
train_path = os.path.join(dir_path, "train_32x32.mat")
test_path = os.path.join(dir_path, "test_32x32.mat")

x = sio.loadmat(train_path)['X']
y = sio.loadmat(test_path)['X']

train_data = np.transpose(x, (2, 0, 1, 3)).astype(np.float32) / 256.0
test_data = np.transpose(y, (2, 0, 1, 3)).astype(np.float32) / 256.0

both_data = np.concatenate((train_data, test_data), axis=3)

BOTH_norm_mean = [np.mean(x) for x in both_data]
BOTH_norm_std = [np.std(x, ddof=0) for x in both_data]


