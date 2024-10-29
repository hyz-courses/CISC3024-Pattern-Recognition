import os
from typing import Dict, Union

import torch

import dvalue
from Final_Project.files import dstruct
from Final_Project.files.dvalue import candidate_drop_rate
from dstruct import (SVHNDataset)

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)

path_dataset = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"
norm_mean = dvalue.FULL_BIAS_norm_mean[0]
norm_std = dvalue.FULL_BIAS_norm_std[0]

exp3_train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"), transform_component=None)
exp3_test_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "test_32x32.mat"), transform_component=None)
# exp3_extra_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "extra_32x32.mat"), transform_component=None)

# ==================================================================== #

# Group 1
candidate_seq = dstruct.candidate_seq
candidate_activation_func = dstruct.candidate_activation_func

exp4_1_hyperparams: Dict[str, Union[int, float]] = dict(num_epochs=15, lr=0.001,
                                                        angle=45, crop=0.08,
                                                        ratio=0.75, bias_idx=0) #TODO

# ==================================================================== #

# Group 2
candidate_drop_rate = dvalue.candidate_drop_rate
candidate_batch_size = dvalue.candidate_batch_size

exp4_2_hyperparams: Dict[str, Union[int, float]] = dict(num_epochs=15, lr=0.001,
                                                        angle=45, crop=0.08,
                                                        ratio=0.75, bias_idx=0,
                                                        seq_idx=0, act_idx=0) #TODO

# ==================================================================== #
