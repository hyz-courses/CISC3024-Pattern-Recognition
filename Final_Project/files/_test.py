import os

import numpy as np
import scipy.io as sio

from Final_Project.files import dvalue
from Final_Project.files.utils import _add_bias

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

train_data = np.transpose(sio.loadmat(train_path)['X'], (2, 0, 1, 3))

TRAIN_BIAS_norm_mean = []
TRAIN_BIAS_norm_std = []
for bias in dvalue.candidate_channel_biases:
    tmp = _add_bias(train_data.copy(), bias)
    tmp = tmp.astype(np.float32) / 255.0

    TRAIN_BIAS_norm_mean.append([np.mean(x) for x in tmp])
    TRAIN_BIAS_norm_std.append([np.std(x, ddof=0) for x in tmp])

a=1
