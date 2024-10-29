from typing import Tuple, List

TA_norm_mean: List[float] = [0.4377, 0.4438, 0.4728]
"""norm_mean prepared by TA of CISC-3024"""

TA_norm_std: List[float] = [0.1980, 0.2010, 0.1970]
"""norm_std prepared by TA of CISC-3024"""

FULL_BIAS_norm_mean: Tuple[List[float], List[float], List[float], List[float]] = (
    [0.43011287, 0.42947713, 0.44553235], [0.482172, 0.47879672, 0.49193108],
    [0.5299231, 0.52622163, 0.5367366], [0.57701516, 0.57407594, 0.5768523])
"""norm_mean of all pixel values in 3 mat files in folder "SVHN_mat" """

FULL_BIAS_norm_std: Tuple[List[float], List[float], List[float], List[float]] = (
    [0.1968019, 0.19883512, 0.19997141], [0.19747807, 0.19839704, 0.1997657],
    [0.2061324, 0.20671913, 0.20880404], [0.2491606, 0.24821031, 0.2517171])
"""norm_std of all pixel values in 3 mat files in folder "SVHN_mat" """

TRAIN_BIAS_norm_mean: Tuple[List[float], List[float], List[float], List[float]] = (
    [0.4359728, 0.4420371, 0.47095722], [0.48945808, 0.49221805, 0.51708484],
    [0.5346211, 0.53527725, 0.55649185], [0.25027722, 0.25132284, 0.25705674])
"""norm_mean of only "train_32x32.mat" """

TRAIN_BIAS_norm_std: Tuple[List[float], List[float], List[float], List[float]] = (
    [0.19725639, 0.20023046, 0.1962663], [0.19868314, 0.20101954, 0.19824031],
    [0.207664, 0.20977464, 0.20946072], [0.5766456, 0.5753719, 0.5847802])
"""norm_std of only "train_32x32.mat" """

# ==================================================================== #

candidate_angles = [15, 30, 45, 60]
candidate_crops = [0.08, 0.24, 0.40, 0.60]  # Left Boundary

candidate_ratios = [0.25, 0.42, 0.58, 0.75]  # Left Boundary
candidate_channel_biases = [0, 32, 64, 128]

candidate_batch_size = [64, 128, 256, 512]

candidate_drop_rate = [0.1, 0.23, 0.37, 0.5]