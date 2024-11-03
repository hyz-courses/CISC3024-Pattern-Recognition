from typing import Tuple, List

TA_norm_mean: List[float] = [0.4377, 0.4438, 0.4728]
"""norm_mean prepared by TA of CISC-3024"""

TA_norm_std: List[float] = [0.1980, 0.2010, 0.1970]
"""norm_std prepared by TA of CISC-3024"""

# ==================================================================== #

candidate_angles = [15, 30, 45, 60]
candidate_crops = [0.08, 0.24, 0.40, 0.60]  # Left Boundary

candidate_ratios = [0.25, 0.42, 0.58, 0.75]  # Left Boundary
candidate_channel_biases = [0, 32, 64, 128]

candidate_batch_size = [64, 128, 256, 512]

candidate_drop_rate = [0.1, 0.23, 0.37, 0.5]
