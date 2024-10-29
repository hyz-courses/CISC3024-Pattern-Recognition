import torch.nn as nn
from collections import OrderedDict
from typing import OrderedDict as TypingOrderedDict

from Final_Project.files import utils, dstruct

selected_seq = dstruct.candidate_seq[0]
selected_act = dstruct.candidate_activation_func[0]

final_seq = utils.mix_seq_and_act(selected_seq, selected_act)

print(final_seq)