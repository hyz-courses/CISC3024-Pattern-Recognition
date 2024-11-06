import itertools
import os
import time
from typing import Dict, Union, List, Tuple, OrderedDict as TypingOrderedDict, Any

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

import dstruct
import dvalue
from Final_Project.files.utils import train_and_evaluate, mix_seq_and_act, split_train_valid
from dstruct import (SVHNDataset, ContrastEnhanceTransform, SmallVGG)

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)

path_dataset = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"

exp4_train_dataset = SVHNDataset(os.path.join(path_dataset, "train_32x32.mat"), transform=None)
exp4_valid_dataset = SVHNDataset(os.path.join(path_dataset, "test_32x32.mat"), transform=None)

# exp3_universal_train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"), transform=None)

# exp3_train_dataset, exp3_valid_dataset = split_train_valid(exp3_universal_train_dataset, train_ratio=0.8)
# del exp3_universal_train_dataset

# ==================================================================== #

# Group 1
candidate_seq = dstruct.candidate_seq
candidate_seq_name = dstruct.candidate_seq_name
candidate_activation_func = dstruct.candidate_activation_func

exp4_1_hyperparams: Dict[str, Any] = dict(num_epochs=50, lr=1e-3,
                                          angle=45, crop=0.6,
                                          ratio=0.58, factor=1.2,
                                          criterion=nn.CrossEntropyLoss(),
                                          optimizer=optim.Adam)


def run_exp4_1(seq_with_name: Tuple[List[str], List[Tuple]],
               acts: List, hyper_params: Dict[str, Any],
               train_dataset: SVHNDataset,
               valid_dataset: SVHNDataset) -> List[Dict[str, Union[List[float], dict, float, int]]]:
    combinations = list(itertools.product(seq_with_name, acts))
    experiments = []

    a, b = train_dataset.get_meanstd(contrast_factor=1.0)

    train_trans = A.Compose([
        A.Lambda(image=lambda img, **kwargs: ContrastEnhanceTransform(hyper_params['factor'])(img)),
        A.RandomResizedCrop(32, 32, scale=(hyper_params['crop'], 1.0),
                            ratio=(hyper_params['ratio'], 1.0 / hyper_params['ratio'])),
        A.Rotate(limit=hyper_params['angle']),
        A.Normalize(mean=a, std=b),
        ToTensorV2()
    ])
    valid_trans = A.Compose([
        A.Normalize(mean=a, std=b),
        ToTensorV2()
    ])

    train_dataset.transform = train_trans
    valid_dataset.transform = valid_trans

    for i, combo in enumerate(combinations):
        (seq_name, seq), activation = combo
        print(f"Running Exp {i + 1}: seq={seq_name}, activation={activation.__class__.__name__}")

        this_model = SmallVGG()
        x, y = mix_seq_and_act(seq, activation)
        this_model.conv_layers = x  # new conv_layers
        this_model.fc_layers = y  # new fc_layers
        this_model = this_model.to(device)

        num_epochs = hyper_params['num_epochs']
        lr = hyper_params['lr']
        criterion = hyper_params['criterion']
        optimizer = hyper_params['optimizer'](this_model.parameters(), lr=lr)

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

        train_losses, test_losses = train_and_evaluate(this_model, train_loader, valid_loader, criterion,
                                                       optimizer, num_epochs,
                                                       stop_early_params={
                                                           "min_delta": 0.01,
                                                           "patience": 5
                                                       })
        experiments.append({
            "structure of CNN": seq_name,
            "activation": activation.__class__.__name__,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "model_state_dict": this_model.state_dict()
        })

        del this_model, criterion, optimizer
        del train_loader, valid_loader
        torch.cuda.empty_cache()

    return experiments


exp4_1 = run_exp4_1(zip(candidate_seq_name, candidate_seq), candidate_activation_func,
                    exp4_1_hyperparams,
                    exp4_train_dataset, exp4_valid_dataset)
time_str = str(time.time()).replace(".", "")

models_dir_path = os.path.exists("../models") and "../models" or "./models"
torch.save(exp4_1, f"{models_dir_path}/exp4_1_{time_str}.pth")

# ==================================================================== #

# Group 2
# candidate_drop_rate = dvalue.candidate_drop_rate
# candidate_batch_size = dvalue.candidate_batch_size
#
# exp4_2_hyperparams: Dict[str, Any] = dict(num_epochs=15, lr=0.001,
#                                           angle=45, crop=0.08,
#                                           ratio=0.75, bias=dvalue.candidate_channel_biases[0],
#                                           sequence=dstruct.candidate_seq[0],
#                                           activation=dstruct.candidate_activation_func[0],
#                                           norm_mean=dvalue.FULL_BIAS_norm_mean[0],
#                                           norm_std=dvalue.FULL_BIAS_norm_std[0])  #TODO
#
#
# def run_exp4_2(drops: List[float], batches: List[int], hyperparams: Dict[str, Any],
#                train_dataset: SVHNDataset,
#                test_dataset: SVHNDataset) -> List[Dict[str, Union[List[float], dict, float, int]]]:
#     experiments = []
#     cnt = 1
#     transform = A.Compose([
#         A.Lambda(image=lambda img, **kwargs: AddBiasTransform(hyperparams['bias'])(img)),
#         A.RandomResizedCrop(32, 32, scale=(hyperparams['crop'], 1.0),
#                             ratio=(hyperparams['ratio'], 1.0 / hyperparams['ratio'])),
#         A.Rotate(limit=hyperparams['angle']),
#         A.Normalize(mean=hyperparams['norm_mean'], std=hyperparams['norm_std']),
#         ToTensorV2()
#     ])
#     train_dataset.transform = transform
#     test_dataset.transform = transform
#     x, y = mix_seq_and_act(hyperparams['sequence'], hyperparams['activation'])
#
#     for _drop in drops:
#         for _batch in batches:
#             print(f"Experiment {cnt}. Running experiment on drop rate: {_drop} "
#                   f"with batch size of: {_batch}")
#
#             cnt += 1
#
#             train_loader = DataLoader(train_dataset, batch_size=_batch, shuffle=True)
#             test_loader = DataLoader(test_dataset, batch_size=_batch, shuffle=False)
#
#             num_epochs = hyperparams['num_epochs']
#             learning_rate = hyperparams['lr']
#
#             exp4_2_model = SmallVGG()
#             exp4_2_model.conv_layers = x
#             exp4_2_model.fc_layers = y
#
#             exp4_2_model = exp4_2_model.to(device)  # to device after mixing
#
#             criterion = nn.CrossEntropyLoss()
#             optimizer = optim.Adam(exp4_2_model.parameters(), lr=learning_rate)
#
#             train_losses, test_losses = train_and_evaluate(exp4_2_model, train_loader, test_loader, criterion,
#                                                            optimizer, num_epochs)
#             experiments.append({
#                 "drop rate": _drop,
#                 "batch size": _batch,
#                 "train_losses": train_losses,
#                 "test_losses": test_losses,
#                 "model_state_dict": exp4_2_model.state_dict()
#             })
#
#             del exp4_2_model, criterion, optimizer
#             del train_loader, test_loader
#             torch.cuda.empty_cache()
#
#     return experiments
#
#
# exp4_2 = run_exp4_2(candidate_drop_rate, candidate_batch_size, exp4_2_hyperparams,
#                     exp4_train_dataset, exp4_valid_dataset)
# time_str = str(time.time()).replace(".", "")
#
# models_dir_path = os.path.exists("../models") and "../models" or "./models"
# torch.save(exp4_2, f"{models_dir_path}/exp4_2_{time_str}.pth")
# ==================================================================== #
