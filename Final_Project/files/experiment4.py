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
from Final_Project.files.utils import train_and_evaluate, mix_seq_and_act
from dstruct import (SVHNDataset, AddBiasTransform, SmallVGG)

device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_name)

path_dataset = os.path.exists("../data") and "../data/SVHN_mat" or "./data/SVHN_mat"

exp4_train_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "train_32x32.mat"), transform_component=None)
exp4_test_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "test_32x32.mat"), transform_component=None)
# exp4_extra_dataset = SVHNDataset(mat_file=os.path.join(path_dataset, "extra_32x32.mat"), transform_component=None)

# ==================================================================== #

# Group 1
candidate_seq = dstruct.candidate_seq
candidate_seq_name = dstruct.candidate_seq_name
candidate_activation_func = dstruct.candidate_activation_func

exp4_1_hyperparams: Dict[str, Any] = dict(num_epochs=15, lr=0.001,
                                          angle=45, crop=0.08,
                                          ratio=0.75, bias=dvalue.candidate_channel_biases[0],
                                          norm_mean=dvalue.FULL_BIAS_norm_mean[0],
                                          norm_std=dvalue.FULL_BIAS_norm_std[0])  #TODO


def run_exp4_1(sequence_with_name: Tuple[List[str], List[Tuple]],
               acts: List, hyperparams: Dict[str, Union[int, float]],
               train_dataset: SVHNDataset,
               test_dataset: SVHNDataset) -> List[Dict[str, Union[List[float], dict, float, int]]]:
    experiments = []
    cnt = 1
    transform = A.Compose([
        A.Lambda(image=lambda img, **kwargs: AddBiasTransform(hyperparams['bias'])(img)),
        A.RandomResizedCrop(32, 32, scale=(hyperparams['crop'], 1.0),
                            ratio=(hyperparams['ratio'], 1.0 / hyperparams['ratio'])),
        A.Rotate(limit=hyperparams['angle']),
        A.Normalize(mean=hyperparams['norm_mean'], std=hyperparams['norm_std']),
        ToTensorV2()
    ])
    train_dataset.transform = transform
    test_dataset.transform = transform

    for _name, _seq in sequence_with_name:
        for _act in acts:
            print(f"Experiment {cnt}. Running experiment on CNN shape: {_name} "
                  f"with activation function: {type(_act)}")

            cnt += 1

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            num_epochs = hyperparams['num_epochs']
            learning_rate = hyperparams['lr']

            exp4_1_model = SmallVGG()
            x, y = mix_seq_and_act(_seq, _act)
            exp4_1_model.conv_layers = x  # new conv_layers
            exp4_1_model.fc_layers = y  # new fc_layers

            exp4_1_model = exp4_1_model.to(device)  # to device after mixing

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(exp4_1_model.parameters(), lr=learning_rate)

            train_losses, test_losses = train_and_evaluate(exp4_1_model, train_loader, test_loader, criterion,
                                                           optimizer, num_epochs)
            experiments.append({
                "structure of CNN": _name,
                "activation": type(_act),
                "train_losses": train_losses,
                "test_losses": test_losses,
                "model_state_dict": exp4_1_model.state_dict()
            })

            del exp4_1_model, criterion, optimizer
            del train_loader, test_loader
            torch.cuda.empty_cache()

    return experiments


exp4_1 = run_exp4_1(zip(candidate_seq_name, candidate_seq), candidate_activation_func, exp4_1_hyperparams,
                    exp4_train_dataset, exp4_test_dataset)
time_str = str(time.time()).replace(".", "")

models_dir_path = os.path.exists("../models") and "../models" or "./models"
torch.save(exp4_1, f"{models_dir_path}/exp4_1_{time_str}.pth")

# ==================================================================== #

# Group 2
candidate_drop_rate = dvalue.candidate_drop_rate
candidate_batch_size = dvalue.candidate_batch_size

exp4_2_hyperparams: Dict[str, Any] = dict(num_epochs=15, lr=0.001,
                                          angle=45, crop=0.08,
                                          ratio=0.75, bias=dvalue.candidate_channel_biases[0],
                                          sequence=dstruct.candidate_seq[0],
                                          activation=dstruct.candidate_activation_func[0],
                                          norm_mean=dvalue.FULL_BIAS_norm_mean[0],
                                          norm_std=dvalue.FULL_BIAS_norm_std[0])  #TODO


def run_exp4_2(drops: List[float], batches: List[int], hyperparams: Dict[str, Any],
               train_dataset: SVHNDataset,
               test_dataset: SVHNDataset) -> List[Dict[str, Union[List[float], dict, float, int]]]:
    experiments = []
    cnt = 1
    transform = A.Compose([
        A.Lambda(image=lambda img, **kwargs: AddBiasTransform(hyperparams['bias'])(img)),
        A.RandomResizedCrop(32, 32, scale=(hyperparams['crop'], 1.0),
                            ratio=(hyperparams['ratio'], 1.0 / hyperparams['ratio'])),
        A.Rotate(limit=hyperparams['angle']),
        A.Normalize(mean=hyperparams['norm_mean'], std=hyperparams['norm_std']),
        ToTensorV2()
    ])
    train_dataset.transform = transform
    test_dataset.transform = transform
    x, y = mix_seq_and_act(hyperparams['sequence'], hyperparams['activation'])

    for _drop in drops:
        for _batch in batches:
            print(f"Experiment {cnt}. Running experiment on drop rate: {_drop} "
                  f"with batch size of: {_batch}")

            cnt += 1

            train_loader = DataLoader(train_dataset, batch_size=_batch, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=_batch, shuffle=False)

            num_epochs = hyperparams['num_epochs']
            learning_rate = hyperparams['lr']

            exp4_2_model = SmallVGG()
            exp4_2_model.conv_layers = x
            exp4_2_model.fc_layers = y

            exp4_2_model = exp4_2_model.to(device)  # to device after mixing

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(exp4_2_model.parameters(), lr=learning_rate)

            train_losses, test_losses = train_and_evaluate(exp4_2_model, train_loader, test_loader, criterion,
                                                           optimizer, num_epochs)
            experiments.append({
                "drop rate": _drop,
                "batch size": _batch,
                "train_losses": train_losses,
                "test_losses": test_losses,
                "model_state_dict": exp4_2_model.state_dict()
            })

            del exp4_2_model, criterion, optimizer
            del train_loader, test_loader
            torch.cuda.empty_cache()

    return experiments


exp4_2 = run_exp4_2(candidate_drop_rate, candidate_batch_size, exp4_2_hyperparams,
                    exp4_train_dataset, exp4_test_dataset)
time_str = str(time.time()).replace(".", "")

models_dir_path = os.path.exists("../models") and "../models" or "./models"
torch.save(exp4_2, f"{models_dir_path}/exp4_2_{time_str}.pth")
# ==================================================================== #
