Final Project Source File, Preview HTML, and Report are [Here](Final_Project).

Experiment Reults Summary Table: (Final Project/Experiment Results Summary Table)

Experiment sets/subsets are listed in the [Google Drive](https://drive.google.com/drive/folders/1SpnG2BSAXtR2b4Uza9Y7q3iZ23l2TTU9?usp=sharing) due to the large file size.

Experiment Set Structure:
```python
[ # One experiment set: A list of experiment objects.
    { # One experiment object
        "HYPER_PARAM_0": CANDIDATE_VALUE_0_0,        # Value of the candidate of the first hyper-param
        "HYPER_PARAM_1": CANDIDATE_VALUE_1_0,        # Value of the candidate of the second hyper-param
        "train_losses": train_losses_0,              # List of training losses over epochs
        "valid_losses": valid_losses_0,              # List of validation losses over epochs
        "model_state_dict": exp_model_0.state_dict()    # Model state
    },
    {
        "HYPER_PARAM_0": CANDIDATE_VALUE_0_1,
        "HYPER_PARAM_1": CANDIDATE_VALUE_1_1,
        "train_losses": train_losses_1,
        "valid_losses": valid_losses_1,
        "model_state_dict": exp_model_1.state_dict()
    },
    ...
]

```
