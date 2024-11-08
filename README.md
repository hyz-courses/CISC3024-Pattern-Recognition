CISC3024 Final Project: Classification of SVHN Dataset with VGG
==================================
Huang Yanzhen, DC126732

Mai Jiajun, DC127853

1. Final project root directory is [Final_Project](Final_Project). Here are some quick accesses:
    - [Source `.ipynb` file](Final_Project/main.ipynb)
    - [Source `.ipynb` HTMLPreview](Final_Project/preview.html)
    - [Experiment Results Summary Table](Final_Project/Experiment_Results_Summary_Form.xlsx)

    - Huang Yanzhen:
        - [Project Slides](Final_Project/CISC3024_Final_Project_Slides_DC126732.pptx)
        - [Project Report PDF](Final_Project/CISC3024_Final_Project_Report_DC126732.pdf)
    - Mai Jiajun:
        - [Project Slides](Final_Project/CISC3024_Final_Project_Slides_DC127853.pptx)
        - [Project Report PDF](Final_Project/CISC3024_Final_Project_Report_DC127853.pdf)

2. Experiment sets/subsets are stored in the `.pth` files, they include hyper-parameter values, training and validation losses over epochs, and the model state. All of such files are listed in the [Google Drive](https://drive.google.com/drive/folders/1SpnG2BSAXtR2b4Uza9Y7q3iZ23l2TTU9?usp=sharing) due to the large file size. Create a `model` folder in the root directory of the final project and put them into it.

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

3. Dataset `.mat` files is also in the [Google Drive](https://drive.google.com/drive/u/0/folders/1pr9iEMgCk4kuQ7r_EPj_aP3IWA7nGqYS).  Create a `data` folder in the root directory of the final project and put them into it.
