import os
from typing import Dict, Any

import torch
import gradio as gr
import albumentations as A
from PIL import Image, ImageOps
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import nn, optim

from Final_Project.files.dstruct import ContrastEnhanceTransform

dir_path = os.path.exists("../models") and "../models" or "./models"
# model_path = os.path.join(dir_path, "exp2_17301145036294217.pth")
model_path = os.path.join(dir_path, "model_path.pth")
model = torch.load(model_path)
model.eval()

hyperparams: Dict[str, Any] = dict(num_epochs=50, lr=1e-3,
                                   angle=45, crop=0.6,
                                   ratio=0.58, factor=1.2,
                                   criterion=nn.CrossEntropyLoss(),
                                   optimizer=optim.Adam,
                                   # batch_size=128
                                   )
transform_a = A.Compose([
    A.Lambda(image=lambda img, **kwargs: ContrastEnhanceTransform(hyperparams['factor'])(img)),
    A.RandomResizedCrop(32, 32, scale=(hyperparams['crop'], 1.0),
                        ratio=(hyperparams['ratio'], 1.0 / hyperparams['ratio'])),
    A.Rotate(limit=hyperparams['angle'])
])


def predict_digit_from_sketch(image):
    image = ImageOps.invert(image)

    # Convert to numpy array and apply albumentations transformations
    image_np = np.array(image)
    image_a = transform_a(image=image_np)["image"]

    img_mean = np.mean(image_np, axis=(0, 1))
    img_std = np.std(image_np, axis=(0, 1))
    transform_b = A.Compose([
        A.Normalize(mean=img_mean, std=img_std),
        ToTensorV2()
    ])

    image_b = transform_b(image=image_np)["image"]

    image_b = image_b.transpose(2, 0, 1)                                # HWC to CHW format
    image_b = torch.tensor(image_b).unsqueeze(0)                        # Add batch dimension

    with torch.no_grad():
        output = model(image_b)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]   # Get probabilities

    prob_dict = {str(i): round(float(probabilities[i]), 4) for i in range(10)}
    return prob_dict


interface = gr.Interface(
    fn=predict_digit_from_sketch,
    inputs=gr.Sketchpad(shape=(32, 32), invert_colors=True),
    outputs=gr.Label(num_top_classes=10),                               # Display probabilities for all 10 classes
    title="Handwritten Digit Recognition with Sketchpad",
    description="Draw a digit (0-9) and see the model's prediction with probabilities for each digit."
)

interface.launch()                                                      # Launch the Gradio interface
