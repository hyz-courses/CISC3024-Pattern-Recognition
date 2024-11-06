import os
import random
from typing import Dict, Any, Union, Tuple, List

import torch
import gradio as gr
import albumentations as A
from PIL import Image, ImageOps
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch import nn, optim


class SmallVGG(nn.Module):
    def __init__(self, frame_size=32):
        super(SmallVGG, self).__init__()
        self.frame_size = frame_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(frame_size * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class ContrastEnhanceTransform:
    def __init__(self, factor: Union[float, Tuple[float, float]]) -> None:
        if isinstance(factor, tuple):
            self.factor_min = factor[0]
            self.factor_max = factor[1]
        else:
            self.factor_min = 1 / factor
            self.factor_max = factor

    def __call__(self, img: np.ndarray) -> np.ndarray:
        _dtype = img.dtype
        contrast_factor = random.uniform(self.factor_min, self.factor_max)
        img = np.clip(img * contrast_factor, 0, 255)  # apply contrast enhancement
        return img.astype(_dtype)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dir_path = os.path.exists("../models") and "../models" or "./models"
# model_path = os.path.join(dir_path, "exp2_17301145036294217.pth")
model_path = os.path.join(dir_path, "utility_17308807909570491.pth")
model_state = torch.load(model_path)

model = SmallVGG()
model.load_state_dict(model_state)
model.to(device)
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
    inputs=gr.Sketchpad(height=32, width=32),
    outputs=gr.Label(num_top_classes=10),                               # Display probabilities for all 10 classes
    title="Handwritten Digit Recognition with Sketchpad",
    description="Draw a digit (0-9) and see the model's prediction with probabilities for each digit."
)

interface.launch(share=False, server_name="127.0.0.1", server_port=7860)       # Launch the Gradio interface
