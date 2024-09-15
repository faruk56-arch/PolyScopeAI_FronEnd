import cv2
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from ML_Pipeline.network import UNetPP
from argparse import ArgumentParser
from albumentations.augmentations import transforms
from albumentations import Resize
from albumentations.core.composition import Compose
import cv2
import numpy as np
from albumentations.augmentations import transforms
from albumentations import Resize, Compose

# Define the transformation
val_transform = Compose([
    Resize(256, 256),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example means and stds
])

def image_loader(image_data):
    """Adapted to handle an image as a NumPy array."""
    # Assume image_data is a NumPy array, e.g., from cv2.imdecode()
    # Ensure the image is in RGB format if it comes from cv2 which uses BGR by default
    if image_data.shape[2] == 3:  # Check if the image is colored
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    img = val_transform(image=image_data)["image"]
    img = img.astype('float32')
    img = np.transpose(img, (2, 0, 1))  # Change from HWC to CHW format expected by PyTorch
    return img


