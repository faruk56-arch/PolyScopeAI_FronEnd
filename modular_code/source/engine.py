
import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from ML_Pipeline.utils import AverageMeter,iou_score
from albumentations import Resize
# from albumentations.augmentations import transforms
from albumentations import Compose, Resize
from albumentations.augmentations.transforms import Normalize

from albumentations.core.composition import Compose, OneOf
#from albumentations.augmentations.transforms import RandomRotate90
from ML_Pipeline.network import UNetPP, VGGBlock
from ML_Pipeline.dataset import DataSet
from ML_Pipeline.predict import image_loader

import streamlit as st
import torch
import yaml
import numpy as np
import cv2
from albumentations import Compose, Resize, Normalize
from ML_Pipeline.network import UNetPP

import streamlit as st
import torch
import yaml
import numpy as np
import cv2
from albumentations import Compose, Resize, Normalize
from ML_Pipeline.network import UNetPP

# Set random seeds
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --- Page Configuration ---
st.set_page_config(
    page_title="Colorectal Polyp Detection and Segmentation",
    page_icon="🔬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

im_width = config["im_width"]
im_height = config["im_height"]
model_path = "../output/models/logs/model.pth"

# Function to load the segmentation model
@st.cache_resource
def load_segmentation_model():
    model = UNetPP(num_classes=1, input_channels=3, deep_supervision=True)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        st.stop()
    model.eval()
    return model

segmentation_model = load_segmentation_model()

# --- Custom CSS for styling ---
st.markdown(
    """
    <style>
    /* Center the title */
    .css-1v0mbdj.e1fqkh3o3 {
        text-align: center;
    }
    /* Adjust the width of the main content */
    .css-1wrcr25.egzxvld3 {
        max-width: 800px;
        margin: auto;
    }
    /* Style the subheaders */
    .stMarkdown h3 {
        color: #2c3e50;
        text-align: center;
    }
    /* Style the classification result */
    .classification-result h3 {
        text-align: center;
    }
    /* Style the error message */
    .css-10trblm {
        text-align: center;
    }
    /* Custom font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.image("./logo1.jpeg", use_column_width=True)
    st.markdown("""
    ## About
    This application utilizes a deep learning model to detect and segment colorectal polyps from endoscopic images.

    ## Instructions
    - Upload an endoscopic image.
    - View the segmentation results.
    - The application will indicate if a polyp is detected.

    **Disclaimer:** Please note that this application is currently under development and may occasionally produce incorrect results.

    ## Developed by
    - **Momin Faruk**
    - Master's Thesis in AI and Medical Imaging
    """)

# --- Main Content ---
st.title("🔬 Colorectal Polyp Detection and Segmentation")

# Instructions
st.markdown("""
This application allows you to upload an endoscopic image and performs segmentation to detect colorectal polyps.

**How to use the application:**

1. **Upload an Image:** Click on the "Browse files" button below to upload an image in PNG, JPG, or JPEG format.

2. **View Results:** After uploading, the application will display:
   - The input image.
   - The classification result indicating whether a polyp is detected.
   - The predicted segmentation mask.
   - An overlay of the mask on the original image.

**Note:** The model may take a few seconds to process the image.

**Please ensure that you upload a colonoscopy or endoscopic image related to colorectal polyps.**
""")

uploaded_file = st.file_uploader(
    "Choose an endoscopic image to upload",
    type=["png", "jpg", "jpeg"],
    help="Upload an image file in PNG, JPG, or JPEG format."
)

if uploaded_file is not None:
    def image_loader_streamlit(image_data):
        try:
            image_cv = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image_cv is None:
                st.error("Invalid image file. Please upload a valid image.")
                st.stop()
            val_transform = Compose([
                Resize(256, 256),
                Normalize(),
            ])
            img = val_transform(image=image_cv)["image"]
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)
            return img, image_cv
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.stop()

    image, original_image = image_loader_streamlit(uploaded_file.getvalue())

    # Display the input image
    st.subheader("Input Image")
    st.image(
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
        caption='Input Image',
        use_column_width=True
    )

    input_tensor = torch.from_numpy(np.expand_dims(image, 0)).float()

    # Run the segmentation model
    with st.spinner("Processing..."):
        with torch.no_grad():
            try:
                mask = segmentation_model(input_tensor)
                if isinstance(mask, (list, tuple)):
                    mask = mask[-1]
                mask = mask.cpu()
                mask = torch.sigmoid(mask)  # Apply sigmoid activation
                mask_np = mask.numpy()
                mask_np = np.squeeze(np.squeeze(mask_np, axis=0), axis=0)
            except Exception as e:
                st.error(f"Error during model prediction: {e}")
                st.stop()

    # Calculate statistical measures
    mask_mean = np.mean(mask_np)
    mask_std = np.std(mask_np)
    mask_sum = np.sum(mask_np)
    mask_max = np.max(mask_np)

    # Additional heuristic checks
    # Binarize the mask at a lower threshold
    lower_threshold = 0.1
    mask_binary_low = np.zeros_like(mask_np)
    mask_binary_low[mask_np >= lower_threshold] = 1
    mask_binary_low[mask_np < lower_threshold] = 0

    # Convert mask to uint8
    mask_binary_low_uint8 = mask_binary_low.astype(np.uint8)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary_low_uint8, connectivity=8)

    # Number of significant components (excluding background)
    num_components = num_labels - 1

    # Size of the largest component
    if num_components > 0:
        largest_component_size = np.max(stats[1:, cv2.CC_STAT_AREA])  # Exclude background
    else:
        largest_component_size = 0

    mean_threshold = 0.01
    std_threshold = 0.01
    sum_threshold = 500
    max_value_threshold = 0.1
    component_size_threshold = 500
    
    # Check if the image is likely irrelevant
    if ((mask_mean < mean_threshold) and
        (mask_std < std_threshold) and
        (mask_sum < sum_threshold) and
        (mask_max < max_value_threshold) and
        (largest_component_size < component_size_threshold)):
        st.error("Unrecognized image. Please upload an endoscopic image related to colorectal polyps.")
        st.stop()

    # Threshold the mask
    threshold = 0.5
    mask_binary = np.zeros_like(mask_np)
    mask_binary[mask_np >= threshold] = 255
    mask_binary[mask_np < threshold] = 0

    # Determine if the image is positive or negative
    if np.sum(mask_binary) > 0:
        classification = "Positive for colorectal polyp"
        classification_color = "red"
    else:
        classification = "Negative for colorectal polyp"
        classification_color = "green"

    # Resize the mask to the original image size
    mask_resized = cv2.resize(mask_binary, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_resized = mask_resized.astype(np.uint8)

    # Create overlay
    overlay = original_image.copy()
    overlay[mask_resized == 255] = [0, 0, 255]  # Red color for polyp area

    st.markdown("---")
    st.markdown(f"<h3 style='color:{classification_color}; text-align: center;'>{classification}</h3>", unsafe_allow_html=True)

    # Display images in columns
    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            mask_resized,
            caption='Predicted Segmentation Mask',
            use_column_width=True,
            clamp=True,
        )

    with col2:
        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            caption='Overlay on Original Image',
            use_column_width=True,
        )

else:
    st.info("Please upload an image to get started.")
