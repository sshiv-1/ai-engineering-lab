import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL ARCHITECTURE 
# =========================================================
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()

        # --- Feature Extraction Container ---
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # --- Classifier Container ---
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 1. Pass through feature extractor
        x = self.features(x)

        # 2. Flatten (Batch size, Channels * Height * Width)
        x = x.view(x.size(0), -1)

        # 3. Pass through classifier
        x = self.classifier(x)
        
        return x


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = BrainTumorCNN()
    model.load_state_dict(torch.load("best_brain_tumor_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================================================
# PREPROCESSING 
# =========================================================
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def mri_preprocessing(image):
    # Contour Cropping
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        image = image[y:y+h, x:x+w]

    # CLAHE Contrast Enhancement
    gray_cropped = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_cropped)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_image(image):
    image = np.array(image)
    image = mri_preprocessing(image)
    image = Image.fromarray(image)
    image = val_transforms(image)
    image = image.unsqueeze(0)
    return image.to(device)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Prediction", "Model Architecture", "Evaluation Metrics"]
)

# =========================================================
# 1️⃣ PREDICTION PAGE
# =========================================================
if page == "Prediction":

    st.title("🧠 Brain Tumor MRI Classification")
    st.write("Upload an MRI image (PNG / JPG / JPEG)")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", use_container_width=True)

        input_tensor = preprocess_image(image)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        st.subheader("🧾 Prediction Result")
        st.success(f"Predicted Class: {CLASS_NAMES[predicted_class]}")

        st.subheader("📊 Confidence Scores")
        prob_df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": probabilities.cpu().numpy()[0]
        })

        st.bar_chart(prob_df.set_index("Class"))

# =========================================================
# 2️⃣ MODEL ARCHITECTURE PAGE
# =========================================================
elif page == "Model Architecture":

    st.title("📐 Model Architecture")

    st.markdown("""
    ### CNN-Based Brain Tumor Classifier

    - Custom CNN with Convolution + ReLU + MaxPooling
    - Fully Connected Layers
    - Output Layer: 4 Classes
    - Input Size: 224 × 224 RGB
    - ROI Extraction via Contour Detection
    - CLAHE Contrast Enhancement
    - ImageNet Normalization
    """)

    st.code(str(model))


elif page == "Evaluation Metrics":

    st.title("📊 Model Evaluation Metrics")

    st.metric("Test Accuracy", "90.08%")
    st.metric("Best Validation Accuracy", "89.50%")
    st.metric("Learning Rate", "0.0005")
    st.write("Total Test Samples: 1311")

    st.subheader("📄 Classification Report")

    report_df = pd.DataFrame({
        "Class": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "Precision": [0.96, 0.78, 0.94, 0.92],
        "Recall": [0.78, 0.85, 0.97, 0.97],
        "F1-Score": [0.86, 0.82, 0.96, 0.95],
        "Support": [300, 306, 405, 300]
    })

    st.dataframe(report_df, use_container_width=True)

    st.markdown("""
    **Macro Avg F1-Score:** 0.90  
    **Weighted Avg F1-Score:** 0.90  
    """)
