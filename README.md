# waste_management_AI_model

Title: AI-Enabled Waste Classification System under Green Technology and Sustainability Theme

Problem Statement: 

With the rapid growth of urban areas and consumer waste, improper waste sorting has become a significant environmental issue. In many developing regions, workers manually sort waste into categories like plastic, paper, metal, glass, and organic matter. This manual sorting takes a lot of time, is prone to mistakes, and poses health risks for workers.

Poor sorting results in more landfill waste, pollution, and ineffective recycling. It affects environmental sustainability and adds to climate change.

The main goal is to create a smart, automated waste classification system that can accurately identify and sort waste materials using Artificial Intelligence (AI) and Machine Learning (ML) techniques. This system will help achieve sustainable waste management, lessen the workload for people, and encourage eco-friendly recycling practices.

🚀 Features:

📸 Upload an image of waste
🧠 AI model classifies waste into:
       Cardboard 📦
       Glass 🍾
       Metal 🥫
       Paper 🗞️
       Plastic 🧴
       Trash 🗑️
⚡ Fast inference using MobileNetV2
🌗 Light & Dark mode support (Streamlit)
📊 Confidence score display
💻 Simple, clean Streamlit UI

🧠 Model Details

Architecture: MobileNetV2 (Transfer Learning)
Input size: 224 × 224 × 3
Dataset: TrashType Image Dataset
Training Strategy:
    Phase 1: Train classifier head
    Phase 2: Fine-tune top layers

Tools and Technologies Used: 

1. Google Colab:

GPU acceleration for faster model training
Easy file handling and dataset management
Supports Python, TensorFlow, and libraries required for deep learning

2. Python Programming Language used for the entire development of Data loading, Model building, Model training, Model testing.

3. TensorFlow & Keras frameworks used for: Image preprocessing, Model creation (MobileNetV2 + custom layers), Compilation and training, Saving/loading the .keras model.

4. Streamlit for frontend web framework for deployment: Upload and preview images, Run real-time predictions using the trained model, Display confidence and waste category, Provide an interactive UI


Dataset Preparation:

The dataset used contains multiple categories of waste materials (e.g., cardboard, glass, metal, paper, plastic, trash).
It was downloaded in ZIP format and extracted in Google Colab using the following command:
```
!unzip /content/drive/MyDrive/waste_management_dataset.zip -d /content/dataset
```
Data Set used: 
```
https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
```
Colab Code:
```
from google.colab import drive
drive.mount('/content/drive')

!unzip -q /content/drive/MyDrive/waste_management_dataset.zip -d /content/dataset
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "/content/dataset/TrashType_Image_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123

# -----------------------------
# Inspect directory (important)
# -----------------------------
print("Classes (folders) found:")
for d in sorted(os.listdir(DATA_DIR)):
    if os.path.isdir(os.path.join(DATA_DIR, d)):
        print("  -", d)

# -----------------------------
# Load dataset (single root dir)
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Class order:", class_names)

# Quick sanity check on labels
for images, labels in train_ds.take(1):
    print("Sample labels in first batch:", labels.numpy())
    break

# -----------------------------
# Performance optimizations
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# -----------------------------
# Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# -----------------------------
# Base model
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# -----------------------------
# Build model
# -----------------------------
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

x = data_augmentation(inputs)
x = layers.Rescaling(1./127.5, offset=-1)(x)   # <-- FIX
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="waste_classifier")
model.summary()

# ======================================================
# PHASE 1 — TRAIN CLASSIFIER HEAD
# ======================================================
base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🔵 Phase 1: Training classifier head")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
)

# ======================================================
# PHASE 2 — FINE-TUNE TOP LAYERS
# ======================================================
base_model.trainable = True

# unfreeze only last 20–30 layers
for layer in base_model.layers[:-25]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🟢 Phase 2: Fine-tuning top layers")
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
)

# ======================================================
# SAVE MODEL AND CLASS NAMES
# ======================================================
model.save("waste_classifier_model.keras")
np.save("waste_class_names.npy", np.array(class_names))
print("✅ Final model saved at waste_classifier_model.keras")
print("✅ Class names saved at waste_class_names.npy")

```
VS code for streamlit interface:
```
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# --------------------------------
# Page config
# --------------------------------
st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# --------------------------------
# Theme toggle
# --------------------------------
theme = st.toggle("🌗 Dark mode", value=True)

if theme:  # Dark mode
    bg_color = "#0e1117"
    card_color = "#0f2f1f"
    text_color = "#ffffff"
    secondary_text = "#b3b3b3"
    accent = "#6fff9b"
else:      # Light mode
    bg_color = "#f4f6f8"
    card_color = "#ffffff"
    text_color = "#1a1a1a"       # darker text
    secondary_text = "#444444"   # readable gray
    accent = "#0f7a4a"           # darker green

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: {text_color};
        }}
        p {{
            color: {secondary_text};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------
# Load model
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "waste_classifier_model.keras")

@st.cache_resource
def load_waste_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )

model = load_waste_model()

# --------------------------------
# Classes & icons
# --------------------------------
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_icons = {
    "cardboard": "📦",
    "glass": "🧴",
    "metal": "🥫",
    "paper": "🗞️",
    "plastic": "♻️",
    "trash": "🗑️"
}

# --------------------------------
# Header
# --------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">♻️ AI Waste Classifier</h1>
    <p style="text-align:center; font-size:18px;">
        Upload an image of waste and let AI classify it
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# Prediction
# --------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # Preprocess (UNCHANGED)
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(preds))

    predicted_class = class_names[idx]
    confidence = preds[idx] * 100
    icon = class_icons[predicted_class]

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color:{card_color};
                padding:30px;
                border-radius:15px;
                text-align:center;
                box-shadow:0px 0px 15px rgba(0,0,0,0.15);
            ">
                <h1>{icon}</h1>
                <h2 style="color:{accent};">
                    {predicted_class.capitalize()}
                </h2>
                <p style="font-size:18px; color:{secondary_text};">
                    Confidence
                </p>
                <h1 style="color:{text_color};">
                    {confidence:.2f}%
                </h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------
    # Top-3 confidence bars
    # --------------------------------
    st.subheader("🔍 Prediction Confidence")

    top3 = np.argsort(preds)[::-1][:3]
    for i in top3:
        st.write(f"{class_icons[class_names[i]]} **{class_names[i].capitalize()}**")
        st.progress(float(preds[i]))

else:
    st.info("👆 Upload an image to start classification")

```
Conclusion: 

This project successfully demonstrates how Artificial Intelligence can be applied to support sustainable waste management. By using a deep learning model based on MobileNetV2, the system accurately classifies waste into six categories, reducing the need for manual sorting. The Streamlit-based web application enables easy real-time predictions, making the solution practical and user-friendly. Overall, the project shows that AI-driven waste classification can improve recycling efficiency, reduce environmental impact, and contribute to a cleaner, greener future. 

