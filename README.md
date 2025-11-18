# waste_management_AI_model

Title: AI-Enabled Waste Classification System under Green Technology and Sustainability Theme

Problem Statement: 

With the rapid growth of urban areas and consumer waste, improper waste sorting has become a significant environmental issue. In many developing regions, workers manually sort waste into categories like plastic, paper, metal, glass, and organic matter. This manual sorting takes a lot of time, is prone to mistakes, and poses health risks for workers.

Poor sorting results in more landfill waste, pollution, and ineffective recycling. It affects environmental sustainability and adds to climate change.

The main goal is to create a smart, automated waste classification system that can accurately identify and sort waste materials using Artificial Intelligence (AI) and Machine Learning (ML) techniques. This system will help achieve sustainable waste management, lessen the workload for people, and encourage eco-friendly recycling practices.

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
!unzip /content/drive/MyDrive/waste_management_dataset.zip -d /content/dataset
import tensorflow as tf  # ✅ import TensorFlow

DATA_DIR = "/content/dataset/TrashType_Image_Dataset"



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

from tensorflow.keras import layers, models
import tensorflow as tf

IMG_SIZE = (224, 224)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(train_ds.class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
```
VS code foe streamlit interface:
```
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the model
model = load_model('waste_classifier_model.keras')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # update according to your dataset

st.title("♻️ AI Waste Classifier")
st.markdown("Upload an image of waste to predict its type using an AI model trained on green technology principles.")

uploaded_file = st.file_uploader("Upload a waste image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)


    # Prediction
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # Show result
    st.success(f"### 🧾 Predicted: {pred_class.capitalize()} ({confidence:.2f}% confidence)")
```
Conclusion: 

This project successfully demonstrates how Artificial Intelligence can be applied to support sustainable waste management. By using a deep learning model based on MobileNetV2, the system accurately classifies waste into six categories, reducing the need for manual sorting. The Streamlit-based web application enables easy real-time predictions, making the solution practical and user-friendly. Overall, the project shows that AI-driven waste classification can improve recycling efficiency, reduce environmental impact, and contribute to a cleaner, greener future. 

