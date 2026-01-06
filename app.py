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
