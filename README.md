# waste_management_AI_model

## Title: AI-Enabled Waste Classification System under Green Technology and Sustainability Theme

## ♻️ AI Waste Classifier

An AI-powered waste classification system that identifies different types of waste materials from images using Deep Learning (CNN) and provides an interactive web interface built with Streamlit.

## 📌 Problem Statement

Effective waste segregation is a critical challenge in modern waste management systems. Manual sorting is time-consuming, error-prone, and inefficient.
This project aims to automate waste classification by using a deep learning model that can classify waste images into predefined categories, assisting in smarter and faster waste management solutions.
This project aims to develop an AI-assisted waste classification system that can analyze images of waste and predict their material category, helping to improve recycling efficiency and reduce environmental impact, while analyzing the limitations of model performance caused by dataset quality and real-world variability.

## 🎯 Project Objectives

- Build a deep learning model to classify waste images into categories.
- Use transfer learning to improve performance with limited data.
- Deploy the trained model using a simple and user-friendly web interface.
- Allow users to upload an image and get instant classification results.

## 🗂️ Waste Categories

The model is trained to classify images into six categories:

📦 Cardboard
🍾 Glass
🥫 Metal
🗞️ Paper
🧴 Plastic
🗑️ Trash

## 🚀 Features:

- Upload an image of waste (JPG, JPEG, PNG)
- AI-assisted classifies into six categories:
- Displays prediction confidence
- Interactive Streamlit web interface
- Light and Dark mode UI support

## 🧠 Tools and Technology Stack :

🔹 Machine Learning

TensorFlow / Keras
Convolutional Neural Networks (CNN)
Transfer Learning (MobileNetV2)

🔹 Frontend

Streamlit (Python-based web framework)

🔹 Development Tools

Google Colab (model training)
VS Code (application development)
GitHub (version control)

## 🗂️ Dataset Preparation:

The model is trained on the **TrashType Image Dataset**, which contains images belonging to the following classes:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

All images are resized to **224×224** before training and inference.
It was downloaded in ZIP format and extracted in Google Colab using the following command:
```
!unzip /content/drive/MyDrive/waste_management_dataset.zip -d /content/dataset
```
Data Set used: 
```
https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset
```
## ⚙️ Implementation Procedure
Step 1: Dataset Preparation

Waste image dataset organized into six folders (one per class).
Dataset loaded using image_dataset_from_directory.

Step 2: Model Training (Google Colab)

Applied data augmentation (flip, rotation, zoom).
Used transfer learning with a pretrained CNN.
Training done in two phases:
Phase 1: Train classification head
Phase 2: Fine-tune top layers
Achieved ~93% training accuracy and ~86% validation accuracy.

Step 3: Model Saving

Final model saved as waste_classifier_model.keras.

Step 4: Application Development (VS Code)

Built a Streamlit web app for image upload and prediction.
Integrated the trained model for inference.

Step 5: Local Execution

App runs locally using Streamlit and displays prediction + confidence.

## ▶️ How to Use

1. Start the Streamlit app
2. Upload an image (JPG, JPEG, PNG)
3. View the predicted waste category and confidence score

## ⚠️ Known Limitations

The model may misclassify some dataset images, often predicting a dominant class.
In some cases, the model tends to predict a dominant class for multiple inputs.
Performance is sensitive to lighting, background, and image quality.
Dataset imbalance can affect prediction consistency.
Model requires further tuning and dataset expansion for real-world deployment.

⚠️ These limitations are acknowledged and documented as part of the learning process.

## 🔮Future Improvements

- Improve dataset balance
- Deploy to cloud
- Add camera support
- Improve UI accessibility

## ✅ Conclusion :

This project demonstrates the end-to-end development of an AI-powered image classification system, from data preprocessing and model training to deployment using Streamlit.
While the current model has prediction limitations, the project successfully showcases practical implementation of deep learning techniques in waste management and provides a strong foundation for future improvements.
