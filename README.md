# waste_management_AI_model

Title: AI-Enabled Waste Classification System under Green Technology and Sustainability Theme

Problem Statement: 

With the rapid growth of urban areas and consumer waste, improper waste sorting has become a significant environmental issue. In many developing regions, workers manually sort waste into categories like plastic, paper, metal, glass, and organic matter. This manual sorting takes a lot of time, is prone to mistakes, and poses health risks for workers.

Poor sorting results in more landfill waste, pollution, and ineffective recycling. It affects environmental sustainability and adds to climate change.

This project aims to develop an AI-assisted waste classification system that can analyze images of waste and predict their material category, helping to improve recycling efficiency and reduce environmental impact, while analyzing the limitations of model performance caused by dataset quality and real-world variability.

🚀 Features:

 - Upload an image of waste (JPG, JPEG, PNG)
 - AI-assisted classifies into six categories:
       Cardboard 📦
       Glass 🍾
       Metal 🥫
       Paper 🗞️
       Plastic 🧴
       Trash 🗑️
- Displays prediction confidence
- Interactive Streamlit web interface
- Light and Dark mode UI support

Tools Stack :

1. Google Colab:

GPU acceleration for faster model training
Easy file handling and dataset management
Supports Python, TensorFlow, and libraries required for deep learning

2. Python Programming Language used for the entire development of Data loading, Model building, Model training, Model testing.

3. TensorFlow & Keras frameworks used for: Image preprocessing, Model creation (MobileNetV2 + custom layers), Compilation and training, Saving/loading the .keras model.

4. Streamlit for frontend web framework for deployment: Upload and preview images, Run real-time predictions using the trained model, Display confidence and waste category, Provide an interactive UI


Dataset Preparation:

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
## ▶️ How to Use

1. Start the Streamlit app
2. Upload an image (JPG, JPEG, PNG)
3. View the predicted waste category and confidence score

## ⚠️ Observations & Limitations

- The model shows confusion between visually similar waste categories such as **glass, metal, and plastic**.
- In some cases, the model tends to predict a dominant class for multiple inputs.
- This behavior is caused by:
  - Visual similarity between materials
  - Class imbalance in the dataset
  - Label noise and mixed-material objects
- Even dataset images may be misclassified due to these factors.

These observations highlight the inherent difficulty of fine-grained material classification using image-based deep learning models.

## Future Improvements

- Improve dataset balance
- Deploy to cloud
- Add camera support
- Improve UI accessibility

## Conclusion :

This project explored the application of deep learning for automated waste classification using image data. By leveraging transfer learning with a pretrained MobileNetV2 model, an AI-assisted system was developed to classify waste materials into multiple categories. The project successfully demonstrated the complete machine learning pipeline, including dataset preparation, model training, evaluation, and deployment through a Streamlit web application.

During experimentation, the model achieved good overall training and validation accuracy; however, it also revealed challenges such as class imbalance, visual similarity between materials, and dataset label noise. These factors resulted in occasional misclassification, even on dataset images, highlighting the limitations of fine-grained material classification using image-based deep learning alone.

Despite these challenges, the project provides valuable insights into real-world machine learning development, emphasizing the importance of data quality, proper evaluation, and honest documentation of limitations. The implemented system serves as an AI-assisted decision support tool for waste segregation and establishes a strong foundation for future improvements using cleaner datasets, refined class definitions, or advanced models.
