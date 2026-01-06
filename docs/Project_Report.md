```
✅ Step-by-Step Implementation Procedure
(Using Google Colab and VS Code)

🔹 Step 1: Dataset Collection and Preparation (Google Colab)

1. A publicly available waste image dataset containing six classes (cardboard, glass, metal, paper, plastic, trash) was selected.
2. The dataset was uploaded to Google Drive for easy access from Colab.
3.Google Colab was used as the training environment due to its GPU support.
4.The dataset was extracted and organized into class-wise folders.
5.Images were resized to 224×224 pixels to match the input size required by the pretrained model.

🔹 Step 2: Dataset Loading and Preprocessing

1.TensorFlow’s image_dataset_from_directory() function was used to load the dataset.
2.The dataset was split into training (80%) and validation (20%) subsets.
3.Class labels were automatically inferred from folder names.
4.Dataset pipelines were optimized using batching and prefetching.
5.Minimal data augmentation (horizontal flipping) was applied to improve generalization.

🔹 Step 3: Model Selection and Architecture Design

1.MobileNetV2, pretrained on ImageNet, was chosen as the base model due to its efficiency and suitability for lightweight applications.
2.The top classification layers of MobileNetV2 were removed.
3.A custom classification head consisting of global average pooling, batch normalization, dense layers, and dropout was added.
4.The final output layer used a softmax activation to predict class probabilities.

🔹 Step 4: Model Training (Transfer Learning)

1.In Phase 1, the pretrained base model was frozen, and only the custom classification head was trained.
2.In Phase 2, selected top layers of the base model were unfrozen and fine-tuned using a lower learning rate.
3.The model was trained using the Adam optimizer and sparse categorical crossentropy loss.
4.Training and validation accuracy were monitored to evaluate learning progress.

🔹 Step 5: Model Evaluation and Analysis

1.Training and validation performance metrics were analyzed.
2.Misclassification patterns were observed, particularly among visually similar classes.
3.The impact of dataset imbalance and label noise on predictions was identified.
4.Model behavior was analyzed to understand real-world limitations.

🔹 Step 6: Model Saving and Export

1.The trained model was saved in .keras format for deployment.
2.Class label mappings were saved separately to ensure consistency during inference.
3.Model artifacts were downloaded from Colab to the local system.

🔹 Step 7: Deployment Using Streamlit (VS Code)

1.VS Code was used as the development environment for deployment.
2.A Streamlit application was developed to provide a user-friendly interface.
3.The trained model was loaded into the Streamlit app.
4.Users could upload waste images through the web interface.
5.The application displayed the predicted waste category along with a confidence score.
6.UI enhancements such as light/dark mode and icons were added for better usability.

🔹 Step 8: Testing and Validation

1.The deployed Streamlit application was tested using both dataset images and real-world images.
2.Predictions and confidence scores were observed to evaluate model behavior.
3.Model performance was analyzed with respect to class confusion and dominant predictions.
4.Edge cases such as visually similar waste materials were identified and examined.
5.The overall system behavior was reviewed to understand real-world applicability and limitations.
```
