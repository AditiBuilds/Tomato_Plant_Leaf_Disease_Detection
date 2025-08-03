# 🍅 Tomato Leaf Disease Detection using Deep Learning

## 📌 Overview

This project presents a high-accuracy, real-time, and scalable **Convolutional Neural Network (CNN)**-based system for **detecting diseases in tomato leaves**. It leverages the **DenseNet-121** architecture through **transfer learning** and includes **image augmentation** and **regularization** techniques to enhance generalization and performance.

Achieved a **validation accuracy of 96.55%** on a 13-class tomato disease dataset.

---

## 🗃️ Dataset

- Source: [Kaggle Tomato Disease Dataset](https://www.kaggle.com/datasets)
- Total Images: `~13,000+`
- Classes (13):
  - Black Mold
  - Gray Spot
  - Bacterial Spot
  - Early Blight
  - Late Blight
  - Leaf Mold
  - Septoria Leaf Spot
  - Spider Mites
  - Target Spot
  - Yellow Leaf Curl Virus
  - Mosaic Virus
  - Powdery Mildew
  - Healthy

> Dataset was divided into `train/` and `val/` folders and preprocessed using TensorFlow's `image_dataset_from_directory`.

---

## 🛠️ Tools & Libraries

- Python
- TensorFlow / Keras
- Matplotlib / Seaborn
- NumPy / Pandas
- OpenCV

---

## 🧠 Model Architecture

- **Base Model:** `DenseNet-121` (pretrained on ImageNet)
- **Custom Layers:**
  - Batch Normalization
  - Dropout (0.35)
  - Dense Layers (256, 120)
  - Output Layer with Softmax (13 classes)
- **Image Size:** 256x256x3
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Regularization:** L2 & Dropout
- **Data Augmentation:** Random Flip & Rotation
- **Callbacks:** EarlyStopping & ModelCheckpoint

---

## 🚀 Training Summary

| Metric              | Value     |
|---------------------|-----------|
| Epochs              | 12 (early stopped) |
| Final Validation Accuracy | **96.55%** |
| Loss                | 0.112     |

Visualized using accuracy/loss plots and confusion matrix.

---

## 📊 Results

- Detailed **classification report** and **confusion matrix** were generated.
- High precision and recall across all disease categories.
- Model performs exceptionally on challenging disease types like:
  - Yellow Leaf Curl Virus
  - Mosaic Virus
  - Powdery Mildew



## 📁 Project Structure

