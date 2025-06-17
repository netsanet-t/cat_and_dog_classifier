# Cats vs Dogs: CNN Architecture Experiments with Batch Normalization & Residual Learning

This Jupyter Notebook presents a deep learning-based image classification project that distinguishes between images of cats and dogs using Convolutional Neural Networks (CNNs). The focus of this work is not only on classification accuracy but also on experimenting with modern architectural innovations that can significantly improve model performance and generalization ability.

The notebook walks through various stages of a typical machine learning pipeline—data loading, preprocessing, model design, training, and evaluation—while integrating advanced concepts such as **batch normalization** and **residual learning** (inspired by ResNet architectures).

> ⚠️ **Note:** The model training and evaluation were conducted on **Google Colab**, taking advantage of GPU acceleration for faster experimentation.

---

## 🐱🐶 Objective

Classify images of cats and dogs using deep learning, while experimenting with:
- Fully convolutional networks (removing pooling layers)
- Batch Normalization
- Residual blocks (ResNet-style shortcut connections)

---

## 🧠 Techniques Explored

- Data preprocessing (resizing, normalizing, grayscale conversion)
- CNN model design using `Conv2D`, `ReLU`, and `softmax`
- Batch Normalization for improved training stability and speed
- Custom implementation of Residual Connections (ResNet blocks)
- Model evaluation with training/validation data split

---

## 📁 File

- `Cat_and_Dog_Classifier.ipynb` — the full notebook containing:
  - Code
  - Comments
  - Experiment results
---

## 🛠️ Requirements

To run this notebook locally:

```bash
pip install tensorflow numpy scikit-learn matplotlib
