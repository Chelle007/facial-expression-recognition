# Facial Emotion Recognition (FER) Using Deep Learning Models

This project implements and evaluates four deep learning architectures for facial emotion recognition on the **RAF-DB** dataset:

- Convolutional Neural Network (Custom CNN)
- MobileNetV2
- EfficientNetB0
- ResNet18

The goal is to compare model performance in terms of **classification accuracy**, **inference speed**, and **model size and** to deploy the best-performing models in a **browser-based real-time emotion recognition demo**.

---

## 📊 Dataset

The project uses the **Raf-DB (Real-world Affective Faces Database)**, which contains real-world facial images annotated with seven basic emotion labels.

Preprocessing steps include:

- Face alignment and resizing
- Data augmentation (rotation, flipping, etc.)
- Normalization for model compatibility

---

## 🌐 Data Source

The RAF-DB dataset is publicly available via [**Kaggle**](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset) for training and evaluation.

---

## 🧪 Training and Evaluation

All training and evaluation were conducted in **Google Colab**, with GPU acceleration.

Each model is trained and benchmarked, and the results (accuracy scores, confusion matrices, and performance comparisons) are documented directly in the notebooks.

---

## ▶️ Running the Jupyter Notebooks

### Environment

All notebooks run directly in **Google Colab** with GPU support.

### Notebook Links

- **Custom CNN:**\
  [https://colab.research.google.com/github/Chelle007/facial-expression-recognition/blob/main/ipynb/CNN\_v3.ipynb](https://colab.research.google.com/github/Chelle007/facial-expression-recognition/blob/main/ipynb/CNN_v3.ipynb)

- **MobileNetV2:**\
  [https://colab.research.google.com/drive/1Epq0YbM-MDjiAdSvqnl7fOrpdZYSywYr#scrollTo=87VDVa1eziFd](https://colab.research.google.com/drive/1Epq0YbM-MDjiAdSvqnl7fOrpdZYSywYr#scrollTo=87VDVa1eziFd)

- **EfficientNet-B0:**\
  https://colab.research.google.com/drive/16-Ra3G_Rwnx3yB7YndqpA5rkmFln_WnK?usp=sharing

- **ResNet18:**\
[https://colab.research.google.com/drive/1GMzoBdbuyrXdY20VvuXb4bSMTmMJ6Z6R#scrollTo=JU-zhrdgzdRA](https://colab.research.google.com/drive/1GMzoBdbuyrXdY20VvuXb4bSMTmMJ6Z6R#scrollTo=JU-zhrdgzdRA)

All dependencies install automatically when each notebook is run.

> The **executed notebooks with output cells** are saved in the `ipynb` folder.

---

## 📦 Dependencies

The following libraries are required for training and running the models (automatically available in Google Colab):

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `collections` (standard library)
- `time` (standard library)
- `google.colab` (for file uploads/downloads)
- `kaggle` (for dataset download)

If running locally (not in Colab), install the key packages:

```
pip install torch torchvision numpy matplotlib
```

---

## 🌐 Web Demo (Real-Time Emotion Recognition)

The web demo allows users to test emotion recognition in the browser using the device camera.

### Folder Structure

Download all files into a single directory before running.

### Option 1: Run Locally

1. Open the project folder
2. Double-click `index.html`
3. Allow camera access when prompted
4. Click **Open Camera** to begin recognition
5. Switch models freely using the interface
6. Click **Stop Camera** to end the session

### Option 2: Run via Local HTTP Server

Requires Python installed.

```
python -m http.server 8000
```

Visit in browser:

```
http://localhost:8000
```

### Option 3: Run in Visual Studio Code&#x20;

1. Open the project folder in VS Code
2. Right-click `index.html`
3. Select **Open with Live Server**

---

## 🎯 Project Purpose

This project demonstrates how different neural network architectures perform on real-world facial emotion recognition tasks and how they can be deployed efficiently on the web.

It provides a complete pipeline from **training → evaluation → deployment**.

