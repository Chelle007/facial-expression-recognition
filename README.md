# Facial Emotion Recognition (FER) with CNN, MobileNetV2, EfficientNetB0, and ResNet18

This project trains and compares 4 different models on the Raf-DB dataset for facial emotion recognition.
It compares their accuracy, speed, model size, and produces tables and graphs shown in the google colab notebook 

---

# How to run the jupyter notebook

## Environment Setup

The project is designed to run on google colab.

## How to run :
Open the notebook directly on Colab:

https://colab.research.google.com/github/Chelle007/facial-expression-recognition/blob/main/ipynb/(name of the notebook)

E.g. Opening the CNN notebook
https://colab.research.google.com/github/Chelle007/facial-expression-recognition/blob/main/ipynb/CNN_v3.ipynb

### Dependencies
Everything is auto-installed in google colab

---

# How to run the web demo

## Structure
 
Make sure to download all files into one folder

### Option 1: Run locally

1.⁠ ⁠Open the folder and double-click "index.html"
2.⁠ ⁠Allow camera access
3.⁠ ⁠Click the open camera button to begin
4.⁠ ⁠Freely switch models 
5.⁠ ⁠Click the stop camera to turn off the camera

### Option 2: Run via Local Server (Recommended)

Make sure python is installed

1.⁠ ⁠Navigate to the folder
2.⁠ ⁠Run this command
 python -m http.server 8000
3.⁠ ⁠Open your browser and visit:
  http://localhost:8000

### Option 3: Using VS Code Live Server Extension

Make sure the Live Server Extension for VS Code is installed

1.⁠ ⁠Open the folder in VS Code
2.⁠ ⁠Right-click "index.html" and choose "Open with live server
