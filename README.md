# Brain Tumor Classification using Convolutional Neural Networks (CNN)
This project implements a Convolutional Neural Network (CNN) model to classify brain tumors using MRI images. The model was developed for the classification of three types of tumors: Glioma, Meningioma, and Pituitary Tumors, along with normal cases. The main goal is to assist in automating the diagnosis process and improve the accuracy and speed of brain tumor detection.

## Table of Contents
* Project Overview
* Requirements
* How to Run
* Results

## Project Overview
The project applies CNNs to brain tumor classification, using the PyTorch framework. The dataset consists of MRI images and the model was trained and evaluated with various performance metrics such as accuracy, precision, Hamming Loss, and RMSE.

Key features:
* Custom CNN model implementation.
* MRI-based brain tumor classification.
* Performance evaluation using cross-validation.

## Requirements
Before running the project, ensure you have the following installed:
* Python 3.7+
* PyTorch
* Torchvision
* PIL (Pillow)
* Matplotlib
* tqdm
Install the necessary dependencies by running:

## How to Run
1. Clone the repository:

```
git clone https://github.com/thomazdsm/BrainTumorClassification.git
cd BrainTumorClassification
```

2. Prepare the dataset:
* Download the Brain Tumor MRI Dataset from Kaggle.
* Extract the dataset into the data/ directory.

3. Train the model:

```
python main.py
```

## Results
The model showed promising results with high accuracy on the training data and reasonable generalization on validation data. Future improvements include applying data augmentation and early stopping to prevent overfitting.