# Computer Vision with K-Neighbors Classifier and Convolutional Neural Networks

## Overview

This project demonstrates the application of machine learning and deep learning techniques to classify images of dogs and roads. It involves training a  K-neighbors classifier and neural networks, including convolutional neural networks (CNNs). Additionally, it showcases the use of a saliency map for explainable AI.

## Table of Contents

1. [Introduction](#introduction)
2. [Load Packages and Create the CNN Classifier](#load-packages-and-create-the-cnn-classifier)
3. [Functions](#functions)
   - [Categorical to One-hot Encoding](#categorical-to-one-hot-encoding)
   - [One-hot Encoding](#one-hot-encoding)
   - [Load Data](#load-data)
   - [Plot One Image](#plot-one-image)
   - [Logits to One-hot Encoding](#logits-to-one-hot-encoding)
4. [CNNClassifier Class](#cnnclassifier-class)
5. [Plot Accuracy Function](#plot-accuracy-function)
6. [Skills Learned](#skills-learned)
7. [Applications](#applications)

## Introduction

In this project, we cover several key aspects of computer vision and machine learning:

1. **Training a simple K-neighbors classifier**: A basic model to classify images.
2. **Training neural networks**: Implementing and training multi-layer perceptrons (MLPs) for classification tasks.
3. **Improving models with convolutional neural networks (CNNs)**: Leveraging CNNs to enhance model performance for image classification.
4. **Explainable AI using saliency maps**: Visualizing model predictions to understand what parts of an image contribute most to the decision.

## Load Packages and Create the CNN Classifier

We begin by importing necessary packages and defining the `CNNClassifier` class. This class encapsulates the architecture and training procedures for a CNN.
python
# Import necessary packages
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from collections import Counter
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Reshape
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Skills Learned

- **Data Preprocessing**: Learned to standardize and prepare data for training machine learning models.
- **K-Neighbors Classifier**: Gained experience in training and evaluating a simple K-neighbors classifier for image classification tasks.
- **Neural Networks**: Learned to implement and train multi-layer perceptrons (MLPs) for classification.
- **Convolutional Neural Networks (CNNs)**: Developed skills in building, training, and evaluating CNNs for more complex image classification problems.
- **Model Evaluation**: Learned to use accuracy scores and visualizations to evaluate and compare model performance.
- **Explainable AI**: Implemented saliency maps to make the AI model's decisions interpretable and explainable.

## Applications

The skills and techniques learned in this project can be applied to various fields, including:

- **Computer Vision**: Developing models for image recognition, object detection, and image segmentation.
- **Medical Imaging**: Applying CNNs to analyze and classify medical images for diagnostics.
- **Autonomous Vehicles**: Using image classification to identify objects and make driving decisions.
- **Retail**: Implementing image recognition for product identification and inventory management.
- **Security**: Developing surveillance systems that can detect and classify objects in real-time.

# CNN Classifier for Dog vs. Road Image Classification

## Import Statements
This section imports necessary libraries for machine learning, data manipulation, and visualization, including scikit-learn, TensorFlow/Keras, NumPy, Pandas, Seaborn, and Matplotlib.

## Utility Functions

### Categorical to One-Hot Encoding
`categorical_to_onehot(labels_in)`:
- Converts categorical labels ('dog' or 'road') to one-hot encoded vectors.
- 'dog' becomes [1, 0], 'road' becomes [0, 1].

### One-Hot Encoding
`one_hot_encoding(input)`:
- Performs one-hot encoding on numerical input.
- Creates a binary matrix representation of categorical data.

### Load Data
`load_data()`:
- Downloads CIFAR data from a cloud link.
- Loads the data into a dictionary containing 'data' and 'labels'.

### Plot One Image
`plot_one_image(data, labels, img_idx)`:
- Displays a single image from the dataset.
- Shows the image label and sets up the plot with proper axis labels.

### Logits to One-Hot Encoding
`logits_to_one_hot_encoding(input)`:
- Converts model output logits to one-hot encoded vectors.
- Used for transforming raw model predictions into classification results.

## CNN Classifier
`class CNNClassifier`:

### Initialization
- Sets up the model with customizable parameters:
  - `num_epochs`: Number of training iterations (default 30)
  - `layers`: Number of convolutional layers (default 4)
  - `dropout`: Dropout rate for regularization (default 0.5)

### Model Architecture
`build_model()`:
- Creates a Sequential model with the following structure:
  1. Input reshaping to 32x32x3
  2. Multiple convolutional layers with ReLU activation
  3. Max pooling and dropout for regularization
  4. Deeper convolutional layers with increased filters
  5. Flattening and dense layers for classification
  6. Final softmax layer for binary output

### Training Method
`fit(*args, **kwargs)`:
- Trains the model on given data.
- Uses specified number of epochs and a batch size of 10.

### Prediction Methods
- `predict(*args, **kwargs)`: Returns one-hot encoded predictions.
- `predict_proba(*args, **kwargs)`: Returns probability predictions.

### Scoring Method
`score(X, y)`:
- Calculates the accuracy of the model's predictions.

## Visualization Function
`plot_acc(history, ax = None, xlabel = 'Epoch #')`:
- Plots training and validation accuracy over epochs.
- Highlights the best epoch and shows a chance level line.
- Uses Seaborn for enhanced visualizations.



