# Computer Vision with K-Neighbors Classifier and Convolutional Neural Networks

## Overview

This project demonstrates the application of machine learning and deep learning techniques to classify images of dogs and roads. It involves training a simple K-neighbors classifier and neural networks, including convolutional neural networks (CNNs). Additionally, it showcases the use of a saliency map for explainable AI.

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

Skills Learned
Data Preprocessing: Learned to standardize and prepare data for training machine learning models.
K-Neighbors Classifier: Gained experience in training and evaluating a simple K-neighbors classifier for image classification tasks.
Neural Networks: Learned to implement and train multi-layer perceptrons (MLPs) for classification.
Convolutional Neural Networks (CNNs): Developed skills in building, training, and evaluating CNNs for more complex image classification problems.
Model Evaluation: Learned to use accuracy scores and visualizations to evaluate and compare model performance.
Explainable AI: Implemented saliency maps to make the AI model's decisions interpretable and explainable.
Applications
The skills and techniques learned in this project can be applied to various fields, including:

Computer Vision: Developing models for image recognition, object detection, and image segmentation.
Medical Imaging: Applying CNNs to analyze and classify medical images for diagnostics.
Autonomous Vehicles: Using image classification to identify objects and make driving decisions.
Retail: Implementing image recognition for product identification and inventory management.
Security: Developing surveillance systems that can detect and classify objects in real-time.
This project provides a foundation for further exploration and application of machine learning and deep learning in various domains.

