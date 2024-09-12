# Image Classification: Sea vs Forest

## Overview

This repository contains a project for binary image classification, distinguishing between sea and forest images. The project includes a custom classifier based on color features extracted from images. The classifier is evaluated using various metrics including accuracy, confusion matrix, precision, and recall. 

## Contents

- `image_classification.py`: Python script for the image classification task using color features and Gaussian Naive Bayes.
- `images/`: Directory containing the images used for training and testing the classifier.
- `README.md`: This file, providing an overview and instructions.

## Dataset

The dataset consists of images categorized into two classes: sea and forest. Images are stored in the `images/` directory, and the classification is based on color features extracted from these images. 

## Implementation

### `image_classification.py`

This script performs the following steps:

1. **Data Preparation**:
   - Load images from the `images/` directory.
   - Extract color features (average Red, Green, and Blue values) from each image.

2. **Feature Extraction**:
   - Calculate the average RGB values for each image to be used as features for classification.

3. **Label Encoding**:
   - Convert categorical labels ('sea' and 'forest') to numeric values for model training.

4. **Model Training**:
   - Train a Gaussian Naive Bayes classifier using the extracted color features.

5. **Model Evaluation**:
   - Evaluate the classifier using accuracy, confusion matrix, precision, and recall metrics.

6. **Error Analysis**:
   - Identify and report misclassified images and discuss the potential reasons for misclassification based on the features used.

### Code Overview

```python
import numpy as np
import pandas as pd
import cv2 as cv
import os
import sklearn.model_selection as sms
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ================================
# Image Classification using Color Features
# This script performs image classification based on color features
# using a Naive Bayes Classifier from Scikit-Learn.
# ================================

# Load images from directory
# ================================
images = os.listdir("./image")  # List all image files in the directory

# Function to extract color features from images
# ================================
def extract_colors(files):
    dataset = pd.DataFrame({'label': [], 'R': [], 'G': [], 'B': [], 'name': []})
    for j in range(len(files)):
        img = cv.imread("./image/" + files[j])
        avgR = round(np.mean(img[:, :, 2]), 3)  # Average Red value
        avgG = round(np.mean(img[:, :, 1]), 3)  # Average Green value
        avgB = round(np.mean(img[:, :, 0]), 3)  # Average Blue value
        dataset.loc[j] = [files[j][0], avgR, avgG, avgB, files[j]]
    return dataset

# Extract features from images
# ================================
raw_dataset = extract_colors(images)  # Extract color features from images

# Convert qualitative labels to numerical ones
# ================================
def label_to_numeric(column):
    if column.dtype == 'object':
        unique_labels, _ = pd.factorize(column)
        return pd.Series(unique_labels, index=column.index)
    return column

dataset = raw_dataset.apply(label_to_numeric)  # Convert labels to numerical values

# Prepare the data for training and testing
# ================================
features = list(set(dataset.columns) - {'label', 'name'})  # Define feature columns
label = dataset['label']  # Define label column
data = dataset[features]  # Feature data

# Split data into training and testing sets
# ================================
data_trainset, data_testset, label_trainset, label_testset = sms.train_test_split(data, label, test_size=0.2)

# Train Gaussian Naive Bayes classifier
# ================================
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(data_trainset, label_trainset)  # Fit the model to the training data

# Predict labels for the test set
# ================================
prediction = naive_bayes_model.predict(data_testset)  # Predict on the test data

# Evaluate the classifier
# ================================
accuracy = accuracy_score(label_testset, prediction)  # Calculate accuracy
print("Accuracy:", accuracy, '\n\n')

# Create a DataFrame for comparison
# ================================
Comparison = pd.concat([label_testset.reset_index(drop=True), pd.Series(prediction)], axis=1)
Comparison.rename(columns={0: 'classifier'}, inplace=True)

# Calculate confusion matrices
# ================================
confusion_mtx = confusion_matrix(label_testset, prediction)
confusion_df = pd.DataFrame(confusion_mtx, columns=['Actual Positive', 'Actual Negative'], index=['Predicted Positive', 'Predicted Negative'])
print('Confusion Matrix:\n', confusion_df, '\n\n')

# Calculate Precision and Recall for each category
# ================================
def calculate_precision_recall(confusion_matrix):
    TP = confusion_matrix.loc['Predicted Positive', 'Actual Positive']
    FN = confusion_matrix.loc['Predicted Negative', 'Actual Positive']
    TN = confusion_matrix.loc['Predicted Negative', 'Actual Negative']
    FP = confusion_matrix.loc['Predicted Positive', 'Actual Negative']
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return precision, recall

precision, recall = calculate_precision_recall(confusion_df)
print('Precision:', precision, 'Recall:', recall)

# Find misclassified images
# ================================
misclassified = Comparison[Comparison['label'] != Comparison['classifier']]
result_list = misclassified.index.tolist()
Incorrect_isolation = pd.DataFrame(list(result_list)).rename(columns={0: 'name'})

print('Incorrect isolation is \n', Incorrect_isolation)
```

## Results

- **Accuracy**: 88.24%
- **Confusion Matrix**:
  - Category Sea:
    - True Positives: 8
    - False Negatives: 1
    - True Negatives: 7
    - False Positives: 1
  - Category Forest:
    - True Positives: 7
    - False Negatives: 1
    - True Negatives: 8
    - False Positives: 1
- **Precision**:
  - Sea: 0.89
  - Forest: 0.88
- **Recall**:
  - Sea: 0.89
  - Forest: 0.88

### Misclassified Images

The following images were misclassified:
- `s27.jpg`
- `j44.jpg`

**Discussion**: The misclassifications could be due to the similarity in color features between sea and forest images. Further feature extraction methods or advanced classifiers could improve the accuracy.

## Running the Code

To run the classification script, ensure you have the required libraries installed and the dataset placed in the `images/` directory. Then execute the script as follows:

```bash
python image_classification.py
```

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `opencv-python`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install numpy pandas opencv-python scikit-learn
```
