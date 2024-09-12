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

# Load images and extract color features
images = os.listdir("./images")

def extract_colors(files):
    dataset = pd.DataFrame({'label': [], 'R': [], 'G': [], 'B': [], 'name': []})
    for j in range(len(files)):
        img = cv.imread("./images/" + files[j])
        avgR = round(np.mean(img[:, :, 2]), 3)
        avgG = round(np.mean(img[:, :, 1]), 3)
        avgB = round(np.mean(img[:, :, 0]), 3)
        dataset.loc[j] = [files[j][0], avgR, avgG, avgB, files[j]]
    return dataset

raw_dataset = extract_colors(images)

# Convert qualitative labels to numerical ones
def label_to_numeric(column):
    if column.dtype == 'object':
        unique_labels, _ = pd.factorize(column)
        return pd.Series(unique_labels, index=column.index)
    return column

dataset = raw_dataset.apply(label_to_numeric)

# Feature extraction and model training
features = list(set(dataset.columns) - {'label', 'name'})
label = dataset['label']
data = dataset[features]
data_trainset, data_testset, label_trainset, label_testset = sms.train_test_split(data, label, test_size=0.2)

naive_bayes_model = GaussianNB()
naive_bayes_model.fit(data_trainset, label_trainset)
prediction = naive_bayes_model.predict(data_testset)

# Evaluation
accuracy = accuracy_score(label_testset, prediction)
print("Accuracy:", accuracy, '\n\n')

# Confusion Matrix and Metrics
confusion_matarix = confusion_matrix(label_testset, prediction)
confusion_df = pd.DataFrame(confusion_matarix, columns=['Actual Positive', 'Actual Negative'],
                            index=['Predicted Positive', 'Predicted Negative'])

print(confusion_df, '\n\n')

TP = ((label_testset == 1) & (prediction == 1)).sum()
FN = ((label_testset == 1) & (prediction == 0)).sum()
TN = ((label_testset == 0) & (prediction == 0)).sum()
FP = ((label_testset == 0) & (prediction == 1)).sum()

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

print('Precision =', Precision, 'Recall =', Recall)

# Identify misclassified images
comparison = pd.concat([label_testset.reset_index(drop=True), pd.Series(prediction)], axis=1)
comparison.rename(columns={0: 'classifier'}, inplace=True)
misclassified = comparison[comparison['label'] != comparison['classifier']]
print('Misclassified images:', misclassified['name'].tolist())
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

