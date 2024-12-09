# Image Classification

This material was fully designed from scratch in August 2023 for the AIoT class. The class aims to provide students with a basic understanding of how to use image processing and machine learning for classification tasks. Students will gain hands-on experience using [Jupyter Notebook](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/image_classification.ipynb).

Image classification can be broadly divided into two parts: (1) feature extraction and description, and (2) classification.

![Image Framework](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/figures/framework.jpg)

---

## Feature Extraction & Description

In this step, we extract features such as color, texture, and shape from images.

![Image Feature Description](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/figures/feature-description.jpg)

### Color
Color features can be analyzed using color histograms or by extracting specific ranges of colors.

![Image Color](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/figures/color.jpg)

```python
import cv2
import matplotlib.pyplot as plt

apple = cv2.imread(path)
hsv = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV)

# Define red color masks
red_min = cv2.inRange(hsv, (0, 43, 46), (10, 255, 255))
red_max = cv2.inRange(hsv, (156, 43, 46), (180, 255, 255))
red = cv2.bitwise_or(red_min, red_max)

apple_color = cv2.bitwise_and(apple, apple, mask=red)
plt.imshow(apple_color)
plt.show()
```

---

### Texture
Texture refers to the visual and tactile quality of a surface, characterized by patterns, variations, and details. The Gray-Level Co-occurrence Matrix (GLCM) is often used for analyzing texture.

![Image Texture](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/figures/texture.jpg)  
![GLCM Example](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/figures/glcm.jpg)

```python
from skimage.feature import graycomatrix, graycoprops

for patch in patches:
    glcm = graycomatrix(patch, distances=[5], angles=[0])
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    correlation = graycoprops(glcm, 'correlation')
```

---

### Shape
Shape features, such as contours and Fourier descriptors, capture the distinctive properties of an object's boundary.

![Image Shape](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/figures/shape.jpg)

---

## Classification

After extracting features, we can classify images using various machine learning models. Below are a few examples:

---

### Decision Tree
```python
from sklearn import tree
from sklearn.metrics import roc_auc_score as ras
from sklearn.tree import export_text

# Train a decision tree model
decision_tree = tree.DecisionTreeClassifier(max_depth=None, random_state=random_state)
decision_tree.fit(X_train, Y_train.values)

# Evaluate accuracy
train_preds = decision_tree.predict_proba(X_train)
print('Training Accuracy:', ras(Y_train.values, train_preds, multi_class='ovr'))

valid_preds = decision_tree.predict_proba(X_valid)
print('Validation Accuracy:', ras(Y_valid.values, valid_preds, multi_class='ovr'))

# Display decision tree structure
r = export_text(decision_tree, feature_names=glcm_features)
print(r)
```

---

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train.values)

# Evaluate accuracy
train_preds = knn.predict_proba(X_train)
print('Training Accuracy:', ras(Y_train.values, train_preds, multi_class='ovr'))

valid_preds = knn.predict_proba(X_valid)
print('Validation Accuracy:', ras(Y_valid.values, valid_preds, multi_class='ovr'))
```

---

### Support Vector Machine (SVM)
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

svm = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', probability=True, random_state=random_state))
svm.fit(X_train, Y_train.values)

# Evaluate accuracy
train_preds = svm.predict_proba(X_train)
print('Training Accuracy:', ras(Y_train.values, train_preds, multi_class='ovr'))

valid_preds = svm.predict_proba(X_valid)
print('Validation Accuracy:', ras(Y_valid.values, valid_preds, multi_class='ovr'))
```

---

This README provides an overview of the key steps in image classification, along with examples of feature extraction and classification methods. For more details, please refer to the [Jupyter Notebook](https://github.com/b07611031/Image-processing-algorithm-implementation/Image-Classification/image_classification.ipynb).