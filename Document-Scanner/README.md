# Document Scanner

## Overview

The **Document Scanner** project aims to provide a simple and efficient way to scan photos of documents and convert them into clean, rectangular, and binarized digital images. This program is designed to help users digitize documents for records or printing purposes, removing irrelevant information from the photo for an improved reading experience.

---

## Objectives

This project utilizes image processing techniques to achieve the following:
1. **Detect objects** in the photo.
2. **Match the document features** with detected objects.
3. **Apply perspective transformation** to rectify and crop the document.
4. **Binarize the final image** for better readability.

The program is packaged with a user-friendly graphical user interface (GUI) for easy operation and customization.

---

## Research Methods

### 1. Object Detection
- The input RGB image is converted to the HSV color space.
- The **saturation channel** is used as input for object detection.
- **Gaussian blur** is applied to reduce image noise.
- A **Laplacian filter** is used to detect contours in the image.
- Only **closed contours** are preserved as potential document features.

### 2. Document Feature Matching
The program differentiates documents in the image using:
1. **Binarization**:  
   - The saturation channel is binarized using a threshold calculated by **Otsu’s method**.
   - Documents with a white background are distinguished from the rest of the image.
2. **Filtration**:  
   - Contours are sorted by size.
   - The contour with the **largest area** is selected as the document feature.

### 3. Perspective Transformation
- The selected document contour is transformed into a new rectangular image using geometric transformation.
- The contour is simplified using the **Ramer–Douglas–Peucker algorithm** to remove unnecessary edges.
- A rectangle with the **largest area** is fitted to the simplified contour, and the transformation is applied to map the rectangle to a new image.

### 4. Image Binarization
- The transformed image is binarized using **Otsu’s method** to enhance readability.

---

## Results

### Image Processing Workflow
Figure 1 illustrates the key steps of the image processing workflow, showing the progression from the raw image to the final processed document.

![Image Processing Results](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Document-Scanner/figures/results.png)  
**Figure 1: Image processing results.**

### User-Friendly GUI
The program is packaged as a Windows Executable File (EXE) with a graphical user interface (GUI). Key features of the GUI include:
- **Brightness, contrast, and rotation adjustments** for enhanced results.
- A simple and intuitive interface for scanning and saving processed documents.

![GUI of the Scanner](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Document-Scanner/figures/gui.png)  
**Figure 2: The GUI of the developed scanner.**

For a detailed demonstration, refer to the included video file: [demo.mkv](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Document-Scanner/figures/demo.mkv).

### Performance
- The program processes images efficiently and delivers acceptable results in most cases.
- Limitations:
  - **Complex backgrounds** or **multiple lines** in the image can reduce accuracy.
  - These are common constraints of traditional image processing techniques.

---

## Future Work

To build a more robust scanner capable of handling a wider variety of scenarios, **deep learning-based image segmentation** can be considered. Deep learning models can generalize better to real-world variations, improving the accuracy and reliability of document detection.

---

## References

1. Gonzalez, R. C., & Woods, R. E. (2008). *Digital Image Processing*. Upper Saddle River, N.J.: Prentice Hall.