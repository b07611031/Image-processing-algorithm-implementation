# Feature Detection and Feature Description

## Introduction

This repository provides a Python implementation for detecting feature points and describing them as feature vectors, which are essential for tasks such as object recognition, image stitching, and 3D reconstruction. The implementation includes multiple algorithms for feature detection and description, allowing users to explore and understand these concepts hands-on.

Key Features:
- Feature detection using **Harris Corner** and **Multiscale Harris Corner** algorithms.
- Feature filtering using **Strongest Points** and **Adaptive Non-Maximal Suppression (ANMS)**.
- Feature description using **MSOP (Multi-Scale Oriented Patches)** with patch extraction and Haar wavelet transformation.

Follow the instructions below to install and use this repository.

---

## Step 1: Detect Feature Points

In this step, we implement two feature detection algorithms:
1. **Harris Corner Detection** [1]
2. **Multiscale Harris Corner Detection** [2]  

Multiscale Harris Corner applies a 3x3 convolution window to the image and retains the local maxima within the window. The detected points can then be filtered using:
- **Strongest Points**: Selects points with the highest intensity values.
- **ANMS (Adaptive Non-Maximal Suppression)**: Ensures spatially distributed points by suppressing nearby points with lower intensity.

**Example Outputs:**
The following examples demonstrate feature detection using Multiscale Harris Corner and filtering using Strongest Points and ANMS.

- **Input Image**:  
  ![Input Image](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/cat.jpg)

- **Harris Corner Detection**:  
  ![Harris Corner](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/Harris.png)

- **Multiscale Harris Corner Detection**:  
  ![Multiscale Harris Corner](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/multiHarris.png)

- **Strongest 100 Points**:  
  ![Strongest 100](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/cat_multiHarris_max_100.png)

- **ANMS 100 Points (Radius = 54)**:  
  ![ANMS 100](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/cat_multiHarris_anms_100.png)

- **Strongest 250 Points**:  
  ![Strongest 250](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/cat_multiHarris_max_250.png)

- **ANMS 250 Points (Radius = 30)**:  
  ![ANMS 250](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/cat_multiHarris_max_250_1.png)

---

## Step 2: Describe Feature Points

After detecting feature points, we implement the feature description method from the **MSOP (Multi-Scale Oriented Patches)** algorithm [2]. This involves the following steps:

1. **Orientation Calculation**: Compute the orientation of the feature point.
2. **Patch Extraction**: Extract a \(40x40\) pixel patch around the feature point from the original image.
3. **Patch Resizing**: Resize the patch to \(8x8\) pixels.
4. **Feature Vector Generation**: Transform the patch using Haar wavelet transformation and normalize it to have a mean intensity of 0 and standard deviation of 1.

**Example Output**:
Below is an example of an extracted \(8x8\) patch after standardization.

![Normalized 8x8 Patch](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/Feature-Detection-Description/figures/cat_patch.png)

The extracted patch has been normalized to a mean brightness of 0 and a standard deviation of 1.

---

## References

[1] Harris, C., & Stephens, M. (1988, August). A combined corner and edge detector. In *Alvey vision conference* (Vol. 15, No. 50, pp. 10-5244).

[2] Brown, M., Szeliski, R., & Winder, S. (2005, June). Multi-image matching using multi-scale oriented patches. In *2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)* (Vol. 1, pp. 510-517). IEEE.