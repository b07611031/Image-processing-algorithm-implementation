# High Dynamic Range Imaging (HDR)

This repository provides a Python implementation for converting Low Dynamic Range (LDR) images into High Dynamic Range (HDR) images using alignment, mapping, and tone mapping techniques. Follow the instructions below to install and use this tool effectively.

---

## Getting Started

### Installation

1. Clone the repository to your local machine.
2. Navigate to the `code` directory.
3. Install the required dependencies using `pip`:

```bash
cd code
pip install -r requirements.txt
```

---

### Generating HDR Images

Run the following command to generate an HDR image from a set of LDR images:

```bash
python main.py <LDR-image-set> --savename=<save-name> --alignment=<True/False>
```

**Parameters:**
- `<LDR-image-set>`: Path to the folder containing your LDR image set.
- `--savename`: Name of the output HDR image.
- `--alignment`: (Optional) Set to `True` to align images before generating the HDR image.

**Example Usage:**
```bash
python main.py ./vending --savename=vending
python main.py ./vending --savename=vending_align --alignment=True
```

Output images will be saved in the `./output` directory.

---

## Features

### Converting LDR Image Sets to HDR Images

HDR images can represent a broader range of brightness and color details compared to LDR images. This program performs the conversion in three steps:

1. **Image Alignment:** Align the LDR image set using the MTB algorithm.
2. **LDR to HDR Mapping:** Map the LDR images to HDR using the Debevec algorithm.
3. **Tone Mapping:** Adjust the HDR image for display compatibility using tone mapping.

---

### Step 1: Image Alignment Using MTB

The Median Threshold Bitmap (MTB) algorithm aligns LDR images by identifying optimal offsets based on median brightness values. This ensures proper alignment, especially when images have slight shifts during capture.

Run the program with `--alignment=True` to enable alignment. For example:
```bash
python main.py ./vending --savename=vending_align --alignment=True
```

**Example Outputs:**
- Without alignment:
  ![HDR without alignment](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/vending.jpg)

- With alignment (max_offset=8):
  ![HDR with alignment](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/vending_align.jpg)

**Note:** The MTB algorithm is not designed to handle significant rotation or scaling. For best results, ensure stable lighting and minimize movement during image capture.

---

### Step 2: LDR to HDR Mapping

The LDR-to-HDR mapping uses the method proposed by Debevec et al. to estimate scene radiance from multiple exposure images.

To optimize computation, this implementation samples \(n\) random points from the images instead of processing all pixels. Adjust the sampling size with the `--sample-size` parameter. Larger sample sizes improve quality but increase processing time.

**Tips for Optimizing Mapping:**
- Experiment with `--sample-size` (e.g., \(n=500, 2000, 5000\)).
- Consider preprocessing images by resizing to reduce computation time.

---

### Step 3: Tone Mapping

Tone mapping converts HDR images into LDR images suitable for standard displays. This step ensures that details in highlights and shadows are preserved while adapting to display limitations.

Adjust the gamma value using the `--gamma` parameter to control brightness and contrast.

**Example Results:**
- \(n=500\), \(\gamma=1.5\):
  ![Gamma = 1.5](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/hdrtonemapping.jpg)

- \(n=500\), \(\gamma=2.2\):
  ![Gamma = 2.2](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/hdrtonemapping_1.jpg)

---

## Comparing Results

You can compare the generated HDR images with results from OpenCV's HDR package to evaluate quality. Below is an example comparison:

- **Original LDR Image (Exposure Time = 0.4 sec):**
  ![Raw (0.4 sec)](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/raw_04.jpg)

- **Our HDR Result (\(n=500, \gamma=2.2\)):**
  ![Our HDR result](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/hdrtonemapping_1.jpg)

- **OpenCV HDR Result:**
  ![OpenCV HDR](https://github.com/b07611031/Image-processing-algorithm-implementation/blob/main/High-Dynamic-Range-Imaging/figures/fusion_mertens.jpg)

---

## References

1. Ward, G. (2003). Fast, robust image registration for compositing high dynamic range photographs from hand-held exposures. *Journal of Graphics Tools*, *8*(2), 17–30.

2. Debevec, P. E., & Malik, J. (2008). Recovering high dynamic range radiance maps from photographs. In *ACM SIGGRAPH 2008 Classes* (pp. 1–10).