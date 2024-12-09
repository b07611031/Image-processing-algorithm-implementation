# Image Stitching

**Team 18: R11631008 鍾承晏 & R11631012 林雲**

## 開始

### 安裝

```bash
cd code
pip install -r requirements.txt
```

### 生成全景影像(Panorama)

```bash
python main.py ./images --savename=output
python main.py ./images --savename=output --focal_length=1000

# python main.py <Panorama-image-set> --savename=<save-name, default=./output.jpg> --focal_length=<focal length, default=1000>
# Finally, the outputs will be saved to the ./output directory.
```