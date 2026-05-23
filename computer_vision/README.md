# 🔋 Capstone Redbacks Project 7 – Computer Vision

> **Battery detection and segmentation for safe robotic e-waste extraction.**  
> Built by the Computer Vision Sub-Team, Redbacks Project 7 — Deakin University.

---

## ⚡ Quickstart

```bash
# 1. Clone
git clone https://github.com/moksh07b/Capstone-Redback-Project_7-Computer_Vision.git
cd Capstone-Redback-Project_7-Computer_Vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
bash download_dataset.sh          # Linux / Mac
download_dataset.bat              # Windows

# 4. Run inference on an image
yolo task=segment mode=predict model=runs/segment/train/weights/best.pt source=your_image.jpg
```

---

## 📌 Project Overview

Lithium batteries in mobile phones are a significant hazard during recycling — they can cause **fire, explosion, and toxic leakage** if punctured or mishandled.

This module uses deep learning-based **object detection and segmentation** to accurately locate batteries inside disassembled phones, enabling a robotic arm to safely extract them.

The pipeline covers:
- Real + synthetic dataset preparation
- Multi-model training (YOLO, RetinaNet, SSD)
- Quantitative performance evaluation
- Recommendations for robotic deployment

---

## 🎯 Objectives

- Detect and segment lithium batteries in mobile devices using computer vision
- Train and compare multiple detection architectures
- Evaluate speed vs. accuracy trade-offs for real-time robotic use
- Build a scalable, reproducible training and evaluation pipeline
- Support future integration with ROS / Isaac Sim / MATLAB

---

## 🧠 Models Implemented

| Model | Type | Role |
|---|---|---|
| YOLOv8n-seg | Segmentation | Best deployment balance |
| YOLOv11x-seg | Segmentation | Highest accuracy |
| YOLO26n-seg | Segmentation | Experimental comparison |
| RetinaNet | Bounding Box | Baseline detector |
| SSD | Bounding Box | Lightweight baseline |

---

## 📂 Repository Structure

```text
Capstone-Redback-Project_7-Computer_Vision/
│
├── assets/                        ← Evaluation outputs per model
│   ├── YOLOv8n-seg/
│   ├── YOLOv11x-seg/
│   ├── YOLO26n-seg/
│   ├── SSD/
│   ├── RetinaNet/
│   ├── confusion_matrix.png
│   ├── f1_curve.png
│   ├── pr_curve.png
│   └── results.png
│
├── dataset/                       ← Downloaded from Kaggle (not in repo)
│   ├── images/train|val|test/
│   ├── labels/train|val|test/
│   ├── annotations_coco/
│   └── data.yaml
│
├── docs/
│   ├── Model Research Report.pdf  ← Full model evaluation report
│   └── makingNewAutomationBlenderFile.md
│
├── scripts/
│   ├── augment_dataset.py         ← Data augmentation (run once before training)
│   ├── blenderGenerationScript.py ← Synthetic image generation (Blender)
│   ├── check_dataset.py           ← Image-label matching validation
│   ├── evaluate_all_metrics.py    ← Full evaluation of all 5 models → CSV
│   ├── evaluate_ssd_retinanet.py  ← SSD / RetinaNet curve plots
│   ├── imgscraper.py              ← YouTube scraper + SAM auto-segmentation
│   ├── visual_compare_all_models.py ← Side-by-side visual comparison of all models
│   └── YoutubeScraper.md          ← YouTube scraper documentation
│
├── requirements.txt
├── download_dataset.sh            ← Linux/Mac dataset downloader
├── download_dataset.bat           ← Windows dataset downloader
├── .gitignore
└── README.md
```

---

## 📥 Dataset

The dataset is hosted on Kaggle due to size limitations.

🔗 **Kaggle Dataset:** https://www.kaggle.com/datasets/mokshbansal07/project-7

The dataset combines **real-world disassembly images** and **Blender-generated synthetic images**.

### Dataset Structure

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations_coco/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── data.yaml
```

All labels follow **YOLO segmentation format** (polygon masks).

### data.yaml

```yaml
path: dataset

train: images/train
val:   images/val
test:  images/test

names:
  0: battery
```

---

## ⚙️ Installation

### Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- Ultralytics YOLO
- OpenCV, NumPy, Pandas, Matplotlib

### 1. Clone

```bash
git clone https://github.com/moksh07b/Capstone-Redback-Project_7-Computer_Vision.git
cd Capstone-Redback-Project_7-Computer_Vision
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

**Linux / Mac:**
```bash
bash download_dataset.sh
```

**Windows:**
```cmd
download_dataset.bat
```

**Manual (any OS):**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\YOU\.kaggle\ (Windows)
mkdir dataset
kaggle datasets download -d mokshbansal07/project-7 -p dataset --unzip
```

---

## 🔧 Scripts

### `augment_dataset.py` — Data Augmentation

Run **once before training** to expand the dataset. For each original image, creates 4 augmented copies with transformed images and correctly updated segmentation polygon labels.

Augmentations applied: horizontal flip, brightness/contrast, hue/saturation, rotation (±15°), scale/translate, Gaussian blur.

> ⚠️ Safe to re-run — automatically skips images already containing `_aug` in the filename.

**Run:**
```bash
python scripts/augment_dataset.py
```

---

### `blenderGenerationScript.py` — Synthetic Data Generation

Run inside **Blender's Scripting IDE** to generate labelled synthetic phone images.

The script:
- Randomly positions the camera and light source around each phone model
- Randomises background materials from a `material_folder/`
- Iterates through all phone models in the `PhoneInsides` collection
- Renders 10 images per phone model into `output_folder/`

**Requirements:**
- Blender with a scene containing: `Camera`, `Light`, `Cube` (background), `PhoneInsides` collection, `Border` collection
- A `material_folder/` with `.blend` material files

```text
# Run from inside Blender → Scripting tab → Open file → Run Script
```

---

### `check_dataset.py` — Dataset Validation

Validates that every image in the dataset has a matching label file, and flags any orphaned images or labels.

**Run:**
```bash
python scripts/check_dataset.py
```

---

### `evaluate_all_metrics.py` — Full Model Evaluation

Evaluates all 5 models (YOLOv8n, YOLOv26n, YOLOv11x, SSD, RetinaNet) on the test set and produces a single comparison CSV.

**Metrics calculated:** Mask IoU / Box IoU, Average and Median Centroid Error (px), Inference Latency (ms), FPS.

**Output:** `runs/evaluation_results/model_metrics_summary.csv`

> ⚠️ Update the YOLO model folder names inside the script before running — see the comment block above `YOLO_MODELS`.

**Run:**
```bash
python scripts/evaluate_all_metrics.py
```

---

### `evaluate_ssd_retinanet.py` — SSD / RetinaNet Curve Plots

Generates precision, recall, F1, precision-recall curves and confusion matrix for SSD or RetinaNet.

**Output:** 5 `.png` files saved to `runs/evaluation_results/`

> Set `MODEL_TYPE = "ssd"` or `MODEL_TYPE = "retinanet"` at the top of the script before running.

**Run:**
```bash
python scripts/evaluate_ssd_retinanet.py
```

---

### `imgscraper.py` — YouTube Scraper + Auto-Segmentation

Downloads phone teardown/repair videos from a YouTube playlist, extracts scene-change frames using FFMPEG, and auto-segments the battery using **Meta's Segment Anything Model (SAM)**.

**Pipeline:**
1. Downloads each video from the playlist (mp4, highest resolution)
2. Extracts frames at scene changes (threshold: 0.3)
3. Filters blurry frames (Laplacian variance < 100)
4. Runs SAM point-prompted segmentation (centre point)
5. Crops and saves masked PNGs to `dataset/scraped/masked/`

**Requirements:**
```bash
pip install pytubefix opencv-python ffmpeg-python segment-anything torch pillow
# Download SAM checkpoint:
# https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth
# Place at: scripts/sam_vit_b_01ec64.pth
```

**Run:**
```bash
python scripts/imgscraper.py
```

> 📌 **For future students:** The segmentation quality is imperfect — the centre-point prompt sometimes captures hands or background. Priority improvements: smarter prompting strategy, deduplication to avoid re-scraping already-processed videos, and filtering out wide-angle/non-battery frames. Contribute teardown videos to the playlist linked in `YoutubeScraper.md`.

---

### `visual_compare_all_models.py` — Visual Comparison

Randomly samples 5% of test images and generates side-by-side comparison images showing each model's prediction vs the ground truth mask, with IoU and centroid error annotated.

**Output:** Comparison `.png` files saved to `runs/evaluation_results/`

> ⚠️ Update the YOLO model folder names inside the script before running — see the comment block above `YOLO_MODELS`.

**Run:**
```bash
python scripts/visual_compare_all_models.py
```

---

## 🚀 Training

### YOLOv8n-seg

```bash
yolo task=segment mode=train model=yolov8n-seg.pt data=dataset/data.yaml epochs=50 imgsz=640 batch=4 device=0
```

### YOLOv11x-seg

```bash
yolo task=segment mode=train model=yolo11x-seg.pt data=dataset/data.yaml epochs=50 imgsz=640 batch=4 device=0
```

### YOLO26n-seg

```bash
yolo task=segment mode=train model=yolov26n-seg.pt data=dataset/data.yaml epochs=50 imgsz=640 batch=4 device=0
```

Training outputs are saved to `runs/segment/train/`.

---

## 🧪 Validation

```bash
yolo task=segment mode=val model=runs/segment/train/weights/best.pt data=dataset/data.yaml device=0
```

---

## 📊 Results & Analysis

Full quantitative results are available in:
- Per-model `results.csv` files inside `assets/<ModelName>/`
- [`docs/Model Research Report.pdf`](docs/Model%20Research%20Report.pdf)

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

- High True Positives across all YOLO models
- Very low False Positives
- Minimal False Negatives

### F1 Score Curve

![F1 Curve](assets/f1_curve.png)

- Best F1 at moderate confidence (0.35–0.50)
- Precision increases / Recall decreases at higher confidence
- **Recommended inference threshold: 0.35–0.50**

### Precision-Recall Curve

![PR Curve](assets/pr_curve.png)

### Training Results

![Training Results](assets/results.png)

---

## 📈 Model Comparison

| Model | mAP@50 | Inference (ms) | FPS | Avg Centroid Error (px) | Notes |
|---|---|---|---|---|---|
| YOLOv8n-seg | 0.99 | 51.75 | 19.32 | 25.37 | Best deployment balance |
| YOLOv11x-seg | 0.99 | 72.95 | 13.71 | 41.22 | Highest accuracy |
| YOLO26n-seg | 0.98 | 60.21 | 16.61 | 34.63 | Experimental |
| RetinaNet | 0.90–0.97 | 81.98 | 12.20 | 133.77 | Baseline detector |
| SSD | 0.88–0.96 | 24.83 | 40.28 | 254.48 | Fast but unreliable |

---

## 🏆 Final Recommendation

### ✅ Best for Deployment — YOLOv8n-seg

- Real-time inference speed (19.32 FPS)
- Lowest centroid error (25.37 px) — most precise robotic targeting
- Low parameter count (3.4M) — efficient on embedded hardware
- Proven Ultralytics ecosystem support

### 🧠 Best for Accuracy — YOLOv11x-seg

- Highest precision, recall, and F1 score
- Best mask quality (Mask IoU: 0.866)
- Best choice for offline or non-latency-constrained evaluation

---

## 🔮 Future Work

- [ ] Improve SAM segmentation prompting in `imgscraper.py`
- [ ] Add video deduplication to avoid re-scraping the same teardowns
- [ ] Integration with robotic arm (ROS / Isaac Sim / MATLAB)
- [ ] Expand synthetic dataset via Blender automation
- [ ] Multi-object detection (battery + PCB + connectors)
- [ ] Real-world deployment and latency testing on target hardware

---

## 🖥️ Hardware & Environment

Training was performed using:
- **GPU:** NVIDIA GeForce RTX 4060 Laptop
- **CUDA:** 12.6 (Driver: 581.83)
- **Python:** 3.13.5
- **OS:** Windows 11 Home (Version 25H2, Build 26200.8457)

GPU is strongly recommended. CPU-only training is possible but very slow.

---

## 📋 Recommended `.gitignore`

```gitignore
dataset/
runs/
*.pt
*.zip
*.mp4
*.cache
__pycache__/
.kaggle/
```

---


## 📄 Documentation

- [Model Research Report (PDF)](docs/Model%20Research%20Report.pdf)
- [Blender Automation Guide](docs/makingNewAutomationBlenderFile.md)
- [YouTube Scraper Guide](scripts/YoutubeScraper.md)
