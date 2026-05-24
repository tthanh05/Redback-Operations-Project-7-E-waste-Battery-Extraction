# Computer Vision Module — E-waste Battery Extraction

> Battery detection and instance segmentation for safer robotic e-waste battery extraction.  
> This folder contains the computer vision notebooks, scripts, reports, trained weights, ONNX exports, and evaluation assets for the Redback Operations Project 7 e-waste battery extraction project.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLO](https://img.shields.io/badge/YOLO-Instance%20Segmentation-green)
![Detectron2](https://img.shields.io/badge/Detectron2-Mask%20R--CNN-orange)
![MMDetection](https://img.shields.io/badge/MMDetection-RTMDet--Ins-purple)
![ONNX](https://img.shields.io/badge/ONNX-Exported%20Models-lightgrey)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Purpose](#folder-purpose)
- [Main Outcomes](#main-outcomes)
- [Repository Structure](#repository-structure)
- [Folder and File Guide](#folder-and-file-guide)
- [Models Covered](#models-covered)
- [Dataset Requirement](#dataset-requirement)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Notebook Workflows](#notebook-workflows)
- [Using the Stored Weights](#using-the-stored-weights)
- [ONNX Exports](#onnx-exports)
- [Scripts](#scripts)
- [Training](#training)
- [Evaluation](#evaluation)
- [Synthetic Data Generation](#synthetic-data-generation)
- [YouTube Scraping and Auto-Segmentation](#youtube-scraping-and-auto-segmentation)
- [Results Summary](#results-summary)
- [Compatibility Notes](#compatibility-notes)
- [Known Repository Notes](#known-repository-notes)
- [Recommended Future Work](#recommended-future-work)
- [License and Acknowledgement](#license-and-acknowledgement)

---

## Project Overview

Lithium-ion batteries inside mobile phones are hazardous during e-waste processing. If a battery is punctured, bent, crushed, or mishandled, it may create fire, explosion, leakage, or worker-safety risks. This computer vision module supports safer battery extraction by detecting and segmenting visible phone batteries from 2D rear-phone images.

The main target class is:

```text
battery
```

The project focuses on single-class battery localisation using object detection and instance segmentation. The output from the perception model can support downstream robotic tasks such as:

- locating the battery region,
- estimating a battery mask,
- finding an approximate battery centroid,
- comparing model speed and accuracy,
- preparing deployment-friendly models for simulation or robotic integration.

---

## Folder Purpose

This `computer_vision/` folder is the main working area for the computer vision sub-team. It contains:

- training notebooks,
- evaluation notebooks,
- helper scripts,
- dataset preparation scripts,
- Blender synthetic-data generation scripts,
- YouTube scraping experiments,
- saved model weights,
- selected ONNX exports,
- visual evaluation outputs,
- written reports and project documentation.

This README is written as the main guide for the contents of the `computer_vision/` folder. It combines the older five-model documentation with the newer eight-model training and evaluation workflow.

---

## Main Outcomes

The project contains two main evaluation directions:

1. **Five-model workflow**
   - YOLOv8n-seg
   - YOLOv11x-seg
   - YOLO26n-seg
   - RetinaNet
   - SSD

2. **Eight-model instance segmentation workflow**
   - YOLOv5n-seg
   - YOLOv7-seg / YOLOv7-tiny-seg
   - YOLOv8n-seg
   - YOLO11n-seg
   - YOLO26n-seg
   - Mask R-CNN R50-FPN
   - PointRend Mask R-CNN
   - RTMDet-Ins tiny

The main practical recommendation from the reported results is that **YOLOv8n-seg** is the strongest deployment-oriented model because it provides a good balance between segmentation quality, inference speed, and centroid accuracy.

---

## Repository Structure

The current `computer_vision/` folder contains:

```text
computer_vision/
│
├── Documents/
│   ├── Cost evaluation on e-waste battery extraction.pdf
│   ├── E-waste battery dataset creation report.pdf
│   ├── Ewaste Battery Dataset Creation Report.pdf
│   ├── Literature Review and Dataset Collection for Robotic E.pdf
│   └── Model Evaluation for YOLO v8n, v11n and v26n.pdf
│
├── assets/
│   ├── RetinaNet/
│   ├── SSD/
│   ├── YOLO26n-seg/
│   ├── YOLOv11x-seg/
│   ├── YOLOv8n-seg/
│   ├── confusion_matrix.png
│   ├── f1_curve.png
│   ├── pr_curve.png
│   ├── precision_confidence_curve.png
│   ├── recall_confidence_curve.png
│   └── results.png
│
├── codes/
│   ├── E_waste_8_model_benchmark.ipynb
│   ├── E_waste_8_model_evaluation.ipynb
│   ├── Evaluation of YOLO models.ipynbp
│   └── RTMDet_Ins_Tiny_EWaste_Colab.ipynb
│
├── docs/
│   ├── Model Research Reprt.pdf
│   └── makingNewAutomationBlenderFile.md
│
├── model_benchmark_yehezkiel/
│   ├── code/
│   └── README.md
│
├── onnx/
│   ├── yolo11n-seg_best.onnx
│   ├── yolo26n-seg_best.onnx
│   └── yolov8n-seg_best.onnx
│
├── scripts/
│   ├── YoutubeScraper.md
│   ├── augment_dataset.py
│   ├── blenderGenerationScript.py
│   ├── check_dataset.py
│   ├── evaluate_all_metrics.py
│   ├── evaluate_ssd_retinanet.py
│   ├── imgscraper.py
│   └── visual_compare_all_models.py
│
├── weights/
│   ├── .gitattributes
│   ├── mask-rcnn_50-FPN_model_final.pth
│   ├── pointrend_model_final.pth
│   ├── rtmdet-ins_best.pth
│   ├── yolo11n-seg_best.pt
│   ├── yolo26n-seg_best.pt
│   ├── yolov5n-seg_best.pt
│   ├── yolov7-seg_best.pt
│   └── yolov8n-seg_best.pt
│
├── .gitattributes
├── .gitignore
├── README.md
├── UpdatedReadme.md
├── download_dataset.bat
├── download_dataset.sh
├── evaluation
├── makingNewAutomationBlenderFile.txt
└── requirements.txt
```

---

## Folder and File Guide

### `Documents/`

This folder stores written project documents and PDF reports. It is useful for understanding the broader project context, cost considerations, dataset creation process, literature review, and earlier YOLO model evaluation.

Files currently included:

| File | Purpose |
|---|---|
| `Cost evaluation on e-waste battery extraction.pdf` | Cost-related discussion for the e-waste battery extraction project. |
| `E-waste battery dataset creation report.pdf` | Dataset creation documentation. |
| `Ewaste Battery Dataset Creation Report.pdf` | Another version of the dataset creation report. |
| `Literature Review and Dataset Collection for Robotic E.pdf` | Literature review and dataset collection background. |
| `Model Evaluation for YOLO v8n, v11n and v26n.pdf` | Report focused on YOLOv8n, YOLOv11n/YOLO11, and YOLO26 model evaluation. |

---

### `assets/`

This folder stores evaluation outputs, plots, and visual evidence. It includes both general plots and model-specific result folders.

Main folders:

| Folder | Purpose |
|---|---|
| `RetinaNet/` | RetinaNet training/evaluation outputs and plots. |
| `SSD/` | SSD training/evaluation outputs and plots. |
| `YOLO26n-seg/` | YOLO26n segmentation result plots and validation outputs. |
| `YOLOv11x-seg/` | YOLOv11x segmentation result plots and validation outputs. |
| `YOLOv8n-seg/` | YOLOv8n segmentation result plots and validation outputs. |

Main root-level plot files:

| File | Purpose |
|---|---|
| `confusion_matrix.png` | Overall confusion matrix visualisation. |
| `f1_curve.png` | F1-confidence curve. |
| `pr_curve.png` | Precision-recall curve. |
| `precision_confidence_curve.png` | Precision-confidence curve. |
| `recall_confidence_curve.png` | Recall-confidence curve. |
| `results.png` | Training/evaluation summary plot. |

These files are useful for README figures, report screenshots, and qualitative model comparison evidence.

---

### `codes/`

This folder stores the main Jupyter/Google Colab notebooks.

| File | Purpose |
|---|---|
| `E_waste_8_model_benchmark.ipynb` | Main training and benchmarking notebook for YOLO, Detectron2, and refinement-based experiments. |
| `E_waste_8_model_evaluation.ipynb` | Final eight-model evaluation notebook, including quantitative metrics and qualitative comparison figures. |
| `Evaluation of YOLO models.ipynbp` | Older YOLO evaluation notebook-style file. The extension appears to be non-standard and may need to be renamed to `.ipynb` before use. |
| `RTMDet_Ins_Tiny_EWaste_Colab.ipynb` | Separate Colab notebook for RTMDet-Ins tiny using MMDetection in a controlled environment. |

Recommended notebook order:

1. `E_waste_8_model_benchmark.ipynb`
2. `RTMDet_Ins_Tiny_EWaste_Colab.ipynb`
3. `E_waste_8_model_evaluation.ipynb`

---

### `docs/`

This folder stores supporting documentation.

| File | Purpose |
|---|---|
| `Model Research Reprt.pdf` | Model research report. The filename appears to contain a spelling typo: `Reprt` instead of `Report`. |
| `makingNewAutomationBlenderFile.md` | Markdown guide for creating or modifying Blender automation files for synthetic dataset generation. |

---

### `model_benchmark_yehezkiel/`

This is a separate benchmark workspace inside the computer vision folder.

It contains:

```text
model_benchmark_yehezkiel/
├── code/
└── README.md
```

The nested README describes a benchmark on a smaller single-class e-waste battery dataset. It includes YOLO-family models and a proposed hybrid method called **BatteryMask-RefineNet**.

BatteryMask-RefineNet is described as a two-stage pipeline:

1. YOLOv8n-seg produces a coarse battery mask.
2. A lightweight U-Net-style refiner improves the mask boundary around the detected battery region.

This folder is useful for understanding the additional refinement experiment and the earlier benchmark setup.

---

### `onnx/`

This folder contains selected YOLO segmentation models exported to ONNX format.

| File | Related model | Purpose |
|---|---|---|
| `yolo11n-seg_best.onnx` | YOLO11n-seg | Deployment-oriented ONNX export. |
| `yolo26n-seg_best.onnx` | YOLO26n-seg | Deployment-oriented ONNX export. |
| `yolov8n-seg_best.onnx` | YOLOv8n-seg | Main deployment-oriented ONNX export. |

These files are mainly intended for model handover to non-training environments, such as MATLAB-side perception experiments, ONNX Runtime, TensorRT, or future robotic simulation workflows.

Current ONNX limitation:

```text
ONNX exports are included only for selected Ultralytics-compatible YOLO models.
ONNX exports for YOLOv5, YOLOv7, Detectron2, and RTMDet are not included in this folder.
```

---

### `scripts/`

This folder contains reusable scripts for dataset preparation, validation, scraping, synthetic generation, and evaluation.

| Script | Purpose |
|---|---|
| `YoutubeScraper.md` | Documentation for the YouTube scraping workflow and its limitations. |
| `augment_dataset.py` | Applies image augmentation and updates YOLO segmentation labels. |
| `blenderGenerationScript.py` | Runs inside Blender to generate synthetic phone images. |
| `check_dataset.py` | Checks image-label matching in the dataset. |
| `evaluate_all_metrics.py` | Evaluates multiple models and creates summary metrics. |
| `evaluate_ssd_retinanet.py` | Evaluates SSD/RetinaNet and generates curve plots. |
| `imgscraper.py` | Downloads teardown/repair videos, extracts frames, and uses SAM-assisted segmentation. |
| `visual_compare_all_models.py` | Creates side-by-side visual comparisons of model predictions and ground truth. |

---

### `weights/`

This folder stores trained checkpoints.

| File | Model |
|---|---|
| `mask-rcnn_50-FPN_model_final.pth` | Mask R-CNN R50-FPN |
| `pointrend_model_final.pth` | PointRend Mask R-CNN |
| `rtmdet-ins_best.pth` | RTMDet-Ins tiny |
| `yolo11n-seg_best.pt` | YOLO11n-seg |
| `yolo26n-seg_best.pt` | YOLO26n-seg |
| `yolov5n-seg_best.pt` | YOLOv5n-seg |
| `yolov7-seg_best.pt` | YOLOv7-seg / YOLOv7-tiny-seg |
| `yolov8n-seg_best.pt` | YOLOv8n-seg |

If the repository uses Git LFS and the downloaded files appear as small pointer files, run:

```bash
git lfs install
git lfs pull
```

---

### Root-level files inside `computer_vision/`

| File | Purpose |
|---|---|
| `.gitattributes` | Git/Git LFS tracking rules. |
| `.gitignore` | Ignore rules for generated files, datasets, runs, caches, and large local artefacts. |
| `README.md` | Main documentation file for this folder. |
| `UpdatedReadme.md` | Additional/older README-style documentation. Useful as reference, but should be merged into this main README to avoid duplication. |
| `download_dataset.bat` | Windows dataset download helper. |
| `download_dataset.sh` | Linux/Mac dataset download helper. |
| `evaluation` | Very small placeholder-style file. It does not appear to be a full evaluation directory. |
| `makingNewAutomationBlenderFile.txt` | Text version of the Blender automation guide. |
| `requirements.txt` | Python dependency list. |

---

## Models Covered

### Main segmentation models

| Model | Framework / Toolchain | Checkpoint included | ONNX included | Notes |
|---|---|---:|---:|---|
| YOLOv5n-seg | YOLOv5 segmentation repository | Yes | No | Lightweight YOLO baseline. |
| YOLOv7-seg / YOLOv7-tiny-seg | YOLOv7 segmentation repository | Yes | No | Requires YOLOv7-specific setup. |
| YOLOv8n-seg | Ultralytics | Yes | Yes | Main deployment recommendation. |
| YOLO11n-seg | Ultralytics | Yes | Yes | Newer YOLO comparison model. |
| YOLO26n-seg | Ultralytics-compatible workflow | Yes | Yes | Experimental YOLO comparison model. |
| Mask R-CNN R50-FPN | Detectron2 | Yes | No | Two-stage instance segmentation baseline. |
| PointRend Mask R-CNN | Detectron2 + PointRend | Yes | No | Boundary-aware segmentation model. |
| RTMDet-Ins tiny | MMDetection/MMYOLO-style workflow | Yes | No | Lightweight instance segmentation experiment. |

### Additional detection baselines

| Model | Type | Notes |
|---|---|---|
| RetinaNet | Object detection | Detection baseline; mainly bounding-box based. |
| SSD | Object detection | Fast baseline; useful for speed comparison, but less reliable for precise robotic targeting. |

---

## Dataset Requirement

The private project dataset is not fully included in this folder. The notebooks expect a YOLO segmentation dataset with the following structure:

```text
final_dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The label format is YOLO polygon segmentation format:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

For this project:

```text
class_id = 0
class_name = battery
```

A typical `dataset.yaml` file is:

```yaml
path: /content/drive/MyDrive/E-waste Battery Extraction CV/final_dataset
train: images/train
val: images/val
test: images/test

nc: 1
names:
  - battery
```

Some older scripts also expect a dataset folder named:

```text
dataset/
```

with this layout:

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

Before running notebooks or scripts, check the dataset path variables carefully.

---

## Installation

### 1. Clone the branch

From the parent repository:

```bash
git clone -b computer-vision https://github.com/tthanh05/Redback-Operations-Project-7-E-waste-Battery-Extraction.git
cd Redback-Operations-Project-7-E-waste-Battery-Extraction/computer_vision
```

If using Git LFS for large model files:

```bash
git lfs install
git lfs pull
```

### 2. Create a virtual environment

Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file includes common packages such as:

```text
torch
torchvision
ultralytics
opencv-python
numpy
pandas
matplotlib
seaborn
pyyaml
tqdm
scikit-learn
Pillow
albumentations
```

GPU acceleration is strongly recommended for model training and full evaluation.

---

## Quickstart

From inside the `computer_vision/` folder:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download or place the dataset
bash download_dataset.sh
# or on Windows:
download_dataset.bat

# 3. Check image-label matching
python scripts/check_dataset.py

# 4. Run YOLOv8n-seg inference on one image
yolo task=segment mode=predict \
  model=weights/yolov8n-seg_best.pt \
  source=path/to/test_image.jpg \
  imgsz=640 \
  conf=0.25
```

Python API example:

```python
from ultralytics import YOLO

model = YOLO("weights/yolov8n-seg_best.pt")

results = model.predict(
    source="path/to/test_image.jpg",
    imgsz=640,
    conf=0.25,
    save=True
)
```

---

## Notebook Workflows

### `codes/E_waste_8_model_benchmark.ipynb`

This is the main training and benchmarking notebook. It includes:

- dataset checking,
- YOLOv5n-seg training,
- YOLOv7-seg / YOLOv7-tiny-seg training,
- YOLOv8n-seg training,
- YOLO11n-seg training,
- optional YOLO26n-seg training,
- YOLO-to-COCO conversion for Detectron2,
- Mask R-CNN R50-FPN training,
- PointRend Mask R-CNN training,
- BatteryMask-RefineNet refinement experiment,
- metric collection helpers,
- qualitative visualisation helpers.

The notebook assumes a Google Drive project root similar to:

```text
/content/drive/MyDrive/E-waste Battery Extraction CV
```

Common editable variables:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/E-waste Battery Extraction CV")
DATASET_ROOT = PROJECT_ROOT / "final_dataset"
YOLO_DATA_YAML = DATASET_ROOT / "dataset.yaml"
```

---

### `codes/E_waste_8_model_evaluation.ipynb`

This notebook performs the final eight-model evaluation.

It produces:

1. a 9-metric comparison table,
2. a 3 x 3 qualitative comparison figure containing:
   - input image,
   - YOLOv5n-seg prediction,
   - YOLOv7-seg prediction,
   - YOLOv8n-seg prediction,
   - YOLO11n-seg prediction,
   - YOLO26n-seg prediction,
   - Mask R-CNN prediction,
   - PointRend prediction,
   - RTMDet-Ins prediction.

The reported metrics are:

```text
Mask mAP@50
Mask mAP@50:95
Precision
Recall
Mean Mask IoU
Latency (ms/image)
FPS
Parameters (M)
Centroid Error (px)
```

The final CSV is saved to:

```text
04_metrics_and_visualisations/eight_model_eval_selected_test/eight_models_9_metrics_selected_test.csv
```

The qualitative outputs are saved under:

```text
04_metrics_and_visualisations/qualitative_examples_8_models/
```

Visual convention:

```text
Green mask      = predicted mask
Red contour     = predicted mask boundary
Orange contour  = ground-truth mask boundary
Red cross       = predicted centroid
Orange cross    = ground-truth centroid
```

---

### `codes/RTMDet_Ins_Tiny_EWaste_Colab.ipynb`

This is a separate notebook for RTMDet-Ins tiny.

RTMDet-Ins tiny is separated because MMDetection has stricter dependency requirements than YOLO and Detectron2. The notebook creates a clean Python 3.10 micromamba environment named:

```text
rtmdet310
```

The notebook includes:

- YOLO segmentation label to COCO JSON conversion,
- micromamba environment creation,
- PyTorch/MMCV/MMEngine/MMDetection installation,
- RTMDet-Ins tiny config generation,
- RTMDet-Ins tiny training,
- selected test-set evaluation,
- COCO metric parsing,
- parameter counting,
- optional qualitative prediction.

Important RTMDet output folder:

```text
03_rtmdet_models/model_07_rtmdet_ins_tiny
```

Important RTMDet output files:

```text
rtmdet_ins_tiny_battery.py
rtmdet_selected_test_metrics.json
parameter_count.json
train_rtmdet_ins_tiny_live.log
```

---

## Using the Stored Weights

The `weights/` folder contains final saved checkpoints. The notebooks may expect the weights inside the original Google Drive training-output folders. If running directly from this GitHub branch, either update the model paths in the notebooks or copy the weights into the expected structure.

Suggested mapping:

```text
weights/yolov5n-seg_best.pt
→ 01_yolo_models/model_01_yolov5n_seg/weights/best.pt

weights/yolov7-seg_best.pt
→ 01_yolo_models/model_02_yolov7_seg/weights/best.pt

weights/yolov8n-seg_best.pt
→ 01_yolo_models/model_03_yolov8n_seg/weights/best.pt

weights/yolo11n-seg_best.pt
→ 01_yolo_models/model_04_yolo11n_seg/weights/best.pt

weights/yolo26n-seg_best.pt
→ 01_yolo_models/optional_yolo26n_seg/weights/best.pt

weights/mask-rcnn_50-FPN_model_final.pth
→ 02_detectron2_models/model_05_mask_rcnn_R50_FPN/model_final.pth

weights/pointrend_model_final.pth
→ 02_detectron2_models/model_07_pointrend_mask_rcnn_R50_FPN/model_final.pth

weights/rtmdet-ins_best.pth
→ 03_rtmdet_models/model_07_rtmdet_ins_tiny/best.pth
```

Colab helper:

```python
from pathlib import Path
import shutil

REPO_ROOT = Path("/content/Redback-Operations-Project-7-E-waste-Battery-Extraction/computer_vision")
PROJECT_ROOT = Path("/content/drive/MyDrive/E-waste Battery Extraction CV")

copy_map = {
    "weights/yolov5n-seg_best.pt": PROJECT_ROOT / "01_yolo_models/model_01_yolov5n_seg/weights/best.pt",
    "weights/yolov7-seg_best.pt": PROJECT_ROOT / "01_yolo_models/model_02_yolov7_seg/weights/best.pt",
    "weights/yolov8n-seg_best.pt": PROJECT_ROOT / "01_yolo_models/model_03_yolov8n_seg/weights/best.pt",
    "weights/yolo11n-seg_best.pt": PROJECT_ROOT / "01_yolo_models/model_04_yolo11n_seg/weights/best.pt",
    "weights/yolo26n-seg_best.pt": PROJECT_ROOT / "01_yolo_models/optional_yolo26n_seg/weights/best.pt",
    "weights/mask-rcnn_50-FPN_model_final.pth": PROJECT_ROOT / "02_detectron2_models/model_05_mask_rcnn_R50_FPN/model_final.pth",
    "weights/pointrend_model_final.pth": PROJECT_ROOT / "02_detectron2_models/model_07_pointrend_mask_rcnn_R50_FPN/model_final.pth",
    "weights/rtmdet-ins_best.pth": PROJECT_ROOT / "03_rtmdet_models/model_07_rtmdet_ins_tiny/best.pth",
}

for src_rel, dst in copy_map.items():
    src = REPO_ROOT / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")
    else:
        print(f"Missing source file: {src}")
```

---

## ONNX Exports

The selected YOLO ONNX exports are stored in:

```text
onnx/
```

Included exports:

```text
onnx/yolov8n-seg_best.onnx
onnx/yolo11n-seg_best.onnx
onnx/yolo26n-seg_best.onnx
```

Example export command:

```bash
yolo export model=weights/yolov8n-seg_best.pt format=onnx imgsz=640
```

Potential deployment or integration targets:

- ONNX Runtime,
- TensorRT,
- MATLAB deep learning workflow,
- Isaac Sim / robotics simulation pipeline,
- edge GPU inference.

---

## Scripts

### `scripts/augment_dataset.py`

Applies dataset augmentation before training.

Expected behaviour:

- creates augmented copies of original images,
- updates segmentation polygon labels,
- skips files already containing `_aug`,
- supports transformations such as flip, brightness/contrast, hue/saturation, rotation, scaling/translation, and blur.

Run:

```bash
python scripts/augment_dataset.py
```

---

### `scripts/blenderGenerationScript.py`

Runs inside Blender to generate synthetic phone images.

The script is intended to:

- randomise camera position,
- randomise lighting,
- randomise background materials,
- iterate through phone models,
- render synthetic phone images.

Run inside Blender:

```text
Blender → Scripting tab → Open script → Run Script
```

Expected Blender scene objects may include:

```text
Camera
Light
Cube / background object
PhoneInsides collection
Border collection
material_folder/
output_folder/
```

---

### `scripts/check_dataset.py`

Checks whether every image has a matching label file and whether every label has a matching image.

Run:

```bash
python scripts/check_dataset.py
```

Useful before training, because missing labels or orphaned labels can cause incorrect training/evaluation behaviour.

---

### `scripts/evaluate_all_metrics.py`

Evaluates multiple models and creates a summary CSV.

Older documentation describes this script as evaluating:

```text
YOLOv8n-seg
YOLO26n-seg
YOLOv11x-seg
SSD
RetinaNet
```

Typical metrics:

```text
Mask IoU / Box IoU
Average Centroid Error
Median Centroid Error
Inference Latency
FPS
```

Run:

```bash
python scripts/evaluate_all_metrics.py
```

Before running, check and update model paths inside the script.

---

### `scripts/evaluate_ssd_retinanet.py`

Generates curve plots and a confusion matrix for SSD or RetinaNet.

Typical outputs:

```text
precision curve
recall curve
F1 curve
precision-recall curve
confusion matrix
```

Run:

```bash
python scripts/evaluate_ssd_retinanet.py
```

Before running, set the model type inside the script if required:

```python
MODEL_TYPE = "ssd"
# or
MODEL_TYPE = "retinanet"
```

---

### `scripts/imgscraper.py`

Experimental YouTube scraping and auto-segmentation script.

Pipeline:

1. download phone teardown/repair videos,
2. extract scene-change frames,
3. filter blurry frames,
4. apply SAM-assisted segmentation,
5. save candidate masked images.

Possible extra requirements:

```bash
pip install pytubefix opencv-python ffmpeg-python segment-anything torch pillow
```

SAM checkpoint note:

```text
Place the SAM checkpoint in the expected scripts path if the script requires it.
```

Run:

```bash
python scripts/imgscraper.py
```

Important limitation:

```text
The segmentation quality is not guaranteed. SAM point prompts may capture hands, tools, or background.
Manual checking is recommended before using scraped masks for model training.
```

---

### `scripts/visual_compare_all_models.py`

Creates side-by-side qualitative comparison images across models.

The output is useful for reports and presentations because it shows model prediction quality visually.

Run:

```bash
python scripts/visual_compare_all_models.py
```

Before running, update model folders and test image paths inside the script if needed.

---

### `scripts/YoutubeScraper.md`

Documentation for the YouTube scraping workflow. This should be read before using `imgscraper.py`, especially if future students want to expand the dataset from teardown videos.

---

## Training

### YOLOv8n-seg

```bash
yolo task=segment mode=train \
  model=yolov8n-seg.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  device=0
```

### YOLO11n / YOLOv11-style segmentation

```bash
yolo task=segment mode=train \
  model=yolo11n-seg.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  device=0
```

### YOLO26n-seg

```bash
yolo task=segment mode=train \
  model=yolo26n-seg.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  device=0
```

Training outputs are usually saved to:

```text
runs/segment/train/
```

Important output files:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
```

---

## Evaluation

### YOLO validation

```bash
yolo task=segment mode=val \
  model=weights/yolov8n-seg_best.pt \
  data=dataset/data.yaml \
  imgsz=640 \
  device=0
```

### Final eight-model evaluation

Use:

```text
codes/E_waste_8_model_evaluation.ipynb
```

Before running, confirm that:

```text
dataset exists,
model weights exist,
Detectron2 config files exist,
RTMDet config file exists,
ONNX files exist if export testing is required.
```

### Evaluation subset

The evaluation notebook creates a selected test subset:

```text
eval_test_subset_original_plus_selected_aug/
```

The intended rule is:

```text
original image + aug_1/aug_2/aug_3 if enough augmentations exist;
otherwise include all available augmentations.
```

It also creates COCO-style annotations for Detectron2 and RTMDet evaluation:

```text
eval_coco_annotations/test_selected.json
final_dataset/mmdet_annotations/test_selected_original_plus_aug.json
```

This helps keep evaluation more consistent across YOLO, Detectron2, and MMDetection models.

---

## Synthetic Data Generation

Synthetic data generation is supported through Blender.

Relevant files:

```text
scripts/blenderGenerationScript.py
docs/makingNewAutomationBlenderFile.md
makingNewAutomationBlenderFile.txt
```

The Blender workflow is useful because real phone disassembly images are limited and difficult to collect. Synthetic data can increase visual diversity by changing:

- phone model,
- viewpoint,
- lighting,
- material/background,
- camera distance,
- scene layout.

Recommended use:

1. Read `docs/makingNewAutomationBlenderFile.md`.
2. Prepare the Blender scene with the expected object/collection names.
3. Run `scripts/blenderGenerationScript.py` inside Blender.
4. Review generated images manually.
5. Add high-quality synthetic samples to the training set.

---

## YouTube Scraping and Auto-Segmentation

The scraping workflow is experimental and should not be treated as fully automatic dataset creation.

Relevant files:

```text
scripts/imgscraper.py
scripts/YoutubeScraper.md
```

Purpose:

- collect phone teardown/repair frames,
- extract candidate battery images,
- use SAM-assisted segmentation,
- produce possible extra dataset samples.

Main limitations:

- duplicate video frames may be collected,
- blurry or wide-angle frames may pass filtering,
- SAM may segment hands, tools, or background instead of batteries,
- masks need manual review before use in training.

Recommended future improvements:

- better SAM prompt strategy,
- frame deduplication,
- automatic filtering for battery-visible images,
- manual review interface,
- metadata logging for scraped videos.

---

## Results Summary

The older five-model README reports the following comparison:

| Model | mAP@50 | Inference (ms) | FPS | Avg Centroid Error (px) | Notes |
|---|---:|---:|---:|---:|---|
| YOLOv8n-seg | 0.99 | 51.75 | 19.32 | 25.37 | Best deployment balance |
| YOLOv11x-seg | 0.99 | 72.95 | 13.71 | 41.22 | Highest accuracy |
| YOLO26n-seg | 0.98 | 60.21 | 16.61 | 34.63 | Experimental comparison |
| RetinaNet | 0.90–0.97 | 81.98 | 12.20 | 133.77 | Detection baseline |
| SSD | 0.88–0.96 | 24.83 | 40.28 | 254.48 | Fast but less reliable |

Interpretation:

- **YOLOv8n-seg** is the best deployment-oriented model because it provides strong accuracy with better speed and centroid localisation.
- **YOLOv11x-seg** is useful for accuracy-focused evaluation but is slower.
- **SSD** is fast, but its centroid error is too high for precise robotic targeting.
- **RetinaNet** is useful as a detector baseline but does not provide the same segmentation value as YOLO segmentation models.
- The eight-model workflow extends the comparison by adding YOLOv5n-seg, YOLOv7-seg, Mask R-CNN, PointRend, and RTMDet-Ins tiny.

For the most complete final comparison, use:

```text
codes/E_waste_8_model_evaluation.ipynb
```

---

## Compatibility Notes

### General recommendation

Use Google Colab with GPU for the notebooks.

Recommended:

```text
Runtime: Google Colab
Hardware accelerator: GPU
Preferred GPU: A100 if available
Image size: 640
```

### YOLO-family models

Different YOLO versions may require different repositories or setup cells.

General packages:

```bash
pip install ultralytics opencv-python pycocotools pandas numpy tqdm pyyaml
```

YOLOv7 may also require:

```bash
pip install filterpy
```

The safest approach is to run the notebook setup cells in order instead of mixing YOLO repositories manually.

---

### Detectron2 models

Mask R-CNN and PointRend require Detectron2.

Typical Colab installation:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

PointRend may require the Detectron2 project folder:

```text
/content/detectron2_repo/projects/PointRend
```

Detectron2 models require both:

```text
checkpoint .pth file
matching config file
```

Expected config outputs from the benchmark notebook:

```text
02_detectron2_models/model_05_mask_rcnn_R50_FPN/config_used.yaml
02_detectron2_models/model_07_pointrend_mask_rcnn_R50_FPN/config_used.yaml
```

If config files are missing, rerun the Detectron2 config-generation cells or add the generated configs to the repository.

---

### RTMDet-Ins tiny / MMDetection

RTMDet-Ins tiny uses a separate micromamba environment because MMDetection can conflict with the default Colab environment.

Expected environment:

```text
Environment name: rtmdet310
Python: 3.10
PyTorch: 2.1.2
Torchvision: 0.16.2
CUDA wheel index: cu121
MMCV: 2.1.0
MMEngine: >= 0.10.3
MMDetection: v3.3.0
NumPy: < 2
```

Typical setup pattern:

```bash
micromamba create -y -n rtmdet310 python=3.10 pip

micromamba run -n rtmdet310 python -m pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

micromamba run -n rtmdet310 python -m pip install \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
```

The RTMDet checkpoint requires the matching config file:

```text
03_rtmdet_models/model_07_rtmdet_ins_tiny/rtmdet_ins_tiny_battery.py
```

If only the checkpoint is available, rerun the config-generation cell in:

```text
codes/RTMDet_Ins_Tiny_EWaste_Colab.ipynb
```

---

## Known Repository Notes

The folder is functional, but it contains several naming and organisation issues that future maintainers should be aware of.

| Item | Note |
|---|---|
| `Evaluation of YOLO models.ipynbp` | The extension appears to be a typo. Rename to `.ipynb` if it is meant to be opened as a notebook. |
| `Model Research Reprt.pdf` | Filename appears to contain a spelling typo. |
| `E-waste battery dataset creation report.pdf` and `Ewaste Battery Dataset Creation Report.pdf` | These may be duplicate or different versions. Keep the final version clearly labelled. |
| `README.md` and `UpdatedReadme.md` | Two README-style files exist. This main README should replace/merge them to avoid confusion. |
| `evaluation` | Appears to be a very small placeholder-style file, not a full evaluation folder. |
| `makingNewAutomationBlenderFile.md` and `makingNewAutomationBlenderFile.txt` | Duplicate guide content in different formats. Keep one canonical version if possible. |
| Large model weights | Convenient for marking, but for long-term maintenance, consider Releases, Git LFS, Drive, Kaggle, or a model registry. |
| Dataset availability | Dataset access may depend on Kaggle/private project permissions. Make access instructions explicit before final submission. |

---

## Recommended Future Work

1. **Clean file naming**
   - Rename typo files.
   - Remove or archive duplicate documentation.
   - Rename `.ipynbp` to `.ipynb` if appropriate.

2. **Add exact reproduction paths**
   - Document every notebook path variable.
   - Add expected Google Drive folder structure.
   - Add model path checks before evaluation.

3. **Add config files for non-YOLO models**
   - Include Detectron2 `config_used.yaml` files.
   - Include RTMDet `rtmdet_ins_tiny_battery.py` if it is required for loading the `.pth` checkpoint.

4. **Improve ONNX coverage**
   - Keep current YOLOv8n, YOLO11n, and YOLO26n ONNX exports.
   - Investigate whether YOLOv5/YOLOv7 export is worth including.
   - Keep Detectron2 and RTMDet exports separate unless they are tested properly.

5. **Separate real-only and augmented evaluation**
   - Report performance on real/original test images separately from augmented test images.
   - This avoids overestimating real-world generalisation.

6. **Add model cards**
   - Add one short model card for the recommended deployment model.
   - Include expected input, output format, limitations, and safety considerations.

7. **Prepare robotic integration output format**

   Example output format:

   ```json
   {
     "class": "battery",
     "confidence": 0.95,
     "bbox": [x1, y1, x2, y2],
     "centroid": [cx, cy],
     "mask_polygon": [[x1, y1], [x2, y2], [x3, y3]]
   }
   ```

8. **Improve synthetic data workflow**
   - Add examples of generated images.
   - Document how labels are produced.
   - Validate synthetic-to-real performance impact.

9. **Improve scraper reliability**
   - Add deduplication.
   - Improve SAM prompting.
   - Add human review before accepting masks into training.

---

## License and Acknowledgement

This computer vision module is part of the Redback Operations Project 7 e-waste battery extraction project.

The repository is released under the license provided in the parent repository. Check the root-level `LICENSE` file for the full license text.

Acknowledgement is given to the Computer Vision Sub-Team and project contributors who worked on dataset preparation, model training, benchmarking, documentation, and deployment preparation for safer robotic e-waste battery extraction.
