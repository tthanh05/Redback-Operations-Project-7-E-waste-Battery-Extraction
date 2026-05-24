# Redback Operations Project 7 – E-waste Battery Extraction

> Computer vision pipeline for detecting and segmenting lithium-ion batteries in disassembled mobile phones, supporting safer robotic e-waste battery extraction.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLO](https://img.shields.io/badge/YOLO-Detection%20%26%20Segmentation-green)
![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Branch Purpose](#branch-purpose)
- [Why This Project Matters](#why-this-project-matters)
- [Main Features](#main-features)
- [Repository Structure](#repository-structure)
- [Where to Start](#where-to-start)
- [Models Included](#models-included)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running Inference](#running-inference)
- [Training Models](#training-models)
- [Evaluation](#evaluation)
- [ONNX Export and Deployment](#onnx-export-and-deployment)
- [Synthetic Data Generation](#synthetic-data-generation)
- [Documentation](#documentation)
- [Known Notes](#known-notes)
- [Recommended Future Work](#recommended-future-work)
- [License](#license)

---

## Project Overview

This repository contains the **Computer Vision branch** for the Redback Operations Project 7 e-waste battery extraction system. The project focuses on identifying lithium-ion batteries inside disassembled mobile phones using object detection and instance segmentation models.

The computer vision module is designed to support a future robotic extraction workflow. In that workflow, the perception system detects the battery, estimates its location and mask boundary, and provides useful visual information for downstream robotic control, path planning, or simulation integration.

The main computer vision work is located inside:

```text
computer_vision/
```

The root `README.md` is intended to be the main entry point for reviewers, team members, and future developers. More detailed implementation notes are provided inside:

```text
computer_vision/README.md
```

---

## Branch Purpose

This branch is dedicated to the **battery detection and segmentation component** of the e-waste robotic extraction project.

It includes:

- model training notebooks,
- evaluation notebooks,
- dataset preparation scripts,
- trained model weights,
- ONNX exports,
- visual comparison outputs,
- Blender synthetic data generation documentation,
- YouTube scraping and auto-segmentation experiments,
- written project documentation and reports.

The goal is not only to train a single model, but also to compare several candidate models and identify which one is most suitable for practical robotic use.

---

## Why This Project Matters

Lithium-ion batteries are hazardous during e-waste processing. If a battery is punctured, bent, crushed, or handled incorrectly, it may cause fire, toxic leakage, or other safety risks. Manual extraction can also expose workers to dangerous materials and repetitive handling.

Computer vision can reduce this risk by helping a robotic system locate the battery before extraction. A good model should not only detect whether a battery is present, but also provide useful localisation information such as:

- a bounding box,
- a segmentation mask,
- the approximate battery centre,
- mask/box quality,
- inference speed,
- deployment suitability.

For this reason, this project evaluates models using both accuracy-oriented and robotics-oriented metrics.

---

## Main Features

### 1. Battery Detection and Segmentation

The project focuses on detecting a single class:

```yaml
0: battery
```

The most important model family in this branch is the YOLO segmentation family, because it provides real-time object detection and mask prediction.

### 2. Multi-model Benchmarking

The branch includes experiments and outputs for multiple models, including YOLO-based segmentation models and additional detection/segmentation baselines.

### 3. Quantitative Evaluation

The evaluation pipeline considers both traditional computer vision metrics and robotics-relevant metrics, such as:

- precision,
- recall,
- F1-score,
- mAP,
- mask IoU,
- box IoU,
- centroid error,
- inference latency,
- FPS.

### 4. Qualitative Evaluation

The repository includes visual outputs such as:

- confusion matrices,
- precision-recall curves,
- F1 curves,
- training result plots,
- validation predictions,
- side-by-side model comparison images.

### 5. Synthetic Data Support

The branch includes Blender automation documentation and scripts for generating synthetic phone images. This is useful because collecting real disassembled phone images is difficult, time-consuming, and limited by device availability.

### 6. Deployment Preparation

The repository contains ONNX exports for selected YOLO segmentation models, which is useful for future integration with deployment platforms or robotic pipelines.

---

## Repository Structure

The current branch has the following high-level structure:

```text
Redback-Operations-Project-7-E-waste-Battery-Extraction/
│
├── computer_vision/
│   ├── Documents/
│   ├── assets/
│   ├── codes/
│   ├── docs/
│   ├── model_benchmark_yehezkiel/
│   ├── onnx/
│   ├── scripts/
│   ├── weights/
│   ├── README.md
│   ├── UpdatedReadme.md
│   ├── requirements.txt
│   ├── download_dataset.bat
│   ├── download_dataset.sh
│   ├── makingNewAutomationBlenderFile.txt
│   └── evaluation
│
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md
```

---

## Folder Guide

### `computer_vision/`

Main working directory for the computer vision component.

This folder contains the implementation, documentation, trained models, evaluation results, scripts, and supporting files for the battery detection and segmentation pipeline.

### `computer_vision/Documents/`

Contains written PDF documentation and reports, including:

- cost evaluation,
- dataset creation report,
- literature review and dataset collection report,
- model evaluation report.

These files are useful for project marking, final reporting, and understanding the background work behind the technical implementation.

### `computer_vision/assets/`

Contains model evaluation assets and visual outputs.

Typical files include:

- confusion matrices,
- precision-confidence curves,
- recall-confidence curves,
- F1 curves,
- precision-recall curves,
- training result plots,
- validation prediction examples.

Model-specific folders include outputs for models such as:

```text
YOLOv8n-seg/
YOLOv11x-seg/
YOLO26n-seg/
RetinaNet/
SSD/
```

### `computer_vision/codes/`

Contains the main Jupyter/Colab notebooks used for training, benchmarking, and evaluation.

Current notebook-style files include:

```text
E_waste_8_model_benchmark.ipynb
E_waste_8_model_evaluation.ipynb
Evaluation of YOLO models.ipynbp
RTMDet_Ins_Tiny_EWaste_Colab.ipynb
```

Recommended usage:

- use the benchmark notebook to understand training and model comparison,
- use the evaluation notebook to reproduce or inspect evaluation results,
- use the RTMDet notebook for the MMDetection/MMYOLO-style experiment,
- check file extensions before running notebooks locally.

### `computer_vision/docs/`

Contains additional project documentation.

Important files include:

```text
Model Research Reprt.pdf
makingNewAutomationBlenderFile.md
```

The Blender guide explains how to create or modify Blender scenes for synthetic dataset generation.

### `computer_vision/model_benchmark_yehezkiel/`

Contains a separate benchmark workspace focused on e-waste battery instance segmentation.

This subfolder includes:

```text
code/
README.md
```

It documents benchmarking of YOLO-family instance segmentation models and a proposed hybrid refinement method called **BatteryMask-RefineNet**.

The subproject README describes BatteryMask-RefineNet as a two-stage pipeline:

1. YOLOv8n-seg generates an initial coarse battery mask.
2. A lightweight U-Net-style refinement model improves the mask boundary.

This folder is useful if the reader wants to understand the alternative benchmarking direction and hybrid refinement idea.

### `computer_vision/onnx/`

Contains exported ONNX model files for selected YOLO segmentation models:

```text
yolo11n-seg_best.onnx
yolo26n-seg_best.onnx
yolov8n-seg_best.onnx
```

These files are useful for deployment experiments, model portability, and possible integration with robotics or edge inference systems.

### `computer_vision/scripts/`

Contains reusable Python scripts for dataset preparation, validation, evaluation, scraping, synthetic generation, and visual comparison.

Current scripts include:

```text
YoutubeScraper.md
augment_dataset.py
blenderGenerationScript.py
check_dataset.py
evaluate_all_metrics.py
evaluate_ssd_retinanet.py
imgscraper.py
visual_compare_all_models.py
```

A short summary of each script is provided below.

| Script | Purpose |
|---|---|
| `augment_dataset.py` | Applies image augmentation and updates segmentation labels. |
| `blenderGenerationScript.py` | Generates synthetic phone images inside Blender. |
| `check_dataset.py` | Checks whether image files and label files are correctly matched. |
| `evaluate_all_metrics.py` | Evaluates multiple models and produces combined metrics. |
| `evaluate_ssd_retinanet.py` | Evaluates SSD and RetinaNet outputs and generates plots. |
| `imgscraper.py` | Scrapes teardown/repair videos and uses segmentation assistance to create battery masks. |
| `visual_compare_all_models.py` | Produces side-by-side visual comparisons of model predictions. |
| `YoutubeScraper.md` | Documents the YouTube scraping workflow and limitations. |

### `computer_vision/weights/`

Contains trained model weights.

Current files include:

```text
mask-rcnn_50-FPN_model_final.pth
pointrend_model_final.pth
rtmdet-ins_best.pth
yolo11n-seg_best.pt
yolo26n-seg_best.pt
yolov5n-seg_best.pt
yolov7-seg_best.pt
yolov8n-seg_best.pt
```

These weights allow future users to run inference or continue evaluation without retraining every model from scratch.

---

## Where to Start

For a new user or marker, the recommended reading order is:

1. Read this root `README.md` for the complete branch overview.
2. Open `computer_vision/README.md` for the detailed technical workflow.
3. Open `computer_vision/codes/E_waste_8_model_benchmark.ipynb` to inspect the training and benchmarking workflow.
4. Open `computer_vision/codes/E_waste_8_model_evaluation.ipynb` to inspect the model evaluation workflow.
5. Check `computer_vision/assets/` for result plots and qualitative outputs.
6. Check `computer_vision/weights/` and `computer_vision/onnx/` for trained and exported models.
7. Read the PDFs in `computer_vision/Documents/` for report-level explanation.

---

## Models Included

The branch contains work related to the following model families.

| Model | Type | Weight/Output Status | Main Use |
|---|---|---|---|
| YOLOv5n-seg | Instance segmentation | Weight included | Lightweight YOLO baseline |
| YOLOv7-seg | Instance segmentation | Weight included | YOLO segmentation comparison |
| YOLOv8n-seg | Instance segmentation | Weight and ONNX included | Recommended deployment-oriented model |
| YOLO11n-seg | Instance segmentation | Weight and ONNX included | Newer YOLO comparison |
| YOLO26n-seg | Instance segmentation | Weight and ONNX included | Experimental YOLO comparison |
| Mask R-CNN R50-FPN | Instance segmentation | Weight included | Two-stage segmentation baseline |
| PointRend | Instance segmentation | Weight included | Boundary-aware segmentation comparison |
| RTMDet-Ins Tiny | Instance segmentation | Weight included | Lightweight instance segmentation experiment |
| RetinaNet | Object detection | Evaluation assets included | Detection baseline |
| SSD | Object detection | Evaluation assets included | Fast detection baseline |

---

## Dataset

The project is designed around a single-class battery dataset.

Expected class format:

```yaml
names:
  0: battery
```

Expected YOLO-style dataset structure:

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

A typical YOLO segmentation label line follows this structure:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

where polygon coordinates are normalised between `0` and `1`.

### Dataset Access

The dataset is referenced through Kaggle download scripts:

```text
computer_vision/download_dataset.sh
computer_vision/download_dataset.bat
```

Before running the scripts, make sure Kaggle credentials are available on the machine.

For Linux/Mac:

```bash
mkdir -p ~/.kaggle
# place kaggle.json inside ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

For Windows:

```text
C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json
```

Then run the relevant download script from inside `computer_vision/`.

---

## Installation

### 1. Clone the repository

```bash
git clone -b computer-vision https://github.com/tthanh05/Redback-Operations-Project-7-E-waste-Battery-Extraction.git
cd Redback-Operations-Project-7-E-waste-Battery-Extraction/computer_vision
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

The main dependencies include:

- PyTorch,
- TorchVision,
- Ultralytics,
- OpenCV,
- NumPy,
- Pandas,
- Matplotlib,
- Seaborn,
- PyYAML,
- tqdm,
- scikit-learn,
- Pillow,
- Albumentations.

GPU acceleration is strongly recommended for training and large-scale evaluation.

---

## Quickstart

From the `computer_vision/` directory:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
bash download_dataset.sh
# or on Windows:
download_dataset.bat

# 3. Check dataset image-label matching
python scripts/check_dataset.py

# 4. Run YOLO inference using a trained weight
yolo task=segment mode=predict model=weights/yolov8n-seg_best.pt source=path/to/image.jpg
```

---

## Running Inference

Example using YOLOv8n-seg:

```bash
cd computer_vision

yolo task=segment mode=predict \
  model=weights/yolov8n-seg_best.pt \
  source=path/to/test_image.jpg \
  imgsz=640 \
  conf=0.25
```

Example using the Python API:

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

The prediction output normally includes:

- bounding box,
- segmentation mask,
- confidence score,
- saved visualisation image.

---

## Training Models

The exact training workflow may differ between notebooks and scripts, but a typical YOLO segmentation training command is:

```bash
yolo task=segment mode=train \
  model=yolov8n-seg.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=4 \
  device=0
```

For larger GPUs, batch size can be increased depending on memory availability.

Typical outputs are saved under:

```text
runs/segment/train/
```

Important outputs include:

```text
weights/best.pt
weights/last.pt
results.csv
results.png
confusion_matrix.png
```

---

## Evaluation

The branch contains both notebook-based and script-based evaluation.

### Main Evaluation Notebook

```text
computer_vision/codes/E_waste_8_model_evaluation.ipynb
```

This notebook is intended for comparing multiple trained models and generating quantitative/qualitative outputs.

### Combined Metric Evaluation Script

```bash
python scripts/evaluate_all_metrics.py
```

This script is intended to compare multiple models and produce summary metrics such as:

- mask IoU,
- box IoU,
- centroid error,
- latency,
- FPS.

### SSD and RetinaNet Evaluation

```bash
python scripts/evaluate_ssd_retinanet.py
```

This script is used for the detection-only baseline models.

### Visual Model Comparison

```bash
python scripts/visual_compare_all_models.py
```

This script generates side-by-side qualitative comparisons across models.

---

## Result Summary

The inner `computer_vision/README.md` reports the following model-level comparison:

| Model | mAP@50 | Inference | FPS | Avg Centroid Error | Notes |
|---|---:|---:|---:|---:|---|
| YOLOv8n-seg | 0.99 | 51.75 ms | 19.32 | 25.37 px | Best deployment balance |
| YOLOv11x-seg | 0.99 | 72.95 ms | 13.71 | 41.22 px | Highest accuracy |
| YOLO26n-seg | 0.98 | 60.21 ms | 16.61 | 34.63 px | Experimental |
| RetinaNet | 0.90–0.97 | 81.98 ms | 12.20 | 133.77 px | Baseline detector |
| SSD | 0.88–0.96 | 24.83 ms | 40.28 | 254.48 px | Fast but less reliable |

Based on these reported results, **YOLOv8n-seg** is the most practical deployment candidate because it provides the best balance between segmentation quality, speed, and centroid accuracy.

For accuracy-focused offline analysis, **YOLOv11x-seg** is also useful, but it is heavier and slower.

---

## ONNX Export and Deployment

The repository includes ONNX exports for selected YOLO segmentation models:

```text
computer_vision/onnx/yolov8n-seg_best.onnx
computer_vision/onnx/yolo11n-seg_best.onnx
computer_vision/onnx/yolo26n-seg_best.onnx
```

ONNX export is useful because it allows trained models to be tested outside the original PyTorch/Ultralytics training environment.

Example YOLO ONNX export command:

```bash
yolo export model=weights/yolov8n-seg_best.pt format=onnx imgsz=640
```

Potential future deployment targets include:

- ONNX Runtime,
- TensorRT,
- robotic simulation pipelines,
- edge GPU devices,
- perception modules for ROS/Isaac Sim/MATLAB integration.

---

## Synthetic Data Generation

The project includes Blender-based synthetic image generation support.

Relevant files:

```text
computer_vision/scripts/blenderGenerationScript.py
computer_vision/docs/makingNewAutomationBlenderFile.md
computer_vision/makingNewAutomationBlenderFile.txt
```

The synthetic data pipeline is intended to create additional training images by rendering phone internals under different conditions.

The Blender script supports variation in:

- phone model,
- camera position,
- light position,
- background material,
- render angle,
- visual scene setup.

This is useful because real disassembled phone datasets are difficult to collect at large scale.

---

## YouTube Scraping and Auto-Segmentation

The project also includes an experimental scraping pipeline.

Relevant files:

```text
computer_vision/scripts/imgscraper.py
computer_vision/scripts/YoutubeScraper.md
```

The purpose of this pipeline is to collect frames from phone teardown or repair videos and assist with battery mask generation.

The general workflow is:

1. download teardown/repair videos,
2. extract useful frames,
3. filter low-quality frames,
4. apply automatic or semi-automatic segmentation,
5. save candidate battery images/masks.

This part should be treated as experimental. It can help expand the dataset, but the generated masks should still be checked manually before being used as reliable training labels.

---

## Documentation

Important documentation files include:

```text
computer_vision/Documents/Cost evaluation on e-waste battery extraction.pdf
computer_vision/Documents/E-waste battery dataset creation report.pdf
computer_vision/Documents/Ewaste Battery Dataset Creation Report.pdf
computer_vision/Documents/Literature Review and Dataset Collection for Robotic E.pdf
computer_vision/Documents/Model Evaluation for YOLO v8n, v11n and v26n.pdf
computer_vision/docs/Model Research Reprt.pdf
computer_vision/docs/makingNewAutomationBlenderFile.md
```

These documents explain the broader project background, dataset creation, literature review, model evaluation, and Blender automation process.

---

## Known Notes

This branch contains both current and legacy project files. Some files may have naming inconsistencies because they were created during different stages of the project.

Examples:

- `Evaluation of YOLO models.ipynbp` appears to have a notebook-like name but a non-standard `.ipynbp` extension.
- `Model Research Reprt.pdf` appears to contain a spelling typo in the filename.
- Both `README.md` and `UpdatedReadme.md` exist inside `computer_vision/`.
- The `evaluation` item appears to be a very small placeholder-style file rather than a full evaluation folder.
- Large trained weights are stored directly in the repository. This is convenient for marking and reproduction, but a production repository would normally store large model files in Releases, cloud storage, or a model registry.

These notes do not prevent the branch from being useful, but future cleanup would improve maintainability.

---

## Recommended Future Work

Suggested future improvements include:

1. **Clean repository structure**

   Rename inconsistent files, remove duplicate documentation, and clearly separate active files from legacy files.

2. **Improve dataset access instructions**

   Confirm whether the Kaggle dataset is public or private. If private, explain how future team members can request access.

3. **Add exact reproduction commands**

   Add commands for reproducing each model result, including dataset path, weight path, image size, confidence threshold, and output folder.

4. **Improve deployment testing**

   Test ONNX models using ONNX Runtime or TensorRT and compare latency against PyTorch inference.

5. **Add robotics integration output format**

   Provide a standard output format for downstream control systems, such as:

   ```json
   {
     "class": "battery",
     "confidence": 0.95,
     "bbox": [x1, y1, x2, y2],
     "centroid": [cx, cy],
     "mask": "polygon_or_binary_mask"
   }
   ```

6. **Improve synthetic-to-real validation**

   Compare performance on real-only test images and synthetic images separately to avoid overestimating real-world performance.

7. **Improve scraping pipeline**

   Add deduplication, better segmentation prompts, quality filtering, and manual review tools for scraped images.

8. **Add a model card**

   Add a short model card for the recommended deployment model, including limitations, expected input, output format, and safety considerations.

---

## Suggested README Strategy

This root README should remain the **main branch entry point**. It should explain what the branch contains, where to start, what models are included, and what the main results mean.

The more detailed technical instructions should stay inside:

```text
computer_vision/README.md
```

That inner README can be more implementation-focused, including script-level explanations, training commands, dataset setup, evaluation commands, and detailed notes for future developers.

This avoids repeating the same long information in two places while still making the GitHub branch understandable at first glance.

---

## License

This repository is released under the Apache-2.0 License. See:

```text
LICENSE
```

for the full license text.

---

## Acknowledgement

This work was developed as part of the Redback Operations Project 7 e-waste battery extraction capstone project. The computer vision branch focuses on perception support for safer battery localisation and future robotic extraction.
