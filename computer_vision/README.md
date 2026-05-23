# E-waste Battery Instance Segmentation — Model Training, Weights, and Evaluation

This branch contains the training notebooks, saved model weights, and evaluation workflow for the e-waste battery instance segmentation task. The project focuses on detecting and segmenting visible batteries from 2D rear-phone images using single-class instance segmentation.

The target class is:

```text
battery
```

The branch mainly supports eight trained/evaluated segmentation models:

1. YOLOv5n-seg
2. YOLOv7-seg / YOLOv7-tiny-seg
3. YOLOv8n-seg
4. YOLO11n-seg
5. YOLO26n-seg
6. Mask R-CNN R50-FPN
7. PointRend Mask R-CNN
8. RTMDet-Ins tiny

---

## Repository structure

```text
.
├── codes/
│   ├── E_waste_8_model_benchmark.ipynb
│   ├── E_waste_8_model_evaluation.ipynb
│   ├── RTMDet_Ins_Tiny_EWaste_Colab.ipynb
│   └── README.md
│
├── weights/
│   ├── mask-rcnn_50-FPN_model_final.pth
│   ├── pointrend_model_final.pth
│   ├── rtmdet-ins_best.pth
│   ├── yolo11n-seg_best.pt
│   ├── yolo26n-seg_best.pt
│   ├── yolov5n-seg_best.pt
│   ├── yolov7-seg_best.pt
│   └── yolov8n-seg_best.pt
│
├── onnx/
│   ├── yolo11n-seg_best.onnx
│   ├── yolo26n-seg_best.onnx
│   └── yolov8n-seg_best.onnx
│
└── .gitattributes
```

The `codes/` folder contains the main Colab notebooks. The `weights/` folder stores the trained model weights used for evaluation and qualitative comparison. The `onnx/` folder stores selected YOLO segmentation exports prepared for MATLAB-side integration. The dataset itself is not included in this branch because it is a private project dataset.

---

## Notebook summary

### `codes/E_waste_8_model_benchmark.ipynb`

This notebook is the main training and benchmarking notebook. It includes training sections for the YOLO-family models, Detectron2-based models, and a deployment-oriented refinement experiment.

The notebook includes:

- dataset checking;
- YOLOv5n-seg training;
- YOLOv7-seg / YOLOv7-tiny-seg training;
- YOLOv8n-seg training;
- YOLO11n-seg training;
- optional YOLO26n-seg training;
- YOLO-to-COCO annotation conversion for Detectron2;
- Mask R-CNN R50-FPN training;
- PointRend Mask R-CNN training;
- BatteryMask-RefineNet refinement experiment;
- metric collection and qualitative visualisation helpers.

The notebook assumes a Google Drive project root similar to:

```text
/content/drive/MyDrive/E-waste Battery Extraction CV
```

The main editable path variables are:

```python
PROJECT_ROOT = Path('/content/drive/MyDrive/E-waste Battery Extraction CV')
DATASET_ROOT = PROJECT_ROOT / 'final_dataset'
YOLO_DATA_YAML = DATASET_ROOT / 'dataset.yaml'
```

---

### `codes/E_waste_8_model_evaluation.ipynb`

This notebook performs the final eight-model evaluation. It produces:

1. a 9-metric comparison table;
2. a 3 x 3 qualitative comparison figure containing the input image and eight model predictions.

The evaluated models are:

```text
YOLOv5n-seg
YOLOv7-seg / YOLOv7-tiny-seg
YOLOv8n-seg
YOLO11n-seg
YOLO26n-seg
Mask R-CNN R50-FPN
PointRend Mask R-CNN
RTMDet-Ins tiny
```

The metrics reported are:

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

The qualitative comparison figure is saved under:

```text
04_metrics_and_visualisations/qualitative_examples_8_models/
```

The qualitative visualisation uses the following convention:

```text
Green mask      = predicted mask
Red contour     = predicted mask boundary
Orange contour  = ground-truth mask boundary
Red cross       = predicted centroid
Orange cross    = ground-truth centroid
```

---

### `codes/RTMDet_Ins_Tiny_EWaste_Colab.ipynb`

This is a separate notebook for training and evaluating RTMDet-Ins tiny.

RTMDet-Ins tiny is kept in a separate notebook because MMDetection has stricter dependency requirements than the YOLO and Detectron2 workflows. The notebook creates a clean Python 3.10 micromamba environment named:

```text
rtmdet310
```

The notebook includes:

- YOLO segmentation label to COCO JSON conversion;
- micromamba environment creation;
- installation of PyTorch, MMCV, MMEngine, and MMDetection;
- RTMDet-Ins tiny config generation;
- RTMDet-Ins tiny training;
- selected test-set evaluation;
- COCO metric parsing;
- parameter counting;
- optional qualitative prediction.

The main RTMDet output folder is:

```text
03_rtmdet_models/model_07_rtmdet_ins_tiny
```

Important RTMDet output files include:

```text
rtmdet_ins_tiny_battery.py
rtmdet_selected_test_metrics.json
parameter_count.json
train_rtmdet_ins_tiny_live.log
```

---

## Dataset requirement

The notebooks expect the private dataset to use a YOLO segmentation layout:

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

The label format is YOLO segmentation polygon format:

```text
class_id x1 y1 x2 y2 x3 y3 ... xn yn
```

For this project:

```text
class_id = 0
class_name = battery
```

The expected `dataset.yaml` structure is:

```yaml
path: /content/drive/MyDrive/E-waste Battery Extraction CV/final_dataset
train: images/train
val: images/val
test: images/test

nc: 1
names:
  - battery
```

---

## Evaluation subset

The evaluation notebook creates a fixed selected test subset at:

```text
eval_test_subset_original_plus_selected_aug/
```

The selection rule is:

```text
original image + aug_1/aug_2/aug_3 if at least four augmentations exist,
otherwise all available augmentations are included.
```

The evaluation subset is also converted into COCO format for Detectron2 and RTMDet evaluation:

```text
eval_coco_annotations/test_selected.json
final_dataset/mmdet_annotations/test_selected_original_plus_aug.json
```

This keeps the evaluation consistent across YOLO, Detectron2, and MMDetection-based models.

---

## Using the stored weights

The `weights/` folder contains the final saved checkpoints. The evaluation notebook originally expects the weights inside the Google Drive training-output structure. If running directly from this GitHub branch, either update `MODEL_PATHS` in the evaluation notebook or copy the weights into the expected Drive folders.

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

A simple Colab copy helper can be used after cloning the repository:

```python
from pathlib import Path
import shutil

REPO_ROOT = Path("/content/Capstone-Redback-Project_7-Computer_Vision")
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

## ONNX exports

The `onnx/` folder contains selected YOLO segmentation models exported to ONNX format for MATLAB integration:

```text
onnx/yolov8n-seg_best.onnx
onnx/yolo11n-seg_best.onnx
onnx/yolo26n-seg_best.onnx
```

These files are deployment-oriented exports of the corresponding `.pt` checkpoints in the `weights/` folder. They are mainly intended for testing model handover from the Python/Colab training workflow into the MATLAB-based perception or simulation workflow.

The ONNX folder currently includes only the selected Ultralytics-compatible YOLO models. ONNX exports for YOLOv5, YOLOv7, Detectron2, and MMDetection/RTMDet models are not included in this branch. The Detectron2 and RTMDet models should be loaded from their original `.pth` checkpoints together with their matching configuration files, as described below.

---

## Important configuration notes

### Detectron2 models

The Mask R-CNN and PointRend checkpoints require their matching Detectron2 configuration files.

The benchmark notebook saves these configs as:

```text
02_detectron2_models/model_05_mask_rcnn_R50_FPN/config_used.yaml
02_detectron2_models/model_07_pointrend_mask_rcnn_R50_FPN/config_used.yaml
```

These config files are needed for evaluation and inference because Detectron2 must rebuild the exact model structure before loading the `.pth` weights.

If the config files are not available, rerun the Detectron2 setup/config cells in `E_waste_8_model_benchmark.ipynb`, or add the generated config files to the repository.

---

### RTMDet-Ins tiny

The RTMDet checkpoint requires the matching MMDetection config:

```text
03_rtmdet_models/model_07_rtmdet_ins_tiny/rtmdet_ins_tiny_battery.py
```

This config is generated by `RTMDet_Ins_Tiny_EWaste_Colab.ipynb`.

If only the checkpoint is available, rerun the config-generation cell in the RTMDet notebook before evaluation or inference.

---

## Library and compatibility notes

### General Colab runtime

Recommended runtime:

```text
Runtime: Google Colab
Hardware accelerator: GPU
Preferred GPU: A100
Image size: 640
```

The YOLO and Detectron2 notebooks use the default Colab Python environment. The RTMDet notebook uses a separate micromamba environment.

---

### YOLO-family models

The YOLO-family parts use different toolchains:

```text
YOLOv5n-seg       → ultralytics/yolov5 repository
YOLOv7-seg        → yolov7-segmentation repository
YOLOv8n-seg       → Ultralytics package
YOLO11n-seg       → Ultralytics package
YOLO26n-seg       → Ultralytics-compatible checkpoint/model
```

Common packages:

```bash
pip install ultralytics opencv-python pycocotools pandas numpy tqdm pyyaml
```

YOLOv7 also requires:

```bash
pip install filterpy
```

The YOLOv5 and YOLOv7 sections contain repository-specific setup and compatibility patches. These cells should be run as written in the notebooks instead of mixing the repositories manually.

---

### Detectron2 compatibility

Detectron2 is installed from source in Colab:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

PointRend requires the Detectron2 project folder:

```text
/content/detectron2_repo/projects/PointRend
```

The notebook handles PointRend registration carefully to avoid duplicate registration errors. If PointRend import or registry errors occur, restart the Colab runtime and rerun the Detectron2 setup cells in order.

Detectron2 compatibility can be sensitive to Colab's current Python, PyTorch, and CUDA versions. For best reproducibility, run the Detectron2 sections in a fresh Colab runtime and avoid installing MMDetection into the same default environment.

---

### MMDetection / RTMDet compatibility

RTMDet-Ins tiny uses a separate micromamba environment because MMDetection is more sensitive to version conflicts.

The RTMDet notebook uses:

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

The key installation pattern is:

```bash
micromamba create -y -n rtmdet310 python=3.10 pip

micromamba run -n rtmdet310 python -m pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

micromamba run -n rtmdet310 python -m pip install \
    mmcv==2.1.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

git clone --branch v3.3.0 https://github.com/open-mmlab/mmdetection.git /content/mmdetection
cd /content/mmdetection
micromamba run -n rtmdet310 python -m pip install -e . --no-build-isolation
```

When running MMDetection commands, the notebook sets:

```bash
PYTHONPATH=/content/mmdetection:$PYTHONPATH
MPLBACKEND=Agg
MAMBA_ROOT_PREFIX=/content/micromamba
```

These environment variables are important for stable RTMDet training and evaluation in Colab.

---

## Running the project

### 1. Clone this branch

```bash
git clone -b tthanh https://github.com/tthanh05/Capstone-Redback-Project_7-Computer_Vision.git
cd Capstone-Redback-Project_7-Computer_Vision
```

If Git LFS is used for the weight files, run:

```bash
git lfs install
git lfs pull
```

---

### 2. Prepare the dataset

Place the private dataset in Google Drive using the expected structure:

```text
/content/drive/MyDrive/E-waste Battery Extraction CV/final_dataset/
```

Then check that the following paths exist:

```text
final_dataset/dataset.yaml
final_dataset/images/train
final_dataset/images/val
final_dataset/images/test
final_dataset/labels/train
final_dataset/labels/val
final_dataset/labels/test
```

---

### 3. Run training notebooks if needed

For YOLO and Detectron2 training:

```text
codes/E_waste_8_model_benchmark.ipynb
```

For RTMDet-Ins tiny training:

```text
codes/RTMDet_Ins_Tiny_EWaste_Colab.ipynb
```

---

### 4. Run final evaluation

Use:

```text
codes/E_waste_8_model_evaluation.ipynb
```

Before running the evaluation notebook, confirm that:

```text
the dataset exists,
the model weights exist,
Detectron2 config_used.yaml files exist,
RTMDet config file exists,
ONNX files exist if MATLAB-side export testing is required.
```

The final output table is saved to:

```text
04_metrics_and_visualisations/eight_model_eval_selected_test/eight_models_9_metrics_selected_test.csv
```

The qualitative visualisation is saved to:

```text
04_metrics_and_visualisations/qualitative_examples_8_models/
```

---

## Notes and limitations

The dataset is private and is not included in this branch.

The saved weights are included for reproducibility, but Detectron2 and RTMDet models also require their matching configuration files. The ONNX exports are provided only for the selected YOLO models listed in the `onnx/` folder.

The evaluation results depend on the selected test subset created by the notebook. To compare models fairly, all models should be evaluated on the same generated subset.

YOLO, Detectron2, and MMDetection use different ecosystems. For stability, avoid installing all dependencies into one environment. The recommended workflow is:

```text
YOLO + Detectron2: default Colab environment
RTMDet/MMDetection: separate Python 3.10 micromamba environment
```

If Colab package conflicts occur, restart the runtime and rerun the setup cells from the beginning.

---

## Project purpose

This branch provides evidence of the model training, evaluation, weight management, and selected ONNX export work completed for the e-waste battery extraction computer vision task. It supports reproducible comparison across lightweight YOLO models, two-stage Detectron2 models, and RTMDet-Ins tiny using a consistent single-class segmentation evaluation pipeline.
