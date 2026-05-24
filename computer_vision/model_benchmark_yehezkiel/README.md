# E-waste Battery Instance Segmentation

Benchmarking YOLO-family instance segmentation models on a single-class e-waste battery dataset, plus a proposed hybrid refinement method (BatteryMask-RefineNet).

## Models

| # | Model | Framework |
|---|-------|-----------|
| 1 | YOLOv5n-seg | Ultralytics YOLOv5 |
| 2 | YOLOv7-seg | RizwanMunawar/yolov7-segmentation |
| 3 | YOLOv8n-seg | Ultralytics |
| 4 | YOLO11n-seg | Ultralytics |
| 5 | BatteryMask-RefineNet (proposed) | Custom PyTorch |

## BatteryMask-RefineNet

A two-stage hybrid pipeline: YOLOv8n-seg produces a coarse mask, then a lightweight U-Net style refiner crops around the detection and refines the mask boundary. Deployable as two ONNX files.

## Dataset Layout

```
final_dataset/
  dataset.yaml
  images/
    train/   val/   test/
  labels/
    train/   val/   test/
```

Labels use YOLO polygon segmentation format. 65 train / 25 val / 17 test images.

## Notebooks

| File | Purpose |
|------|---------|
| `E_waste_8_model_benchmark.ipynb` | Full training pipeline for all models |
| `ultralytics_eval.ipynb` | Standalone evaluation for YOLOv8n-seg and YOLO11n-seg |

## Requirements

- Google Colab (recommended: A100 GPU)
- Google Drive mounted at `/content/drive`
- Python 3.12, PyTorch 2.x, CUDA

```bash
pip install ultralytics opencv-python pycocotools supervision
pip install pillow==10.4.0
pip install filterpy  # required for YOLOv7
```

## Known Compatibility Fixes

The notebooks apply these patches automatically at runtime:

- `np.int` → `int`, `np.float` → `float` (NumPy 1.24+)
- `np.trapz` → `np.trapezoid` (NumPy 2.0+)
- `font.getsize` → `font.getbbox` (Pillow 10+)
- `torch.load` with `weights_only=False` (PyTorch 2.6+)

## Notes

- YOLOv7 requires batch size ≤ 8 on a T4 GPU due to memory constraints.
- Set DataLoader `num_workers=0` on Colab to avoid shared memory crashes.
- Models show high val mAP (>0.95) but reduced test generalisation on this small dataset.
