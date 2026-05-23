import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torchvision
from torchvision.transforms import functional as F


# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset" , "images", "test")
LABEL_DIR = os.path.join(BASE_DIR, "dataset" , "labels", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "model_metrics_summary.csv")


# =====================================================
# MODEL PATHS
# =====================================================
# =====================================================
# ⚠️ UPDATE THESE PATHS BEFORE RUNNING
# =====================================================
# The folder names inside runs/segment/ are generated automatically
# by YOLO during training and will be different on your machine.
#
# To find your folder names:
#   1. Look inside: dataset/runs/segment/
#   2. Find the folder for each model (e.g. "train", "train2", "yolov8n_seg")
#   3. Make sure it contains: weights/best.pt
#   4. Replace the folder names below accordingly
#
# Example:
#   "YOLOv8n-seg":  .../runs/segment/train/weights/best.pt
#   "YOLOv11x-seg": .../runs/segment/train2/weights/best.pt
#   "YOLOv26n-seg": .../runs/segment/train3/weights/best.pt
# =====================================================

YOLO_MODELS = {
    "YOLOv8n-seg": os.path.join(BASE_DIR, "runs", "segment", "yolov8n_battery_seg2", "weights", "best.pt"),
    "YOLOv26n-seg": os.path.join(BASE_DIR, "runs", "segment", "yolo26n_battery_seg", "weights", "best.pt"),
    "YOLOv11x-seg": os.path.join(BASE_DIR, "runs", "segment", "train6", "weights", "best.pt"),
}

SSD_MODEL_PATH = os.path.join(BASE_DIR, "SSD", "ssd_battery_best.pth")
RETINANET_MODEL_PATH = os.path.join(BASE_DIR, "RetinaNet", "retinanet_battery_best.pth")

CONF_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# GROUND TRUTH FUNCTIONS
# =====================================================

def yolo_seg_to_mask(label_path, img_w, img_h):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()

        if len(values) < 7:
            continue

        coords = list(map(float, values[1:]))
        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_w)
            y = int(coords[i + 1] * img_h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    return mask


def mask_to_box(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# =====================================================
# METRIC FUNCTIONS
# =====================================================

def mask_iou(mask1, mask2):
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return intersection / union


def mask_centroid(mask):
    m = cv2.moments(mask)

    if m["m00"] == 0:
        return None

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])

    return np.array([cx, cy])


def mask_ce(gt_mask, pred_mask):
    gt_c = mask_centroid(gt_mask)
    pred_c = mask_centroid(pred_mask)

    if gt_c is None or pred_c is None:
        return None

    return np.linalg.norm(gt_c - pred_c)


def box_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = areaA + areaB - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def box_ce(gt_box, pred_box):
    if gt_box is None or pred_box is None:
        return None

    gt_cx = (gt_box[0] + gt_box[2]) / 2
    gt_cy = (gt_box[1] + gt_box[3]) / 2

    pred_cx = (pred_box[0] + pred_box[2]) / 2
    pred_cy = (pred_box[1] + pred_box[3]) / 2

    return np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2)


# =====================================================
# PREDICTION FUNCTIONS
# =====================================================

def predict_yolo_mask(model, image_path, w, h):
    start = time.perf_counter()

    results = model.predict(
        image_path,
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    pred_mask = np.zeros((h, w), dtype=np.uint8)
    result = results[0]

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()

        for m in masks:
            m = cv2.resize(m, (w, h))
            m = (m > 0.5).astype(np.uint8) * 255
            pred_mask = cv2.bitwise_or(pred_mask, m)

    latency_ms = (end - start) * 1000
    return pred_mask, latency_ms


def predict_box_model(model, image_rgb):
    img_tensor = F.to_tensor(image_rgb).to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    latency_ms = (end - start) * 1000

    if len(boxes) == 0:
        return None, latency_ms

    valid_indexes = np.where(scores >= CONF_THRESHOLD)[0]

    if len(valid_indexes) == 0:
        return None, latency_ms

    best_index = valid_indexes[np.argmax(scores[valid_indexes])]
    return boxes[best_index].tolist(), latency_ms


# =====================================================
# LOAD MODELS
# =====================================================

print("Using device:", DEVICE)

loaded_yolo_models = {}

for name, path in YOLO_MODELS.items():
    if os.path.exists(path):
        loaded_yolo_models[name] = YOLO(path)
        print("Loaded:", name)
    else:
        print("Missing:", name, path)


print("Loading SSD...")
ssd_model = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=2)
ssd_model.load_state_dict(torch.load(SSD_MODEL_PATH, map_location=DEVICE))
ssd_model.to(DEVICE)
ssd_model.eval()
print("Loaded SSD")


print("Loading RetinaNet...")
retinanet_model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes=2)
retinanet_model.load_state_dict(torch.load(RETINANET_MODEL_PATH, map_location=DEVICE))
retinanet_model.to(DEVICE)
retinanet_model.eval()
print("Loaded RetinaNet")


# =====================================================
# LOAD TEST IMAGES
# =====================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Total test images:", len(image_files))


# =====================================================
# EVALUATE YOLO MODELS
# =====================================================

summary_rows = []

for model_name, model in loaded_yolo_models.items():
    print("\nEvaluating:", model_name)

    ious = []
    ces = []
    latencies = []

    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(image_file)[0] + ".txt"
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        gt_mask = yolo_seg_to_mask(label_path, w, h)
        pred_mask, latency_ms = predict_yolo_mask(model, image_path, w, h)

        iou = mask_iou(gt_mask, pred_mask)
        ce = mask_ce(gt_mask, pred_mask)

        ious.append(iou)
        latencies.append(latency_ms)

        if ce is not None:
            ces.append(ce)

    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    summary_rows.append({
        "Model": model_name,
        "Output Type": "Segmentation Mask",
        "IoU Type": "Mask IoU",
        "Average IoU": round(np.mean(ious), 4),
        "Average Centroid Error (px)": round(np.mean(ces), 2) if len(ces) > 0 else "N/A",
        "Median Centroid Error (px)": round(np.median(ces), 2) if len(ces) > 0 else "N/A",
        "Average Latency (ms/image)": round(avg_latency, 2),
        "FPS": round(fps, 2),
        "Images Evaluated": len(ious)
    })


# =====================================================
# EVALUATE SSD AND RETINANET
# =====================================================

box_models = {
    "SSD": ssd_model,
    "RetinaNet": retinanet_model,
}

for model_name, model in box_models.items():
    print("\nEvaluating:", model_name)

    ious = []
    ces = []
    latencies = []

    for image_file in image_files:
        image_path = os.path.join(IMAGE_DIR, image_file)
        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(image_file)[0] + ".txt"
        )

        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        h, w = image_rgb.shape[:2]

        gt_mask = yolo_seg_to_mask(label_path, w, h)
        gt_box = mask_to_box(gt_mask)

        pred_box, latency_ms = predict_box_model(model, image_rgb)

        iou = box_iou(gt_box, pred_box)
        ce = box_ce(gt_box, pred_box)

        ious.append(iou)
        latencies.append(latency_ms)

        if ce is not None:
            ces.append(ce)

    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    summary_rows.append({
        "Model": model_name,
        "Output Type": "Bounding Box",
        "IoU Type": "Box IoU",
        "Average IoU": round(np.mean(ious), 4),
        "Average Centroid Error (px)": round(np.mean(ces), 2) if len(ces) > 0 else "N/A",
        "Median Centroid Error (px)": round(np.median(ces), 2) if len(ces) > 0 else "N/A",
        "Average Latency (ms/image)": round(avg_latency, 2),
        "FPS": round(fps, 2),
        "Images Evaluated": len(ious)
    })


# =====================================================
# SAVE RESULTS
# =====================================================

df = pd.DataFrame(summary_rows)

print("\n===== MODEL EVALUATION SUMMARY =====")
print(df.to_string(index=False))

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved CSV table to:")
print(OUTPUT_CSV)
