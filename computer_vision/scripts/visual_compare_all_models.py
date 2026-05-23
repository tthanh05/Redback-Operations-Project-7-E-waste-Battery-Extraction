import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision
from torchvision.transforms import functional as F


# =====================================================
# MAIN PATHS
# =====================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images", "test")
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
#   2. Find the folder for each model
#   3. Make sure it contains: weights/best.pt
#   4. Replace the folder names below accordingly
# =====================================================

YOLO_MODELS = {
    "YOLOv8n-seg": os.path.join(BASE_DIR, "runs", "segment", "yolov8n_battery_seg2", "weights", "best.pt"),
    "YOLOv26n-seg": os.path.join(BASE_DIR, "runs", "segment", "yolo26n_battery_seg", "weights", "best.pt"),
    "YOLOv11x-seg": os.path.join(BASE_DIR, "runs", "segment", "train6", "weights", "best.pt"),
}

SSD_MODEL_PATH = os.path.join(BASE_DIR, "SSD", "ssd_battery_best.pth")
RETINANET_MODEL_PATH = os.path.join(BASE_DIR, "RetinaNet", "retinanet_battery_best.pth")

CONF_THRESHOLD = 0.001
SAMPLE_PERCENT = 0.05
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
# VISUAL FUNCTIONS
# =====================================================

def overlay_mask(image, mask, color=(0, 255, 0)):
    output = image.copy()
    colored = image.copy()

    colored[mask > 0] = color
    output = cv2.addWeighted(output, 0.60, colored, 0.40, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 3)

    c = mask_centroid(mask)
    if c is not None:
        cv2.circle(output, (int(c[0]), int(c[1])), 6, (255, 0, 0), -1)

    return output


def draw_box(image, box, color=(0, 255, 0)):
    output = image.copy()

    if box is not None:
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(output, (cx, cy), 6, (255, 0, 0), -1)

    return output


# =====================================================
# LOAD YOLO MODELS
# =====================================================

print("Loading YOLO models...")

loaded_yolo_models = {}

for name, path in YOLO_MODELS.items():
    if os.path.exists(path):
        loaded_yolo_models[name] = YOLO(path)
        print(f"Loaded {name}: {path}")
    else:
        print(f"Missing YOLO model: {path}")


# =====================================================
# LOAD SSD
# =====================================================

print("\nLoading SSD...")

ssd_model = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=2)
ssd_model.load_state_dict(torch.load(SSD_MODEL_PATH, map_location=DEVICE))
ssd_model.to(DEVICE)
ssd_model.eval()

print("Loaded SSD")


# =====================================================
# LOAD RETINANET
# =====================================================

print("\nLoading RetinaNet...")

retinanet_model = torchvision.models.detection.retinanet_resnet50_fpn(weights=None, num_classes=2)
retinanet_model.load_state_dict(torch.load(RETINANET_MODEL_PATH, map_location=DEVICE))
retinanet_model.to(DEVICE)
retinanet_model.eval()

print("Loaded RetinaNet")


# =====================================================
# PREDICTION FUNCTIONS
# =====================================================

def predict_yolo_segmentation(model, model_name, image_path, w, h):
    pred_mask = np.zeros((h, w), dtype=np.uint8)

    results = model.predict(
        image_path,
        conf=CONF_THRESHOLD,
        imgsz=640,
        verbose=False
    )

    result = results[0]

    print(f"\nRunning {model_name}")
    print("Image:", os.path.basename(image_path))

    if result.masks is not None:
        print(f"{model_name}: segmentation masks found")

        masks = result.masks.data.cpu().numpy()

        for m in masks:
            m = cv2.resize(m, (w, h))
            m = (m > 0.3).astype(np.uint8) * 255
            pred_mask = cv2.bitwise_or(pred_mask, m)

    elif result.boxes is not None and len(result.boxes) > 0:
        print(f"{model_name}: no masks found, using box fallback")

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(pred_mask, (x1, y1), (x2, y2), 255, -1)

    else:
        print(f"{model_name}: no detection found")

    return pred_mask


def predict_box_model(model, image_rgb):
    img_tensor = F.to_tensor(image_rgb).to(DEVICE)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = prediction["boxes"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().numpy()

    if len(boxes) == 0:
        return None

    valid_indexes = np.where(scores >= CONF_THRESHOLD)[0]

    if len(valid_indexes) == 0:
        return None

    best_index = valid_indexes[np.argmax(scores[valid_indexes])]
    return boxes[best_index].tolist()


# =====================================================
# SELECT 5% OF TEST IMAGES
# =====================================================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

sample_count = max(1, int(len(image_files) * SAMPLE_PERCENT))
sample_images = random.sample(image_files, sample_count)

print("\nTotal test images:", len(image_files))
print("Selected images:", sample_count)
print("Output folder:", OUTPUT_DIR)


# =====================================================
# CREATE COMPARISON IMAGES
# =====================================================

for image_file in sample_images:
    image_path = os.path.join(IMAGE_DIR, image_file)
    label_path = os.path.join(
        LABEL_DIR,
        os.path.splitext(image_file)[0] + ".txt"
    )

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print("Could not read:", image_path)
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]

    gt_mask = yolo_seg_to_mask(label_path, w, h)
    gt_box = mask_to_box(gt_mask)

    visuals = []
    titles = []

    visuals.append(image_rgb)
    titles.append("Original")

    gt_visual = overlay_mask(image_rgb, gt_mask)
    visuals.append(gt_visual)
    titles.append("Ground Truth Mask")

    # =========================
    # YOLO MODELS
    # =========================

    for model_name, model in loaded_yolo_models.items():
        pred_mask = predict_yolo_segmentation(
            model,
            model_name,
            image_path,
            w,
            h
        )

        iou = mask_iou(gt_mask, pred_mask)
        ce = mask_ce(gt_mask, pred_mask)

        yolo_visual = overlay_mask(image_rgb, pred_mask)

        visuals.append(yolo_visual)

        if ce is None:
            titles.append(f"{model_name}\nMask IoU: {iou:.3f}\nCE: N/A")
        else:
            titles.append(f"{model_name}\nMask IoU: {iou:.3f}\nCE: {ce:.2f}px")

        # Save each YOLO output separately
        individual_name = (
            os.path.splitext(image_file)[0]
            + "_"
            + model_name.replace(" ", "_").replace("/", "_")
            + ".png"
        )

        individual_path = os.path.join(OUTPUT_DIR, individual_name)

        plt.figure(figsize=(6, 6))
        plt.imshow(yolo_visual)
        plt.title(titles[-1], fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(individual_path, dpi=300)
        plt.close()

        print("Saved individual YOLO image:", individual_path)

    # =========================
    # SSD
    # =========================

    ssd_box = predict_box_model(ssd_model, image_rgb)
    ssd_iou = box_iou(gt_box, ssd_box)
    ssd_error = box_ce(gt_box, ssd_box)

    ssd_visual = draw_box(image_rgb, ssd_box)
    visuals.append(ssd_visual)

    if ssd_error is None:
        titles.append(f"SSD\nBox IoU: {ssd_iou:.3f}\nCE: N/A")
    else:
        titles.append(f"SSD\nBox IoU: {ssd_iou:.3f}\nCE: {ssd_error:.2f}px")

    # =========================
    # RETINANET
    # =========================

    ret_box = predict_box_model(retinanet_model, image_rgb)
    ret_iou = box_iou(gt_box, ret_box)
    ret_error = box_ce(gt_box, ret_box)

    ret_visual = draw_box(image_rgb, ret_box)
    visuals.append(ret_visual)

    if ret_error is None:
        titles.append(f"RetinaNet\nBox IoU: {ret_iou:.3f}\nCE: N/A")
    else:
        titles.append(f"RetinaNet\nBox IoU: {ret_iou:.3f}\nCE: {ret_error:.2f}px")

    # =========================
    # SAVE COMPARISON IMAGE
    # =========================

    plt.figure(figsize=(34, 7))

    for i, visual in enumerate(visuals):
        plt.subplot(1, len(visuals), i + 1)
        plt.imshow(visual)
        plt.title(titles[i], fontsize=9)
        plt.axis("off")

    save_name = os.path.splitext(image_file)[0] + "_comparison.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved comparison image:", save_path)


print("\nFinished.")
print("All outputs saved in:")
print(OUTPUT_DIR)
