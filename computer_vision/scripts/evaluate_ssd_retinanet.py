import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops import box_iou
from torchvision.models.detection import retinanet_resnet50_fpn, ssd300_vgg16
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ======================================================
# CONFIG
# ======================================================
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODEL_TYPE = "ssd"   # change to "retinanet" for RetinaNet

if MODEL_TYPE == "ssd":
    WEIGHTS_PATH = os.path.join(BASE_DIR, "SSD", "ssd_battery_best.pth")
elif MODEL_TYPE == "retinanet":
    WEIGHTS_PATH = os.path.join(BASE_DIR, "RetinaNet", "retinanet_battery_best.pth")

IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images", "val")
COCO_JSON = os.path.join(BASE_DIR, "dataset", "annotations_coco", "val_clean.json")

NUM_CLASSES = 2  # background + battery
IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():
    if MODEL_TYPE == "retinanet":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=NUM_CLASSES
        )

    elif MODEL_TYPE == "ssd":
        model = ssd300_vgg16(
            weights=None,
            weights_backbone=None,
            num_classes=NUM_CLASSES
        )

    else:
        raise ValueError("MODEL_TYPE must be 'retinanet' or 'ssd'")

    print(f"Loading weights from: {WEIGHTS_PATH}")

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    # Handles both direct state_dict and checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print(f"{MODEL_TYPE} model loaded successfully.")
    return model


# ======================================================
# LOAD COCO ANNOTATIONS
# ======================================================

def load_coco_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    image_id_to_name = {}
    image_to_boxes = {}

    for img in data["images"]:
        image_id_to_name[img["id"]] = img["file_name"]

    for ann in data["annotations"]:
        image_id = ann["image_id"]

        # COCO bbox format: [x, y, width, height]
        x, y, w, h = ann["bbox"]

        # Convert to xyxy format
        box = [x, y, x + w, y + h]

        if image_id not in image_to_boxes:
            image_to_boxes[image_id] = []

        image_to_boxes[image_id].append(box)

    return image_id_to_name, image_to_boxes


# ======================================================
# EVALUATION
# ======================================================

def evaluate_model():
    model = load_model()

    image_id_to_name, image_to_boxes = load_coco_annotations(COCO_JSON)

    all_scores = []
    all_tp = []

    y_true_cm = []
    y_pred_cm = []

    total_gt_objects = 0

    for image_id, file_name in tqdm(image_id_to_name.items(), desc="Evaluating"):
        image_path = os.path.join(IMAGE_DIR, file_name)

        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).to(DEVICE)

        gt_boxes = image_to_boxes.get(image_id, [])
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32).to(DEVICE)

        total_gt_objects += len(gt_boxes)

        with torch.no_grad():
            output = model([image_tensor])[0]

        pred_boxes = output["boxes"]
        pred_scores = output["scores"]
        pred_labels = output["labels"]

        # Keep only battery class predictions
        battery_mask = pred_labels == 1
        pred_boxes = pred_boxes[battery_mask]
        pred_scores = pred_scores[battery_mask]

        matched_gt = set()

        for i in range(len(pred_boxes)):
            score = pred_scores[i].item()
            pred_box = pred_boxes[i].unsqueeze(0)

            is_true_positive = 0

            if len(gt_boxes) > 0:
                ious = box_iou(pred_box, gt_boxes)[0]
                best_iou, best_gt_idx = torch.max(ious, dim=0)

                if best_iou >= IOU_THRESHOLD and best_gt_idx.item() not in matched_gt:
                    is_true_positive = 1
                    matched_gt.add(best_gt_idx.item())

            all_scores.append(score)
            all_tp.append(is_true_positive)

        # Confusion matrix at image level
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            y_true_cm.append(1)
            y_pred_cm.append(1)
        elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
            y_true_cm.append(1)
            y_pred_cm.append(0)
        elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
            y_true_cm.append(0)
            y_pred_cm.append(1)
        else:
            y_true_cm.append(0)
            y_pred_cm.append(0)

    return (
        np.array(all_scores),
        np.array(all_tp),
        y_true_cm,
        y_pred_cm,
        total_gt_objects
    )


# ======================================================
# PLOT CURVES
# ======================================================

def plot_curves(scores, tp, y_true_cm, y_pred_cm, total_gt_objects):
    thresholds = np.linspace(0.01, 0.99, 100)

    precision_list = []
    recall_list = []
    f1_list = []

    for conf in thresholds:
        selected = scores >= conf

        tp_count = tp[selected].sum()
        fp_count = selected.sum() - tp_count
        fn_count = total_gt_objects - tp_count

        precision = tp_count / (tp_count + fp_count + 1e-6)
        recall = tp_count / (tp_count + fn_count + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "evaluation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_name = MODEL_TYPE.upper()

    # F1 Curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_list)
    plt.xlabel("Confidence")
    plt.ylabel("F1 Score")
    plt.title(f"{model_name} F1 Curve")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_BoxF1_curve.png"), dpi=300)
    plt.close()

    # Precision Curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision_list)
    plt.xlabel("Confidence")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision Curve")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_BoxP_curve.png"), dpi=300)
    plt.close()

    # Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, recall_list)
    plt.xlabel("Confidence")
    plt.ylabel("Recall")
    plt.title(f"{model_name} Recall Curve")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_BoxR_curve.png"), dpi=300)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_list, precision_list)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Precision-Recall Curve")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_BoxPR_curve.png"), dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true_cm, y_pred_cm, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Background", "Battery"]
    )

    disp.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_TYPE}_confusion_matrix.png"), dpi=300)
    plt.close()

    best_index = int(np.argmax(f1_list))

    print("\n===== Evaluation Summary =====")
    print("Model:", model_name)
    print("Best Confidence:", round(thresholds[best_index], 3))
    print("Best F1:", round(f1_list[best_index], 4))
    print("Precision at Best F1:", round(precision_list[best_index], 4))
    print("Recall at Best F1:", round(recall_list[best_index], 4))

    print("\nSaved images:")
    print(f"{MODEL_TYPE}_BoxF1_curve.png")
    print(f"{MODEL_TYPE}_BoxP_curve.png")
    print(f"{MODEL_TYPE}_BoxR_curve.png")
    print(f"{MODEL_TYPE}_BoxPR_curve.png")
    print(f"{MODEL_TYPE}_confusion_matrix.png")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    scores, tp, y_true_cm, y_pred_cm, total_gt_objects = evaluate_model()
    plot_curves(scores, tp, y_true_cm, y_pred_cm, total_gt_objects)
