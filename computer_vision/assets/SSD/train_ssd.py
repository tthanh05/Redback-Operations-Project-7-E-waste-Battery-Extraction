import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
from torchvision.models import VGG16_Weights
import pandas as pd
from tqdm import tqdm

# =========================
# PATHS (UPDATE IF NEEDED)
# =========================
ROOT = Path(r"D:/OnTrack Tasks/SIT374 Capstone/Training and Stuff/final_dataset-20260424T135306Z-3-001/final_dataset")

TRAIN_JSON = ROOT / "annotations_coco" / "train_clean.json"
VAL_JSON = ROOT / "annotations_coco" / "val_clean.json"

TRAIN_IMG_DIR = ROOT / "images" / "train"
VAL_IMG_DIR = ROOT / "images" / "val"

# =========================
# CONFIG
# =========================
NUM_CLASSES = 2   # background + battery
EPOCHS = 50
BATCH_SIZE = 4
LR = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# DATASET CLASS
# =========================
class CocoBatteryDataset(Dataset):
    def __init__(self, img_dir, annotation_file):
        self.img_dir = Path(img_dir)

        with open(annotation_file, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # Group annotations by image
        self.ann_by_image = {}
        for ann in self.annotations:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        anns = self.ann_by_image.get(img_info["id"], [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # battery class

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_info["id"]])
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# LOAD DATA
# =========================
train_dataset = CocoBatteryDataset(TRAIN_IMG_DIR, TRAIN_JSON)
val_dataset = CocoBatteryDataset(VAL_IMG_DIR, VAL_JSON)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# =========================
# MODEL
# =========================
model = ssd300_vgg16(
    weights=None,
    weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
    num_classes=NUM_CLASSES
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =========================
# TRAINING LOOP
# =========================
history = []
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =========================
    # VALIDATION
    # =========================
    model.train()
    val_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Validation"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), ROOT / "ssd_battery_best.pth")
        print("✅ Saved best model")

# =========================
# SAVE RESULTS
# =========================
pd.DataFrame(history).to_csv(ROOT / "ssd_training_results.csv", index=False)

print("\n🎉 Training complete.")
print("Best model:", ROOT / "ssd_battery_best.pth")
print("Results file:", ROOT / "ssd_training_results.csv")