
# =====================================================
# DATA AUGMENTATION SCRIPT
# =====================================================
# Run this ONCE before training to expand the dataset.
# For each original image, creates AUG_PER_IMAGE augmented
# copies with transformed images AND updated segmentation labels.
# Safe to re-run — skips images already containing "_aug".
# =====================================================

import os
import cv2
import albumentations as A

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..")

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

AUG_PER_IMAGE = 4
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.HueSaturationValue(p=0.4),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
)


def read_yolo_seg_label(label_path):
    objects = []

    if not os.path.exists(label_path):
        return objects

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) < 7:
                continue

            class_id = values[0]
            coords = list(map(float, values[1:]))

            points = []
            for i in range(0, len(coords), 2):
                points.append((coords[i], coords[i + 1]))

            objects.append((class_id, points))

    return objects


def save_yolo_seg_label(label_path, objects):
    with open(label_path, "w") as file:
        for class_id, points in objects:
            clean_points = []

            for x, y in points:
                x = min(max(x, 0.0), 1.0)
                y = min(max(y, 0.0), 1.0)
                clean_points.extend([x, y])

            line = class_id + " " + " ".join(f"{v:.6f}" for v in clean_points)
            file.write(line + "\n")


image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    and "_aug" not in os.path.splitext(f)[0]
]

for image_name in image_files:
    name, ext = os.path.splitext(image_name)

    image_path = os.path.join(IMAGE_DIR, image_name)
    label_path = os.path.join(LABEL_DIR, name + ".txt")

    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping unreadable image: {image_name}")
        continue

    objects = read_yolo_seg_label(label_path)

    if not objects:
        print(f"Skipping image without valid label: {image_name}")
        continue

    height, width = image.shape[:2]

    for aug_id in range(1, AUG_PER_IMAGE + 1):
        keypoints = []
        point_counts = []
        class_ids = []

        for class_id, points in objects:
            pixel_points = [(x * width, y * height) for x, y in points]
            keypoints.extend(pixel_points)
            point_counts.append(len(pixel_points))
            class_ids.append(class_id)

        augmented = transform(image=image, keypoints=keypoints)

        aug_image = augmented["image"]
        aug_keypoints = augmented["keypoints"]

        aug_height, aug_width = aug_image.shape[:2]

        new_objects = []
        index = 0

        for class_id, count in zip(class_ids, point_counts):
            obj_points = aug_keypoints[index:index + count]
            index += count

            norm_points = []

            for x, y in obj_points:
                norm_x = x / aug_width
                norm_y = y / aug_height
                norm_points.append((norm_x, norm_y))

            new_objects.append((class_id, norm_points))

        aug_image_name = f"{name}_aug{aug_id}{ext}"
        aug_label_name = f"{name}_aug{aug_id}.txt"

        cv2.imwrite(os.path.join(IMAGE_DIR, aug_image_name), aug_image)
        save_yolo_seg_label(os.path.join(LABEL_DIR, aug_label_name), new_objects)

        print(f"Created: {aug_image_name} and {aug_label_name}")

print("Augmentation completed successfully.")
