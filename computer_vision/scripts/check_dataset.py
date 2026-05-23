import os

# Change these paths if needed
image_folder = "images/train"
label_folder = "labels/train"

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# Get image names without extension
image_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(image_folder)
    if os.path.splitext(f)[1].lower() in image_extensions
]

# Get label names without extension
label_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(label_folder)
    if f.endswith(".txt")
]

image_set = set(image_files)
label_set = set(label_files)

# Matching files
matched = image_set.intersection(label_set)

# Problems
images_without_labels = image_set - label_set
labels_without_images = label_set - image_set

print("===== DATASET MATCH CHECK =====")
print(f"Total images: {len(image_set)}")
print(f"Total labels: {len(label_set)}")
print(f"Matched image-label pairs: {len(matched)}")

print("\nImages without labels:")
for name in sorted(images_without_labels):
    print(name)

print("\nLabels without images:")
for name in sorted(labels_without_images):
    print(name)
