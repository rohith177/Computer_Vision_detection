
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import random

DATASET_PATH = Path(__file__).parent

# Select training images
IMAGE_DIR = DATASET_PATH / "train" / "images"
LABEL_DIR = DATASET_PATH / "train" / "labels"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Find all images, including images inside class folders
images = [
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

# Select 5 random images
selected_images = random.sample(images, min(5, len(images)))

for image_path in selected_images:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    if not label_path.exists():
        print(f"Missing label: {image_path.name}")
        continue

    with open(label_path, "r") as f:

        for line in f:

            values = line.strip().split()

            class_id = int(values[0])
            x_center, y_center, box_width, box_height = map(float, values[1:])

            # Convert YOLO coordinates to pixel coordinates
            x1 = int((x_center - box_width / 2) * width)
            y1 = int((y_center - box_height / 2) * height)
            x2 = int((x_center + box_width / 2) * width)
            y2 = int((y_center + box_height / 2) * height)

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            class_name = CLASS_NAMES[class_id]

            cv2.putText(
                image,
                class_name,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()
