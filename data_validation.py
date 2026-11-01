
from pathlib import Path

# Dataset configuration
DATASET_PATH = Path(__file__).parent

SPLITS = ["train", "validate"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

NUM_CLASSES = 6

def check_split(split):

    print(f"\nChecking {split.upper()} dataset")
    print("-" * 40)

    image_dir = DATASET_PATH / split / "images"
    label_dir = DATASET_PATH / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print("ERROR: Images or labels folder missing!")
        return

    images = [
        file for file in image_dir.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_names = {file.stem for file in images}
    label_names = {file.stem for file in labels}

    missing_labels = image_names - label_names
    print("Images without labels:")
    for name in missing_labels:
        for image in images:
            if image.stem == name:
                print(image)
    missing_images = label_names - image_names

    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")

    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")

    invalid_boxes = 0
    invalid_classes = 0
    empty_labels = 0

    for label_file in labels:

        with open(label_file, "r") as file:
            lines = file.readlines()

        if not lines:
            empty_labels += 1
            continue

        for line_number, line in enumerate(lines, start=1):

            values = line.strip().split()

            if len(values) != 5:
                invalid_boxes += 1
                print(f"Invalid format: {label_file.name}, line {line_number}")
                continue

            try:
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:])
            except ValueError:
                invalid_boxes += 1
                continue

            if class_id < 0 or class_id >= NUM_CLASSES:
                invalid_classes += 1

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
                and x_center - width / 2 >= 0
                and x_center + width / 2 <= 1
                and y_center - height / 2 >= 0
                and y_center + height / 2 <= 1
            ):
                invalid_boxes += 1

    print(f"Empty label files: {empty_labels}")
    print(f"Invalid bounding boxes: {invalid_boxes}")
    print(f"Invalid class IDs: {invalid_classes}")


if __name__ == "__main__":

    for split in SPLITS:
        check_split(split)

    print("\nDataset verification completed!")
