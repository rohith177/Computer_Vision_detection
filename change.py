
import os
import xml.etree.ElementTree as ET

# Dataset location
BASE_DIR = ""

# Six NEU-DET defect classes
classes = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

class_map = {
    name: index for index, name in enumerate(classes)
}


def convert_annotations(split):

    annotation_dir = os.path.join(
        BASE_DIR, split, "annotations"
    )

    label_dir = os.path.join(
        BASE_DIR, split, "labels"
    )

    os.makedirs(label_dir, exist_ok=True)

    for xml_file in os.listdir(annotation_dir):

        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(annotation_dir, xml_file)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        yolo_lines = []

        for obj in root.findall("object"):

            class_name = obj.find("name").text

            if class_name not in class_map:
                print("Unknown class:", class_name)
                continue

            class_id = class_map[class_name]

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            # Convert to YOLO coordinates
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height

            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_lines.append(
                f"{class_id} {x_center:.6f} "
                f"{y_center:.6f} {box_width:.6f} "
                f"{box_height:.6f}"
            )

        # Create matching TXT file
        txt_file = os.path.splitext(xml_file)[0] + ".txt"

        txt_path = os.path.join(label_dir, txt_file)

        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

    print(f"{split} conversion completed!")


convert_annotations("train")
convert_annotations("validation")
