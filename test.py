
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/runs/detect/neu_defect_detection-8/weights/best.pt")

# Evaluate using validation images and labels
metrics = model.val(
    data="data.yaml",
    split="val",
    imgsz=416,
    batch=2,
    device=0,
    workers=0,
    plots=True
)

# Display performance
print("\nMODEL PERFORMANCE")
print("-------------------------")

print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
