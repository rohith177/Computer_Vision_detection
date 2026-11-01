
from ultralytics import YOLO


def main():

    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Start training
    results = model.train(
        data="data.yaml",
        epochs=30,
        imgsz=416,
        batch=2,
        device=0,
        workers=0,
        name="neu_defect_detection",
        project="runs/detect",
        patience=8,
        cache=False,
        save=True,
        plots=True
    )

    print("Training completed!")


if __name__ == "__main__":
    main()

