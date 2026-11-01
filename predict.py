from ultralytics import YOLO
import cv2


# Load trained model
model = YOLO(
    "runs/detect/runs/detect/neu_defect_detection-8/weights/best.pt"
)

# Predict
results = model.predict(
    source="test_image/patch.png",
    conf=0.25,
    imgsz=416,
    save=True
)

# Display predictions
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print("Defect:", model.names[class_id])
        print("Confidence:", round(confidence * 100, 2), "%")


# Display image with bounding boxes
for result in results:
    annotated_image = result.plot()

    cv2.imshow("Steel Defect Detection", annotated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()        