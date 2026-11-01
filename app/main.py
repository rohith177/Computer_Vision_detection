
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import shutil
import uuid
import io

app = FastAPI(title="Steel Defect Detection API")

# Load trained YOLO model once
MODEL_PATH = (
    "runs/detect/runs/detect/"
    "neu_defect_detection-8/weights/best.pt"
)

model = YOLO(MODEL_PATH)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def run_detection(image_path):

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=416,
        save=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "defect": model.names[class_id],
                "confidence": round(confidence * 100, 2),
                "bounding_box": [
                    round(value, 2) for value in coordinates
                ]
            })

    return results, detections


@app.get("/")
def home():
    return {
        "message": "Steel Defect Detection API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image"
        )

    extension = ".png" if file.content_type == "image/png" else ".jpg"
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

    try:
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results, detections = run_detection(image_path)

        return {
            "filename": file.filename,
            "total_defects_detected": len(detections),
            "detections": detections
        }

    finally:
        await file.close()
        image_path.unlink(missing_ok=True)


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image"
        )

    extension = ".png" if file.content_type == "image/png" else ".jpg"
    image_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

    try:
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results, detections = run_detection(image_path)

        # Draw bounding boxes on the image
        annotated_image = results[0].plot()

        # Convert OpenCV image to PNG bytes
        success, encoded_image = __import__("cv2").imencode(
            ".png", annotated_image
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Could not generate annotated image"
            )

        return StreamingResponse(
            io.BytesIO(encoded_image.tobytes()),
            media_type="image/png",
            headers={
                "X-Defect-Count": str(len(detections))
            }
        )

    finally:
        await file.close()
        image_path.unlink(missing_ok=True)
