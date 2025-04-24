from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import cv2
import numpy as np
import yolov5
from offside_detector import OffsideDetector, draw_definitive_results

app = FastAPI(title="Offside Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Load model once globally
model = yolov5.load('keremberke/yolov5m-football')
model.conf = 0.25
model.iou = 0.45
model.multi_label = True

detector = OffsideDetector()

def read_image_as_cv2(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

@app.get("/")
async def root():
    return {"message": "⚽ Offside Detection API is running!"}


@app.post("/detect-offside")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_image_as_cv2(contents)

        results = model(image)
        detections = results.pandas().xyxy[0]
        detections['class_name'] = detections['class'].map(lambda x: model.names[int(x)])

        offside_players = detector.detect_offside(detections, image)
        annotated = draw_definitive_results(image, detections, offside_players, detector)

        _, buffer = cv2.imencode(".jpg", annotated)
        image_stream = io.BytesIO(buffer.tobytes())
        image_stream.seek(0)

        return StreamingResponse(image_stream, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

# Local dev runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
