# main.py
import base64
import cv2
import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Initialize the FastAPI app
app = FastAPI(title="YOLOv8 ONNX Inference API")

# --- Model Loading ---
# Load the ONNX model once when the application starts
try:
    providers = ['CPUExecutionProvider']
    onnx_session = onnxruntime.InferenceSession("yolov8n.onnx", providers=providers)
    model_inputs = onnx_session.get_inputs()
    input_shape = model_inputs[0].shape
    input_height, input_width = input_shape[2], input_shape[3]
    print("ONNX model loaded successfully.")
except Exception as e:
    onnx_session = None
    print(f"Error loading ONNX model: {e}")

# Pydantic model for the request body to ensure data is in the correct format
class ImagePayload(BaseModel):
    image: str  # Base64 encoded image string

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "YOLO API is running."}

@app.post("/detect")
def detect_objects(payload: ImagePayload):
    if onnx_session is None:
        return {"error": "Model not loaded or failed to load."}

    # 1. Decode the image
    try:
        img_bytes = base64.b64decode(payload.image)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        return {"error": f"Failed to decode or read image: {e}"}

    # 2. Pre-process the image
    img_resized = cv2.resize(image, (input_width, input_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    input_tensor = img_transposed.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 3. Run Inference
    try:
        input_name = model_inputs[0].name
        outputs = onnx_session.run(None, {input_name: input_tensor})
        raw_output = outputs[0]
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    # In a real app, you would post-process 'raw_output' here to get bounding boxes.
    # For now, we return a success message and the shape of the output.
    return {
        "message": "Inference successful!",
        "output_shape": raw_output.shape,
        "detections": "Post-processing logic to be implemented here."
    }

# This part allows running the app locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)