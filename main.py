# main.py
import base64
import cv2
import numpy as np
import onnxruntime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
# in the exact same order they were in during training.
CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3",
    "class_4", "class_5", "class_6"
]
# ===================================================================

def post_process(raw_output, original_image, confidence_threshold=0.5, nms_threshold=0.45):
    """
    Decodes the raw YOLOv8 ONNX output into a clean list of detections.
    """
    # --- 1. Transpose and Get Shape ---
    # The output is (1, 11, 8400). Transpose to (1, 8400, 11) for easier processing.
    transposed_output = np.transpose(raw_output, (0, 2, 1))[0]
    original_height, original_width = original_image.shape[:2]

    boxes = []
    confidences = []
    class_ids = []

    # --- 2. Decode and Filter Boxes ---
    for prediction in transposed_output:
        # The first 4 elements are bbox, the rest are class scores
        box_coords = prediction[:4]
        class_scores = prediction[4:]

        # Find the class with the highest score
        class_id = np.argmax(class_scores)
        max_confidence = class_scores[class_id]

        # Filter out detections below the confidence threshold
        if max_confidence > confidence_threshold:
            confidences.append(float(max_confidence))
            class_ids.append(class_id)

            # Convert box coordinates from (center_x, center_y, width, height) to (x1, y1, width, height)
            # And scale them back to the original image size
            cx, cy, w, h = box_coords
            x_scale = original_width / input_width
            y_scale = original_height / input_height

            x1 = int((cx - w / 2) * x_scale)
            y1 = int((cy - h / 2) * y_scale)
            width = int(w * x_scale)
            height = int(h * y_scale)
            boxes.append([x1, y1, width, height])

    # --- 3. Apply Non-Max Suppression (NMS) ---
    # This removes duplicate, overlapping boxes for the same object
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) == 0:
        return []

    # --- 4. Format the Final Detections ---
    final_detections = []
    for i in indices.flatten():
        final_detections.append({
            "class_name": CLASS_NAMES[class_ids[i]],
            "confidence": confidences[i],
            "box": boxes[i] # [x_top_left, y_top_left, width, height]
        })

    return final_detections

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
        # --> Get original image dimensions BEFORE resizing
        original_height, original_width = image.shape[:2]
    except Exception as e:
        return {"error": f"Failed to decode or read image: {e}"}

    # 2. Pre-process the image (same as before)
    img_resized = cv2.resize(image, (input_width, input_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_transposed = np.transpose(img_rgb, (2, 0, 1))
    input_tensor = img_transposed.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # 3. Run Inference (same as before)
    try:
        input_name = model_inputs[0].name
        outputs = onnx_session.run(None, {input_name: input_tensor})
        raw_output = outputs[0]
    except Exception as e:
        return {"error": f"Inference failed: {e}"}

    # 4. ==> CALL YOUR NEW POST-PROCESSING FUNCTION <==
    detections = post_process(raw_output, image)

    # 5. Return the clean, final detections
    return {
        "message": "Inference successful!",
        "detections": detections
    }

# This part allows running the app locally for testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)