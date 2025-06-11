import os
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort  # <-- Import onnxruntime
import io
from PIL import Image
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- Global variables ---
onnx_session = None
mp_hands = None
mp_drawing = None
# We need to manually define the class names now
CLASS_NAMES = {0: 'Baba', 1: 'Book', 2: 'Friend', 3: 'Melon', 4: 'Photograph', 5: 'Salam', 6: 'SalamAlikum'}

# --- Load Models on Startup ---
try:
    print("Loading ONNX model...")
    # Create an ONNX Runtime inference session
    onnx_session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
    print("✅ ONNX model loaded successfully.")

    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3
    )
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe initialized successfully.")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# --- Preprocessing function for the model ---
def preprocess(image_bgr):
    """Preprocesses a single image for ONNX model inference."""
    # ONNX models from YOLO expect a specific input format
    # 1. Resize to the input size the model was trained on (e.g., 224x224)
    input_size = onnx_session.get_inputs()[0].shape[2:] # Gets (height, width) like (224, 224)
    image_resized = cv2.resize(image_bgr, (input_size[1], input_size[0]))
    
    # 2. Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize pixel values to 0-1
    image_normalized = image_rgb.astype(np.float32) / 255.0
    
    # 4. Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    image_chw = np.transpose(image_normalized, (2, 0, 1))
    
    # 5. Add a batch dimension -> (1, Channels, Height, Width)
    return np.expand_dims(image_chw, axis=0)

# --- Helper Function for Prediction ---
def predict_from_image(image_bytes):
    """Takes image bytes, performs prediction, and returns results using onnxruntime."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w, _ = cv_image.shape
        
        results_mp = hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        if results_mp.multi_hand_landmarks:
            skeleton_image = np.ones((h, w, 3), dtype=np.uint8) * 255
            for hand_landmarks in results_mp.multi_hand_landmarks:
                mp_drawing.draw_landmarks(skeleton_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess the skeleton image for the ONNX model
            input_tensor = preprocess(skeleton_image)
            
            # Run inference with onnxruntime
            input_name = onnx_session.get_inputs()[0].name
            output_name = onnx_session.get_outputs()[0].name
            
            # The result is a list of numpy arrays
            onnx_outputs = onnx_session.run([output_name], {input_name: input_tensor})
            
            # The classification scores are in the first output
            scores = onnx_outputs[0][0]
            
            # Apply softmax to convert scores (logits) to probabilities
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
            
            # Get top prediction
            top1_index = np.argmax(probs)
            confidence = probs[top1_index]
            predicted_class = CLASS_NAMES[top1_index]
            
            return {
                "success": True,
                "prediction": predicted_class,
                "confidence": f"{confidence:.2%}"
            }
        else:
            return {"success": False, "error": "No hand was detected in the image."}

    except Exception as e:
        print(f"Prediction error: {e}")
        return {"success": False, "error": "An error occurred during prediction."}

# --- API Endpoints (No changes needed) ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file:
        image_bytes = file.read()
        result = predict_from_image(image_bytes)
        return jsonify(result)
        
# --- Run the App (No changes needed) ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)