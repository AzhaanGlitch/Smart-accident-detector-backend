from flask import Flask, request, jsonify
from flask_cors import CORS  # For cross-origin requests
import tensorflow as tf
import numpy as np
from PIL import Image
import io  # For handling file streams
import json
import os
from datetime import datetime
from geopy.geocoders import Nominatim  # If not installed, pip install geopy

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global config from your testing.py
img_size = (224, 224)
class_names = ['accident', 'non_accident']
FINAL_MODEL_PATH = r"D:\Smart Accident Detector Backend\backend\machine_learning\accident_detection_model.h5"
BEST_MODEL_PATH = r"D:\Smart Accident Detector Backend\backend\machine_learning\best_model.h5"
RESULTS_LOG_PATH = 'prediction_results.json'

# Load models once at startup
model_final = tf.keras.models.load_model(FINAL_MODEL_PATH)
model_best = tf.keras.models.load_model(BEST_MODEL_PATH)

# Load results history
results_history = []
if os.path.exists(RESULTS_LOG_PATH):
    with open(RESULTS_LOG_PATH, 'r') as f:
        results_history = json.load(f)

def save_results_history():
    with open(RESULTS_LOG_PATH, 'w') as f:
        json.dump(results_history, f, indent=2)

def preprocess_image(image_file):
    """Preprocess uploaded image file"""
    img = Image.open(image_file).resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_accident(img_array, model):
    """Run prediction on a single model"""
    prediction = model.predict(img_array, verbose=0)[0][0]
    class_idx = 0 if prediction < 0.5 else 1
    confidence = (1 - prediction) if class_idx == 0 else prediction
    is_accident = (class_idx == 0)
    return class_names[class_idx], confidence, is_accident

def get_location():
    """Placeholder location (as in your code)"""
    try:
        geolocator = Nominatim(user_agent="accident_detector")
        location = geolocator.geocode("New York, NY, USA")
        return location.address if location else "Unknown location"
    except:
        return "Location unavailable"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Preprocess the uploaded file (handle as stream to avoid saving to disk)
        file.seek(0)  # Reset file pointer
        img_array = preprocess_image(file)
        
        # Run predictions
        pred_final, conf_final, is_acc_final = predict_accident(img_array, model_final)
        pred_best, conf_best, is_acc_best = predict_accident(img_array, model_best)
        
        accident_detected = is_acc_final or is_acc_best
        location = get_location() if accident_detected else None
        
        # Prepare response
        result = {
            'final_model': {
                'prediction': pred_final,
                'confidence': float(conf_final) * 100  # As percentage for frontend
            },
            'best_model': {
                'prediction': pred_best,
                'confidence': float(conf_best) * 100  # As percentage
            },
            'accident_detected': accident_detected,
            'location': location
        }
        
        # Log the result (as in your code)
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_filename": file.filename,
            **result  # Merge result dict
        }
        results_history.append(result_entry)
        save_results_history()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Run locally for testing