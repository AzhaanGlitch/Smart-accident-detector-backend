from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime
from geopy.geocoders import Nominatim

app = Flask(__name__)
CORS(app)

# Global config
img_size = (224, 224)
class_names = ['accident', 'non_accident']

# Use relative paths that work on deployment
FINAL_MODEL_PATH = os.path.join('machine_learning', 'accident_detection_model.h5')
BEST_MODEL_PATH = os.path.join('machine_learning', 'best_model.h5')
RESULTS_LOG_PATH = 'prediction_results.json'

# Initialize models as None - DON'T load them on startup
model_final = None
model_best = None

# Load results history
results_history = []
if os.path.exists(RESULTS_LOG_PATH):
    try:
        with open(RESULTS_LOG_PATH, 'r') as f:
            results_history = json.load(f)
    except Exception as e:
        print(f"Error loading results history: {e}")
        results_history = []

def load_model_if_needed(model_type='final'):
    """Load model only when needed (lazy loading)"""
    global model_final, model_best
    
    if model_type == 'final' and model_final is None:
        try:
            if os.path.exists(FINAL_MODEL_PATH):
                print("Loading final model...")
                model_final = tf.keras.models.load_model(FINAL_MODEL_PATH)
                print("Final model loaded successfully!")
            else:
                print(f"Final model not found at {FINAL_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading final model: {e}")
    
    elif model_type == 'best' and model_best is None:
        try:
            if os.path.exists(BEST_MODEL_PATH):
                print("Loading best model...")
                model_best = tf.keras.models.load_model(BEST_MODEL_PATH)
                print("Best model loaded successfully!")
            else:
                print(f"Best model not found at {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading best model: {e}")

def save_results_history():
    try:
        with open(RESULTS_LOG_PATH, 'w') as f:
            json.dump(results_history, f, indent=2)
    except Exception as e:
        print(f"Error saving results history: {e}")

def preprocess_image(image_file):
    """Preprocess uploaded image file"""
    img = Image.open(image_file).resize(img_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_accident(img_array, model):
    """Run prediction on a single model"""
    if model is None:
        return "model_unavailable", 0.0, False
    
    prediction = model.predict(img_array, verbose=0)[0][0]
    class_idx = 0 if prediction < 0.5 else 1
    confidence = (1 - prediction) if class_idx == 0 else prediction
    is_accident = (class_idx == 0)
    return class_names[class_idx], confidence, is_accident

def get_location():
    """Get location information"""
    try:
        geolocator = Nominatim(user_agent="accident_detector")
        location = geolocator.geocode("New York, NY, USA")
        return location.address if location else "Unknown location"
    except:
        return "Location unavailable"

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Smart Accident Detector API is running',
        'models': {
            'final_model_path_exists': os.path.exists(FINAL_MODEL_PATH),
            'best_model_path_exists': os.path.exists(BEST_MODEL_PATH),
            'final_model_loaded': model_final is not None,
            'best_model_loaded': model_best is not None
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load models only when prediction is requested (lazy loading)
        load_model_if_needed('final')
        load_model_if_needed('best')
        
        # Check if at least one model is loaded
        if model_final is None and model_best is None:
            return jsonify({'error': 'No models available for prediction'}), 500
        
        # Preprocess the uploaded file
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
                'confidence': float(conf_final) * 100  # As percentage
            },
            'best_model': {
                'prediction': pred_best,
                'confidence': float(conf_best) * 100  # As percentage
            },
            'accident_detected': accident_detected,
            'location': location
        }
        
        # Log the result
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_filename": file.filename,
            **result
        }
        results_history.append(result_entry)
        save_results_history()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)