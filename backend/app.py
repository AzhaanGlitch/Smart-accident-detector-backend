from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"],
        "supports_credentials": False
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global config
img_size = (224, 224)
class_names = ['accident', 'non_accident']
RESULTS_LOG_PATH = 'prediction_results.json'

# Initialize models as None - will load them when needed
model_final = None
model_best = None
tf = None
np = None
Image = None

def lazy_load_dependencies():
    """Load heavy dependencies only when needed"""
    global tf, np, Image, model_final, model_best
    
    if tf is None:
        try:
            import tensorflow as tf
            import numpy as np
            from PIL import Image
            logger.info("Heavy dependencies loaded successfully")
            
            # Try to load models
            load_models()
            
        except Exception as e:
            logger.error(f"Error loading dependencies: {e}")
            return False
    return True

def load_models():
    """Load trained models"""
    global model_final, model_best
    
    try:
        # Check different possible locations for model files
        possible_paths = [
            'accident_detection_model.h5',
            'best_model.h5',
            os.path.join('backend', 'machine_learning', 'accident_detection_model.h5'),
            os.path.join('backend', 'machine_learning', 'best_model.h5'),
            os.path.join('models', 'accident_detection_model.h5'),
            os.path.join('models', 'best_model.h5')
        ]
        
        final_model_path = None
        best_model_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                if 'accident_detection_model.h5' in path:
                    final_model_path = path
                elif 'best_model.h5' in path:
                    best_model_path = path
        
        if final_model_path:
            model_final = tf.keras.models.load_model(final_model_path)
            logger.info(f"Final model loaded from {final_model_path}")
        else:
            logger.warning("Final model not found")
            
        if best_model_path:
            model_best = tf.keras.models.load_model(best_model_path)
            logger.info(f"Best model loaded from {best_model_path}")
        else:
            logger.warning("Best model not found")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# Load results history
results_history = []

def load_results_history():
    global results_history
    if os.path.exists(RESULTS_LOG_PATH):
        try:
            with open(RESULTS_LOG_PATH, 'r') as f:
                results_history = json.load(f)
        except Exception as e:
            logger.error(f"Error loading results history: {e}")
            results_history = []

def save_results_history():
    try:
        with open(RESULTS_LOG_PATH, 'w') as f:
            json.dump(results_history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving results history: {e}")

def preprocess_image(image_file):
    """Preprocess uploaded image file"""
    if not lazy_load_dependencies():
        raise Exception("Failed to load required dependencies")
        
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
    
    try:
        prediction = model.predict(img_array, verbose=0)[0][0]
        class_idx = 0 if prediction < 0.5 else 1
        confidence = (1 - prediction) if class_idx == 0 else prediction
        is_accident = (class_idx == 0)
        return class_names[class_idx], confidence, is_accident
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "prediction_error", 0.0, False

def get_location():
    """Get location information"""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="accident_detector")
        location = geolocator.geocode("New York, NY, USA")
        return location.address if location else "Unknown location"
    except:
        return "Location unavailable"

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint with system info"""
    
    # Check current working directory and list files
    current_dir = os.getcwd()
    files_in_root = []
    try:
        files_in_root = [f for f in os.listdir(current_dir) if not f.startswith('.')][:15]
    except Exception as e:
        files_in_root = [f"Error: {str(e)}"]
    
    # Check for model files
    model_files = []
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.h5'):
                model_files.append(os.path.join(root, file))
    
    # System info
    system_info = {
        'python_version': sys.version,
        'current_directory': current_dir,
        'files_in_root': files_in_root,
        'model_files_found': model_files,
        'environment_variables': {
            'PORT': os.environ.get('PORT', 'Not set'),
            'RENDER': os.environ.get('RENDER', 'Not set')
        }
    }
    
    # Try to get ML library versions
    ml_info = {}
    try:
        if lazy_load_dependencies():
            ml_info = {
                'tensorflow_available': tf is not None,
                'tensorflow_version': tf.__version__ if tf else 'Not loaded',
                'models_loaded': {
                    'final_model': model_final is not None,
                    'best_model': model_best is not None
                }
            }
    except Exception as e:
        ml_info = {'error': str(e)}
    
    return jsonify({
        'status': 'running',
        'message': 'Smart Accident Detector API is running',
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'ml_info': ml_info
    })

@app.route('/test', methods=['GET', 'OPTIONS'])
def simple_test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'API is working perfectly!',
        'timestamp': datetime.now().isoformat(),
        'method': request.method
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict accident from uploaded image"""
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Check if dependencies can be loaded
        if not lazy_load_dependencies():
            return jsonify({'error': 'ML dependencies not available'}), 500
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if at least one model is loaded
        if model_final is None and model_best is None:
            return jsonify({
                'error': 'No models available for prediction',
                'message': 'Models are not loaded. This might be due to missing model files or insufficient memory.'
            }), 500
        
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
                'confidence': float(conf_final) * 100 if isinstance(conf_final, (int, float)) else 0
            },
            'best_model': {
                'prediction': pred_best,
                'confidence': float(conf_best) * 100 if isinstance(conf_best, (int, float)) else 0
            },
            'accident_detected': accident_detected,
            'location': location,
            'timestamp': datetime.now().isoformat()
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
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    return jsonify({
        'service': 'Smart Accident Detector',
        'status': 'operational',
        'version': '1.0.0',
        'endpoints': {
            'health_check': '/',
            'simple_test': '/test',
            'predict': '/predict (POST)',
            'status': '/status'
        }
    })

# Load results history on startup
load_results_history()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)