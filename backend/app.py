from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Disable TensorFlow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# Global config
img_size = (224, 224)
class_names = ['accident', 'non_accident']

# Use relative paths
FINAL_MODEL_PATH = os.path.join('machine_learning', 'accident_detection_model.h5')
BEST_MODEL_PATH = os.path.join('machine_learning', 'best_model.h5')

# Cache models globally (they'll reload on each serverless function call anyway)
model_final = None
model_best = None

def load_models():
    """Load models on each function execution"""
    global model_final, model_best
    
    if model_final is None:
        try:
            if os.path.exists(FINAL_MODEL_PATH):
                model_final = tf.keras.models.load_model(FINAL_MODEL_PATH, compile=False)
                print("Final model loaded")
        except Exception as e:
            print(f"Error loading final model: {e}")
    
    if model_best is None:
        try:
            if os.path.exists(BEST_MODEL_PATH):
                model_best = tf.keras.models.load_model(BEST_MODEL_PATH, compile=False)
                print("Best model loaded")
        except Exception as e:
            print(f"Error loading best model: {e}")

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

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Smart Accident Detector API is running on Vercel',
        'platform': 'Vercel Serverless'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Note: Vercel routes should start with /api/"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load models (they reload on each serverless function call)
        load_models()
        
        if model_final is None and model_best is None:
            return jsonify({'error': 'No models available for prediction'}), 500
        
        # Preprocess image
        file.seek(0)
        img_array = preprocess_image(file)
        
        # Run predictions
        pred_final, conf_final, is_acc_final = predict_accident(img_array, model_final)
        pred_best, conf_best, is_acc_best = predict_accident(img_array, model_best)
        
        accident_detected = is_acc_final or is_acc_best
        
        # Simplified response (removed location to reduce cold start time)
        result = {
            'final_model': {
                'prediction': pred_final,
                'confidence': float(conf_final) * 100
            },
            'best_model': {
                'prediction': pred_best,
                'confidence': float(conf_best) * 100
            },
            'accident_detected': accident_detected,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# For Vercel, we need to export the app
def handler(request):
    return app(request)

if __name__ == '__main__':
    app.run(debug=True)