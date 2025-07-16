from flask import Blueprint, request, jsonify, current_app
import os
import uuid
from werkzeug.utils import secure_filename
import sys
from datetime import datetime

# Add parent directory to path to import anomaly_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly_detector import CivicAnomalyDetector

# Create blueprint
api = Blueprint('api', __name__)

# Initialize detector
detector = CivicAnomalyDetector()
try:
    detector.load_model("civic_model.pkl")
    model_loaded = True
except:
    model_loaded = False

def allowed_file(filename):
    """Check if file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api.route('/status', methods=['GET'])
def status():
    """API status endpoint"""
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'version': '1.0.0'
    })

@api.route('/detect', methods=['POST'])
def detect():
    """API endpoint for anomaly detection"""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Analyze image
        if model_loaded:
            try:
                result = detector.predict(file_path)
                
                # Format response
                response = {
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename,
                    'is_anomaly': result['is_anomaly'],
                    'anomaly_type': result['anomaly_type'],
                    'confidence': float(result['confidence']),
                    'probabilities': {
                        'normal': float(result['probabilities']['plain']),
                        'pothole': float(result['probabilities']['pothole'])
                    }
                }
                
                return jsonify(response)
            except Exception as e:
                return jsonify({'error': f'Error analyzing image: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Model not loaded'}), 503
    else:
        return jsonify({'error': 'File type not allowed'}), 400