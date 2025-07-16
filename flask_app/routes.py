from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user
import os
import uuid
from werkzeug.utils import secure_filename
import sys
import json
from datetime import datetime

# Add parent directory to path to import anomaly_detector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anomaly_detector import CivicAnomalyDetector

# Create blueprint
main = Blueprint('main', __name__)

# Initialize detector
detector = CivicAnomalyDetector()
try:
    detector.load_model("civic_model.pkl")
    model_loaded = True
except:
    model_loaded = False

# History storage (will be replaced with database)
detection_history = []

def allowed_file(filename):
    """Check if file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@main.route('/')
def index():
    """Home page"""
    return render_template('index.html', model_loaded=model_loaded)

@main.route('/detect', methods=['GET', 'POST'])
def detect():
    """Anomaly detection page"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
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
                    
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'image_name': filename,
                        'image_path': unique_filename,
                        'prediction': result['anomaly_type'],
                        'confidence': result['confidence'],
                        'is_anomaly': result['is_anomaly'],
                        'probabilities': {
                            'plain': float(result['probabilities']['plain']),
                            'pothole': float(result['probabilities']['pothole'])
                        }
                    }
                    detection_history.append(history_entry)
                    
                    return render_template('result.html', 
                                          result=result, 
                                          image_path=url_for('static', filename=f'uploads/{unique_filename}'),
                                          filename=filename)
                except Exception as e:
                    flash(f'Error analyzing image: {str(e)}', 'danger')
                    return redirect(request.url)
            else:
                flash('Model not loaded. Please train the model first.', 'warning')
                return redirect(request.url)
        else:
            flash('File type not allowed. Please upload an image (PNG, JPG, JPEG, BMP).', 'danger')
            return redirect(request.url)
    
    return render_template('detect.html')

@main.route('/analytics')
@login_required
def analytics():
    """Analytics dashboard"""
    if not detection_history:
        flash('No detection history available.', 'info')
    
    return render_template('analytics.html', history=detection_history)

@main.route('/history')
@login_required
def history():
    """Detection history page"""
    return render_template('history.html', history=detection_history)

@main.route('/about')
def about():
    """About page"""
    return render_template('about.html')