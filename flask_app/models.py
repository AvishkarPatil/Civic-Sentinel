from . import db, login_manager
from flask_login import UserMixin
from datetime import datetime

class User(UserMixin, db.Model):
    """User model for authentication"""
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationship with detections
    detections = db.relationship('Detection', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.email}>'

@login_manager.user_loader
def load_user(user_id):
    """User loader function for Flask-Login"""
    return User.query.get(int(user_id))

class Detection(db.Model):
    """Detection history model"""
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_anomaly = db.Column(db.Boolean, nullable=False)
    prob_normal = db.Column(db.Float, nullable=False)
    prob_anomaly = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'image_name': self.image_name,
            'image_path': self.image_path,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'is_anomaly': self.is_anomaly,
            'probabilities': {
                'plain': self.prob_normal,
                'pothole': self.prob_anomaly
            }
        }
    
    def __repr__(self):
        return f'<Detection {self.id}: {self.prediction}>'