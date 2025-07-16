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
    
    def __repr__(self):
        return f'<User {self.email}>'

@login_manager.user_loader
def load_user(user_id):
    """User loader function for Flask-Login"""
    return User.query.get(int(user_id))

class Detection(db.Model):
    """Detection history model"""
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255))
    image_path = db.Column(db.String(255))
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    is_anomaly = db.Column(db.Boolean)
    probabilities = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    
    def __repr__(self):
        return f'<Detection {self.id}: {self.prediction}>'