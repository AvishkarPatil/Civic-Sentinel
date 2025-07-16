from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
import os

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure app
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-for-development')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///civic_sentinel.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize extensions with app
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # Register blueprints
    from .routes import main
    from .auth import auth
    from .api import api
    
    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(api, url_prefix='/api')
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app