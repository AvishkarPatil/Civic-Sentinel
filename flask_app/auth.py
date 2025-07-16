from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from .models import User
from . import db

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        # Check if user exists
        user = User.query.filter_by(email=email).first()
        
        # Check password
        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.', 'danger')
            return redirect(url_for('auth.login'))
        
        # Log in user
        login_user(user, remember=remember)
        return redirect(url_for('main.index'))
    
    return render_template('login.html')

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page"""
    if request.method == 'POST':
        email = request.form.get('email')
        name = request.form.get('name')
        password = request.form.get('password')
        
        # Check if user already exists
        user = User.query.filter_by(email=email).first()
        
        if user:
            flash('Email address already exists', 'danger')
            return redirect(url_for('auth.signup'))
        
        # Create new user
        new_user = User(
            email=email,
            name=name,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        
        # Add user to database
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully!', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('signup.html')

@auth.route('/logout')
@login_required
def logout():
    """Log out user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))