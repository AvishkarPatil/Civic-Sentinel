#!/usr/bin/env python3
"""Update database schema"""

from flask_app import create_app, db

app = create_app()

with app.app_context():
    # Drop and recreate tables
    db.drop_all()
    db.create_all()
    print("✅ Database updated successfully!")