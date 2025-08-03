#!/usr/bin/env python3
"""Test script to debug detection issues"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from anomaly_detector import CivicAnomalyDetector
    print("✅ Successfully imported CivicAnomalyDetector")
    
    # Test model loading
    detector = CivicAnomalyDetector()
    try:
        detector.load_model("civic_model.pkl")
        print("✅ Model loaded successfully")
        
        # Test with a sample image if available
        test_images = [
            "images/normal/normal_civic.jpg",
            "dataset/test/Plain/1.jpg",
            "dataset/train/Plain/a 0.jpg"
        ]
        
        for img_path in test_images:
            if os.path.exists(img_path):
                print(f"🔍 Testing with: {img_path}")
                try:
                    result = detector.predict(img_path)
                    print(f"✅ Prediction successful: {result['anomaly_type']} ({result['confidence']:.2f})")
                    break
                except Exception as e:
                    print(f"❌ Prediction failed: {e}")
            else:
                print(f"⚠️  Image not found: {img_path}")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")