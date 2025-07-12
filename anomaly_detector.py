import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import Dict, List, Tuple

class CivicAnomalyDetector:
    """
    Civic anomaly detector using dataset structure:
    dataset/train/Plain & dataset/train/Pothole
    dataset/test/Plain & dataset/test/Pothole
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.class_names = ['Plain', 'Pothole']
        
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize for consistency
        image = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        features = []
        
        # 1. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # 2. Texture features
        texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(texture_var / 1000)
        
        # 3. Color features
        for channel in range(3):
            features.append(np.mean(image[:, :, channel]))
            features.append(np.std(image[:, :, channel]))
        
        # 4. HSV features
        for channel in range(3):
            features.append(np.mean(hsv[:, :, channel]))
            features.append(np.std(hsv[:, :, channel]))
        
        # 5. Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.append(len(contours))
        
        # 6. Circle detection (potholes)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        circle_count = len(circles[0]) if circles is not None else 0
        features.append(circle_count)
        
        # 7. Dark area detection
        dark_pixels = np.sum(gray < 80)
        dark_ratio = dark_pixels / gray.size
        features.append(dark_ratio)
        
        # 8. Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        return np.array(features)
    
    def load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from folder structure"""
        X = []
        y = []
        
        for split in ['train', 'test']:
            split_path = os.path.join(dataset_path, split)
            if not os.path.exists(split_path):
                continue
                
            for class_idx, class_name in enumerate(self.class_names):
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                
                print(f"Loading {class_name} images from {split}...")
                
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(class_path, filename)
                        try:
                            features = self.extract_features(image_path)
                            X.append(features)
                            y.append(class_idx)
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")
        
        return np.array(X), np.array(y)
    
    def train(self, dataset_path: str):
        """Train the model"""
        print("Loading dataset...")
        X, y = self.load_dataset(dataset_path)
        
        if len(X) == 0:
            raise ValueError("No images found in dataset")
        
        print(f"Loaded {len(X)} images")
        print(f"Plain images: {np.sum(y == 0)}")
        print(f"Pothole images: {np.sum(y == 1)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        print("Training model...")
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate accuracy
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        print(f"Training accuracy: {accuracy:.3f}")
        
        print("Training completed!")
    
    def predict(self, image_path: str) -> Dict:
        """Predict anomaly for single image"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract features
        features = self.extract_features(image_path)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        is_anomaly = prediction == 1  # 1 = Pothole
        confidence = probabilities[prediction]
        
        return {
            'image_path': image_path,
            'is_anomaly': is_anomaly,
            'anomaly_type': 'pothole' if is_anomaly else 'normal',
            'confidence': float(confidence),
            'probabilities': {
                'plain': float(probabilities[0]),
                'pothole': float(probabilities[1])
            }
        }
    
    def evaluate_test_set(self, dataset_path: str):
        """Evaluate on test set"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        test_path = os.path.join(dataset_path, 'test')
        if not os.path.exists(test_path):
            print("No test set found")
            return
        
        X_test = []
        y_test = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_path = os.path.join(test_path, class_name)
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(class_path, filename)
                        try:
                            features = self.extract_features(image_path)
                            X_test.append(features)
                            y_test.append(class_idx)
                        except Exception as e:
                            print(f"Error processing {image_path}: {e}")
        
        if len(X_test) == 0:
            print("No test images found")
            return
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy: {accuracy:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
    
    def save_model(self, model_path: str):
        """Save trained model"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.class_names = model_data['class_names']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {model_path}")