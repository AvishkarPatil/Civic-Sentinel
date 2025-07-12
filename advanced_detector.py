import cv2
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import List, Dict
import json

class AdvancedCivicDetector:
    """
    Advanced civic anomaly detector with ML models and comprehensive feature extraction
    """
    
    def __init__(self):
        self.anomaly_model = IsolationForest(
            contamination=0.15,
            random_state=42,
            n_estimators=200
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.anomaly_threshold = 0.2  # Lower threshold for better detection
        
        self.anomaly_types = {
            'pothole': 0,
            'cracked_pavement': 1,
            'graffiti': 2,
            'debris': 3,
            'broken_streetlight': 4,
            'damaged_sign': 5,
            'other': 6
        }
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Enhanced image preprocessing"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        image = cv2.resize(image, (224, 224))
        
        # Enhance contrast
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def extract_comprehensive_features(self, image: np.ndarray) -> np.ndarray:
        """Extract comprehensive features for high accuracy"""
        features = []
        
        # Convert to different color spaces
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # 1. Edge Features (Multiple methods)
        edges_canny = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges_canny) / edges_canny.size
        features.append(edge_density)
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features.append(np.mean(sobel_magnitude))
        features.append(np.std(sobel_magnitude))
        
        # 2. Texture Features (Local Binary Patterns)
        lbp = self._local_binary_pattern(gray)
        lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 32))
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        features.extend(lbp_hist[:16])  # Top 16 bins
        
        # 3. Color Features (Enhanced)
        for channel in range(3):
            # Color histograms
            hist, _ = np.histogram(image[:, :, channel], bins=32, range=(0, 1))
            hist = hist / np.sum(hist)
            features.extend(hist[:8])  # Top 8 bins per channel
            
            # Color statistics
            features.append(np.mean(image[:, :, channel]))
            features.append(np.std(image[:, :, channel]))
            features.append(np.median(image[:, :, channel]))
        
        # 4. HSV Features
        for channel in range(3):
            features.append(np.mean(hsv[:, :, channel]))
            features.append(np.std(hsv[:, :, channel]))
        
        # 5. Contour Features
        contours, _ = cv2.findContours(edges_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        features.append(contour_count)
        
        # Contour properties
        if contours:
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
            if areas:
                features.append(np.mean(areas))
                features.append(np.std(areas))
                features.append(np.max(areas))
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])
        
        # 6. Shape Features (Hough transforms)
        # Lines (cracks, edges)
        lines = cv2.HoughLinesP(edges_canny, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        line_count = len(lines) if lines is not None else 0
        features.append(line_count)
        
        # Circles (potholes, signs)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        circle_count = len(circles[0]) if circles is not None else 0
        features.append(circle_count)
        
        # 7. Frequency Domain Features
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        features.append(np.mean(magnitude_spectrum))
        features.append(np.std(magnitude_spectrum))
        
        # 8. Gradient Features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        features.append(np.mean(gradient_direction))
        features.append(np.std(gradient_direction))
        
        # 9. Structural Features
        # Variance of Laplacian (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        return np.array(features)
    
    def _local_binary_pattern(self, image, radius=3, n_points=24):
        """Enhanced Local Binary Pattern implementation"""
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                code = 0
                for k in range(min(n_points, 8)):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                        if image[x, y] >= center:
                            code |= (1 << k)
                lbp[i, j] = code
        return lbp
    
    def train_on_normal_images(self, normal_image_paths: List[str]):
        """Train the model on normal civic images"""
        print(f"Training on {len(normal_image_paths)} normal images...")
        
        features_list = []
        for i, image_path in enumerate(normal_image_paths):
            try:
                image = self.preprocess_image(image_path)
                features = self.extract_comprehensive_features(image)
                features_list.append(features)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(normal_image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid images found for training")
        
        # Convert to numpy array
        X = np.array(features_list)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detection model
        self.anomaly_model.fit(X_scaled)
        self.is_trained = True
        
        print("Training completed successfully!")
    
    def detect_anomalies(self, image_path: str) -> Dict:
        """Detect anomalies with high accuracy"""
        if not self.is_trained:
            # Use rule-based detection if not trained
            return self._rule_based_detection(image_path)
        
        # Preprocess image
        image = self.preprocess_image(image_path)
        
        # Extract features
        features = self.extract_comprehensive_features(image)
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict anomaly
        anomaly_score = self.anomaly_model.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_model.predict(features_scaled)[0] == -1
        
        # Classify anomaly type
        anomaly_type = self._classify_anomaly_type(image, features)
        
        # Calculate confidence
        confidence = abs(anomaly_score)
        
        return {
            'image_path': image_path,
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'anomaly_type': anomaly_type,
            'confidence': float(confidence),
            'details': self._get_detailed_analysis(image, features)
        }
    
    def _rule_based_detection(self, image_path: str) -> Dict:
        """Fallback rule-based detection when model is not trained"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Enhanced rule-based detection
        crack_score = self._detect_cracks_advanced(image)
        pothole_score = self._detect_potholes_advanced(image)
        dumping_score = self._detect_dumping_advanced(image)
        damage_score = self._detect_damage_advanced(image)
        
        overall_score = (crack_score * 0.3 + pothole_score * 0.25 + 
                        dumping_score * 0.25 + damage_score * 0.2)
        
        is_anomaly = overall_score > self.anomaly_threshold
        
        # Determine anomaly type
        scores = {
            'cracked_pavement': crack_score,
            'pothole': pothole_score,
            'debris': dumping_score,
            'other': damage_score
        }
        anomaly_type = max(scores, key=scores.get) if is_anomaly else 'normal'
        
        # Override if specific thresholds are met
        if crack_score > 0.5:
            anomaly_type = 'cracked_pavement'
            is_anomaly = True
        elif dumping_score > 0.15:
            anomaly_type = 'debris'
            is_anomaly = True
        elif damage_score > 0.2:
            anomaly_type = 'other'
            is_anomaly = True
        
        return {
            'image_path': image_path,
            'is_anomaly': is_anomaly,
            'anomaly_score': float(overall_score),
            'anomaly_type': anomaly_type,
            'confidence': float(overall_score),
            'details': {
                'crack_detection': round(crack_score, 3),
                'pothole_detection': round(pothole_score, 3),
                'dumping_detection': round(dumping_score, 3),
                'damage_detection': round(damage_score, 3)
            }
        }
    
    def _detect_cracks_advanced(self, image):
        """Advanced crack detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection methods
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        
        # Combine edges
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Morphological operations to connect broken lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines
        lines = cv2.HoughLinesP(closed, 1, np.pi/180, threshold=30, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is not None:
            crack_score = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 15:  # Significant crack
                    crack_score += 0.1
            return min(crack_score, 1.0)
        return 0.0
    
    def _detect_potholes_advanced(self, image):
        """Advanced pothole detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple circle detection with different parameters
        circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=30, param2=15, minRadius=5, maxRadius=50)
        circles2 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 30,
                                   param1=25, param2=20, minRadius=8, maxRadius=60)
        
        pothole_score = 0
        for circles in [circles1, circles2]:
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Check if it's a dark area (pothole characteristic)
                    roi = gray[max(0, y-r):min(gray.shape[0], y+r), 
                              max(0, x-r):min(gray.shape[1], x+r)]
                    if roi.size > 0:
                        avg_intensity = np.mean(roi)
                        if avg_intensity < 100:  # Increased threshold for better detection
                            pothole_score += 0.3
        
        # Also check for dark circular areas using contours
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Reasonable pothole size
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.3:  # Somewhat circular
                        pothole_score += 0.2
        
        return min(pothole_score, 1.0)
    
    def _detect_dumping_advanced(self, image):
        """Advanced dumping detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Multiple unusual color ranges
        unusual_colors = [
            ([0, 100, 100], [10, 255, 255]),    # Bright reds
            ([110, 50, 50], [130, 255, 255]),   # Blues
            ([20, 100, 100], [30, 255, 255]),   # Bright yellows
            ([140, 50, 50], [160, 255, 255])    # Purples
        ]
        
        unusual_pixels = 0
        total_pixels = image.shape[0] * image.shape[1]
        
        for lower, upper in unusual_colors:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            unusual_pixels += np.sum(mask > 0)
        
        color_score = unusual_pixels / total_pixels
        
        # Texture analysis for irregular objects
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000
        
        return min(color_score * 3 + texture_score, 1.0)
    
    def _detect_damage_advanced(self, image):
        """Advanced structural damage detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multiple edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        damage_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Irregularity measure
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.2:  # Very irregular
                        damage_score += 0.15
        
        return min(damage_score, 1.0)
    
    def _classify_anomaly_type(self, image: np.ndarray, features: np.ndarray) -> str:
        """Advanced anomaly type classification"""
        # Use feature analysis for classification
        edge_density = features[0]
        line_count = features[-4] if len(features) > 4 else 0
        circle_count = features[-3] if len(features) > 3 else 0
        color_variance = np.mean(features[25:31]) if len(features) > 31 else 0
        
        # Classification rules
        if line_count > 5 and edge_density > 0.1:
            return 'cracked_pavement'
        elif circle_count > 0 and edge_density < 0.05:
            return 'pothole'
        elif color_variance > 0.3:
            return 'graffiti'
        elif edge_density > 0.15:
            return 'debris'
        elif edge_density < 0.02:
            return 'broken_streetlight'
        else:
            return 'other'
    
    def _get_detailed_analysis(self, image: np.ndarray, features: np.ndarray) -> Dict:
        """Get detailed analysis of the image"""
        return {
            'edge_density': float(features[0]) if len(features) > 0 else 0,
            'texture_complexity': float(np.mean(features[2:18])) if len(features) > 18 else 0,
            'color_variance': float(np.mean(features[18:42])) if len(features) > 42 else 0,
            'structural_irregularity': float(features[-1]) if len(features) > 0 else 0
        }
    
    def batch_detect(self, image_folder: str) -> List[Dict]:
        """Process multiple images"""
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_paths = []
        for file in os.listdir(image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(image_folder, file))
        
        print(f"Processing {len(image_paths)} images...")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_anomalies(image_path)
                results.append(result)
                
                if (i + 1) % 5 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return results
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
            
        import joblib
        model_data = {
            'scaler': self.scaler,
            'anomaly_model': self.anomaly_model,
            'anomaly_types': self.anomaly_types,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        import joblib
        model_data = joblib.load(model_path)
        
        self.scaler = model_data['scaler']
        self.anomaly_model = model_data['anomaly_model']
        self.anomaly_types = model_data['anomaly_types']
        self.is_trained = model_data.get('is_trained', True)
        
        print(f"Model loaded from {model_path}")

def test_advanced_detector():
    """Test the advanced detector"""
    detector = AdvancedCivicDetector()
    
    # Create test images
    create_test_images()
    
    test_images = ["normal_street.jpg", "cracked_road.jpg", "pothole_image.jpg", "dumping_site.jpg"]
    
    print("=== Advanced Civic Anomaly Detection ===\n")
    
    for image_path in test_images:
        if os.path.exists(image_path):
            result = detector.detect_anomalies(image_path)
            
            print(f"📸 Image: {image_path}")
            print(f"🚨 Anomaly: {'YES' if result['is_anomaly'] else 'NO'}")
            print(f"🏷️  Type: {result['anomaly_type']}")
            print(f"📊 Score: {result['anomaly_score']:.3f}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            if 'details' in result and isinstance(result['details'], dict):
                for key, value in result['details'].items():
                    print(f"   - {key}: {value:.3f}")
            print("-" * 50)

def create_test_images():
    """Create comprehensive test images"""
    # Normal street
    normal = np.ones((400, 600, 3), dtype=np.uint8) * 120
    cv2.rectangle(normal, (0, 150), (600, 250), (100, 100, 100), -1)
    cv2.line(normal, (300, 150), (300, 250), (255, 255, 255), 3)
    cv2.imwrite("normal_street.jpg", normal)
    
    # Cracked road with multiple cracks
    cracked = normal.copy()
    cv2.line(cracked, (100, 180), (200, 220), (50, 50, 50), 3)
    cv2.line(cracked, (150, 160), (180, 240), (40, 40, 40), 2)
    cv2.line(cracked, (250, 170), (320, 200), (45, 45, 45), 2)
    cv2.imwrite("cracked_road.jpg", cracked)
    
    # Pothole
    pothole = normal.copy()
    cv2.circle(pothole, (200, 200), 25, (30, 30, 30), -1)
    cv2.circle(pothole, (400, 180), 20, (35, 35, 35), -1)
    cv2.imwrite("pothole_image.jpg", pothole)
    
    # Dumping site with multiple objects
    dumping = normal.copy()
    cv2.circle(dumping, (200, 200), 40, (0, 0, 255), -1)  # Red trash
    cv2.rectangle(dumping, (350, 180), (400, 220), (0, 255, 255), -1)  # Yellow object
    cv2.circle(dumping, (150, 180), 30, (255, 0, 255), -1)  # Purple object
    cv2.imwrite("dumping_site.jpg", dumping)

if __name__ == "__main__":
    test_advanced_detector()