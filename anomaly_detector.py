import cv2
import numpy as np
import os

class CivicAnomalyDetector:
    def __init__(self):
        self.anomaly_threshold = 0.6  # Adjustable threshold
    
    def detect_anomalies(self, image_path):
        """Main function to detect anomalies in civic images"""
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Get all detection scores
        edge_score = self._detect_unusual_edges(image)
        color_score = self._detect_color_anomalies(image)
        contour_score = self._detect_irregular_objects(image)
        
        # Calculate overall anomaly score
        overall_score = (edge_score + color_score + contour_score) / 3
        
        is_anomaly = overall_score > self.anomaly_threshold
        
        return {
            "image_path": image_path,
            "is_anomaly": is_anomaly,
            "overall_score": round(overall_score, 2),
            "details": {
                "edge_anomaly": round(edge_score, 2),
                "color_anomaly": round(color_score, 2),
                "object_anomaly": round(contour_score, 2)
            }
        }
    
    def _detect_unusual_edges(self, image):
        """Detect broken infrastructure using edge analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize to 0-1 scale (higher = more anomalous)
        return min(edge_density * 10, 1.0)
    
    def _detect_color_anomalies(self, image):
        """Detect unusual colors (dumping, graffiti)"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define normal civic colors (grays, browns, greens)
        normal_ranges = [
            ([0, 0, 50], [180, 30, 200]),    # Grays
            ([10, 50, 50], [25, 255, 200]),  # Browns/concrete
            ([35, 40, 40], [85, 255, 200])   # Greens (vegetation)
        ]
        
        normal_pixels = 0
        total_pixels = image.shape[0] * image.shape[1]
        
        for lower, upper in normal_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            normal_pixels += np.sum(mask > 0)
        
        # Calculate anomaly score (more unusual colors = higher score)
        anomaly_ratio = 1 - (normal_pixels / total_pixels)
        return min(anomaly_ratio * 2, 1.0)
    
    def _detect_irregular_objects(self, image):
        """Detect irregular objects using contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to find objects
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        irregular_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                # Calculate irregularity (perimeter^2 / area ratio)
                perimeter = cv2.arcLength(contour, True)
                if area > 0:
                    irregularity = (perimeter * perimeter) / (4 * np.pi * area)
                    if irregularity > 2:  # Irregular shapes
                        irregular_score += 0.1
        
        return min(irregular_score, 1.0)

def test_detector():
    """Test function to demonstrate the detector"""
    detector = CivicAnomalyDetector()
    
    # Test with sample images (you'll need to add actual images)
    test_images = ["test_image.jpg"]  # Add your test images here
    
    for image_path in test_images:
        if os.path.exists(image_path):
            result = detector.detect_anomalies(image_path)
            print(f"\n--- Analysis for {image_path} ---")
            print(f"Anomaly Detected: {result['is_anomaly']}")
            print(f"Overall Score: {result['overall_score']}")
            print(f"Edge Score: {result['details']['edge_anomaly']}")
            print(f"Color Score: {result['details']['color_anomaly']}")
            print(f"Object Score: {result['details']['object_anomaly']}")
        else:
            print(f"Image {image_path} not found")

if __name__ == "__main__":
    test_detector()