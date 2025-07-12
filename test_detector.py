from anomaly_detector import CivicAnomalyDetector
import cv2
import numpy as np

def create_sample_images():
    """Create sample test images for demonstration"""
    
    # Create a "normal" civic image (simple road-like pattern)
    normal_image = np.ones((300, 400, 3), dtype=np.uint8) * 128  # Gray background
    cv2.rectangle(normal_image, (50, 100), (350, 200), (100, 100, 100), -1)  # Road
    cv2.line(normal_image, (200, 100), (200, 200), (255, 255, 255), 2)  # Lane marking
    cv2.imwrite("normal_civic.jpg", normal_image)
    
    # Create an "anomalous" image (with irregular objects and colors)
    anomaly_image = normal_image.copy()
    cv2.circle(anomaly_image, (150, 150), 30, (0, 0, 255), -1)  # Red object (dumping)
    cv2.rectangle(anomaly_image, (250, 120), (300, 180), (255, 0, 255), -1)  # Unusual color
    # Add some irregular edges
    pts = np.array([[100, 50], [120, 80], [140, 45], [160, 75]], np.int32)
    cv2.polylines(anomaly_image, [pts], True, (0, 255, 0), 3)
    cv2.imwrite("anomaly_civic.jpg", anomaly_image)
    
    print("Sample images created: normal_civic.jpg, anomaly_civic.jpg")

def run_tests():
    """Run the anomaly detector on test images"""
    detector = CivicAnomalyDetector()
    
    # Create sample images first
    create_sample_images()
    
    # Test images
    test_images = ["normal_civic.jpg", "anomaly_civic.jpg"]
    
    print("=== Civic Anomaly Detection Test ===\n")
    
    for image_path in test_images:
        result = detector.detect_anomalies(image_path)
        
        print(f"📸 Image: {image_path}")
        print(f"🚨 Anomaly Detected: {'YES' if result['is_anomaly'] else 'NO'}")
        print(f"📊 Overall Score: {result['overall_score']}/1.0")
        print(f"   - Edge Anomaly: {result['details']['edge_anomaly']}")
        print(f"   - Color Anomaly: {result['details']['color_anomaly']}")
        print(f"   - Object Anomaly: {result['details']['object_anomaly']}")
        print("-" * 50)

if __name__ == "__main__":
    run_tests()