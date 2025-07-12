from anomaly_detector import CivicAnomalyDetector

detector = CivicAnomalyDetector()
detector.load_model("civic_model.pkl")

# Test your custom image
result = detector.predict("images/normal/normal_road2.jpg")

print(f"Image: {result['image_path']}")
print(f"Prediction: {result['anomaly_type'].upper()}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Normal: {result['probabilities']['plain']:.3f}")
print(f"Pothole: {result['probabilities']['pothole']:.3f}")
