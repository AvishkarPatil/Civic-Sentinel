from advanced_detector import AdvancedCivicDetector

detector = AdvancedCivicDetector()
detector.load_model("civic_model.pkl")

result = detector.detect_anomalies("images/anomaly/cracked_image4.jpg")

print("=== ML-Based Detection ===")
print(f"Image: {result['image_path']}")
print(f"Anomaly: {'YES' if result['is_anomaly'] else 'NO'}")
print(f"Type: {result['anomaly_type']}")
print(f"Score: {result['anomaly_score']:.3f}")
print(f"Confidence: {result['confidence']:.3f}")
