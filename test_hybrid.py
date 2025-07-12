from hybrid_detector import HybridCivicDetector
import os

# Create hybrid detector
detector = HybridCivicDetector()

# Train or load model
if os.path.exists("civic_model.pkl"):
    print("Loading existing ML model...")
    detector.load_model("civic_model.pkl")
else:
    print("Training ML model on normal images...")
    normal_images = ["images/normal/normal_road1.jpg", "images/normal/normal_road2.jpg"]
    detector.train_on_normal_images(normal_images)
    detector.save_model("civic_model.pkl")

# Test normal road (should be normal)
print("\n=== Testing Normal Road ===")
result = detector.detect_anomalies("images/normal/normal_civic.jpg")
print(f"Anomaly: {'YES' if result['is_anomaly'] else 'NO'}")
print(f"Type: {result['anomaly_type']}")
print(f"Method: {result.get('method', 'unknown')}")
print(f"Score: {result['anomaly_score']:.3f}")

# Test synthetic anomaly (should detect anomaly)
print("\n=== Testing Synthetic Cracked Road ===")
result = detector.detect_anomalies("images/anomaly/cracked_image1.jpg")
print(f"Anomaly: {'YES' if result['is_anomaly'] else 'NO'}")
print(f"Type: {result['anomaly_type']}")
print(f"Method: {result.get('method', 'unknown')}")
print(f"Score: {result['anomaly_score']:.3f}")

print("\n✅ Hybrid detector combines:")
print("   - ML learns what normal roads look like")
print("   - Rules detect actual cracks, potholes, dumping")
print("   - Best of both worlds!")