from advanced_detector import AdvancedCivicDetector
from typing import Dict

class HybridCivicDetector(AdvancedCivicDetector):
    """
    Hybrid detector combining ML (for normal patterns) + Rules (for anomalies)
    Solves the issue where ML-only misses real anomalies
    """
    
    def detect_anomalies(self, image_path: str) -> Dict:
        """Hybrid detection: ML + Rule-based for better accuracy"""
        if not self.is_trained:
            return self._rule_based_detection(image_path)
        
        # Get both ML and rule-based results
        ml_result = self._ml_detection(image_path)
        rule_result = self._rule_based_detection(image_path)
        
        # Hybrid decision logic - MUCH STRICTER thresholds
        if rule_result['is_anomaly'] and any([
            rule_result['details']['crack_detection'] > 0.9,  # Very obvious cracks only
            rule_result['details']['pothole_detection'] > 0.8,  # Clear potholes only
            rule_result['details']['dumping_detection'] > 0.7,   # Obvious dumping only
            rule_result['details']['damage_detection'] > 0.8     # Significant damage only
        ]):
            # Strong rule-based evidence of real anomaly
            rule_result['method'] = 'rule_based'
            return rule_result
        elif not ml_result['is_anomaly']:
            # ML says it's similar to normal training data
            return {
                'image_path': image_path,
                'is_anomaly': False,
                'anomaly_score': 0.1,
                'anomaly_type': 'normal',
                'confidence': 0.9,
                'method': 'ml_normal',
                'details': rule_result['details']
            }
        else:
            # ML detected anomaly but rules didn't find strong evidence
            ml_result['method'] = 'ml_anomaly'
            ml_result['details'] = rule_result['details']
            return ml_result
    
    def _ml_detection(self, image_path: str) -> Dict:
        """Pure ML detection using trained model"""
        image = self.preprocess_image(image_path)
        features = self.extract_comprehensive_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        anomaly_score = self.anomaly_model.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_model.predict(features_scaled)[0] == -1
        anomaly_type = self._classify_anomaly_type(image, features)
        
        return {
            'image_path': image_path,
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'anomaly_type': anomaly_type if is_anomaly else 'normal',
            'confidence': abs(anomaly_score)
        }

def test_hybrid_detector():
    """Test hybrid detector with both normal and anomaly images"""
    detector = HybridCivicDetector()
    
    # Load or train model
    try:
        detector.load_model("civic_model.pkl")
        print("Loaded existing ML model")
    except:
        print("Training new ML model...")
        normal_images = ["images/normal_road1.jpg", "images/normal_road2.jpg"]
        detector.train_on_normal_images(normal_images)
        detector.save_model("civic_model.pkl")
    
    # Test images
    test_cases = [
        ("images/normal_road2.jpg", "Normal Road"),
        ("cracked_road.jpg", "Cracked Road"),
        ("pothole_image.jpg", "Pothole"),
        ("dumping_site.jpg", "Dumping Site")
    ]
    
    print("\n=== Hybrid Detection Results ===")
    
    for image_path, description in test_cases:
        try:
            result = detector.detect_anomalies(image_path)
            
            print(f"\n📸 {description}: {image_path}")
            print(f"🚨 Anomaly: {'YES' if result['is_anomaly'] else 'NO'}")
            print(f"🏷️  Type: {result['anomaly_type']}")
            print(f"📊 Score: {result['anomaly_score']:.3f}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
            print(f"🔧 Method: {result.get('method', 'unknown')}")
            
            if 'details' in result:
                details = result['details']
                print(f"   - Crack: {details['crack_detection']:.3f}")
                print(f"   - Pothole: {details['pothole_detection']:.3f}")
                print(f"   - Dumping: {details['dumping_detection']:.3f}")
                print(f"   - Damage: {details['damage_detection']:.3f}")
            
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    test_hybrid_detector()