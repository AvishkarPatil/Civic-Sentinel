from anomaly_detector import CivicAnomalyDetector
import os

def main():
    """Train the civic anomaly detector"""
    
    # Check if dataset exists
    if not os.path.exists("dataset"):
        print("❌ Dataset folder not found!")
        print("Please create the following structure:")
        print("dataset/")
        print("├── train/")
        print("│   ├── Plain/     (normal road images)")
        print("│   └── Pothole/   (pothole images)")
        print("└── test/")
        print("    ├── Plain/     (test normal roads)")
        print("    └── Pothole/   (test potholes)")
        return
    
    # Create detector
    detector = CivicAnomalyDetector()
    
    print("=== Training Civic Anomaly Detector ===\n")
    
    try:
        # Train the model
        detector.train("dataset")
        
        # Save the trained model
        detector.save_model("civic_model.pkl")
        
        # Evaluate on test set
        print("\n=== Test Set Evaluation ===")
        detector.evaluate_test_set("dataset")
        
        print("\n✅ Training completed successfully!")
        print("📁 Model saved as 'civic_model.pkl'")
        print("🚀 Ready to use for predictions!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    main()