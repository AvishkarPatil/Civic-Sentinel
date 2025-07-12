from anomaly_detector import CivicAnomalyDetector
import os

def test_model():
    """Test the trained model"""
    
    # Check if model exists
    if not os.path.exists("civic_model.pkl"):
        print("❌ No trained model found!")
        print("Please run 'python train.py' first to train the model.")
        return
    
    # Load trained model
    detector = CivicAnomalyDetector()
    detector.load_model("civic_model.pkl")
    
    print("=== Testing Civic Anomaly Detector ===\n")
    
    # Test images from test set
    test_folders = [
        ("dataset/test/Plain", "Normal Road"),
        ("dataset/test/Pothole", "Pothole")
    ]
    
    for folder_path, description in test_folders:
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if files:
                # Test first few images in folder
                for i, filename in enumerate(files[:3]):  # Test first 3 images
                    test_image = os.path.join(folder_path, filename)
                    result = detector.predict(test_image)
                    
                    print(f"📸 {description}: {filename}")
                    print(f"🚨 Prediction: {result['anomaly_type'].upper()}")
                    print(f"🎯 Confidence: {result['confidence']:.3f}")
                    print(f"📊 Probabilities:")
                    print(f"   - Normal: {result['probabilities']['plain']:.3f}")
                    print(f"   - Pothole: {result['probabilities']['pothole']:.3f}")
                    
                    # Check if prediction is correct
                    expected = "pothole" if "Pothole" in folder_path else "normal"
                    correct = "✅ CORRECT" if result['anomaly_type'] == expected else "❌ WRONG"
                    print(f"🎯 {correct}")
                    print("-" * 50)
            else:
                print(f"No images found in {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")

def test_custom_image(image_path):
    """Test a custom image"""
    
    if not os.path.exists("civic_model.pkl"):
        print("❌ No trained model found!")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model and predict
    detector = CivicAnomalyDetector()
    detector.load_model("civic_model.pkl")
    
    result = detector.predict(image_path)
    
    print(f"\n=== Custom Image Test ===")
    print(f"📸 Image: {image_path}")
    print(f"🚨 Prediction: {result['anomaly_type'].upper()}")
    print(f"🎯 Confidence: {result['confidence']:.3f}")
    print(f"📊 Probabilities:")
    print(f"   - Normal Road: {result['probabilities']['plain']:.3f}")
    print(f"   - Pothole: {result['probabilities']['pothole']:.3f}")

if __name__ == "__main__":
    # Test the model on test set
    test_model()
    
    # Test custom image if it exists
    custom_images = [
        "images/normal_road2.jpg",
        "your_test_image.jpg"
    ]
    
    for img_path in custom_images:
        if os.path.exists(img_path):
            test_custom_image(img_path)
            break