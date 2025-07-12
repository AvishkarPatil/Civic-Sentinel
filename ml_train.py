from advanced_detector import AdvancedCivicDetector
import os

detector = AdvancedCivicDetector()

normal_images = []
for file in os.listdir("images/normal"):
    if file.endswith(('.jpg', '.jpeg', '.png')):
        normal_images.append(f"images/normal/{file}")

print(f"Training on {len(normal_images)} normal images...")

detector.train_on_normal_images(normal_images)

detector.save_model("civic_model.pkl")
print("ML model trained and saved!")
