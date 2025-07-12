from advanced_detector import AdvancedCivicDetector
import os

detector = AdvancedCivicDetector()

normal_images = []
for file in os.listdir("images/"):
    if file.startswith("normal") and file.endswith(('.jpg', '.jpeg', '.png')):
        normal_images.append(f"images/{file}")

print(f"Training on {len(normal_images)} normal images...")

detector.train_on_normal_images(normal_images)

detector.save_model("civic_model.pkl")
print("ML model trained and saved!")
