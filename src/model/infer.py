from ultralytics import YOLO
import os

# Load trained model
model = YOLO("runs/detect/mh_plate_detector2/weights/best.pt")

# Path to one of your test images
image_name = "video2_230_jpg.rf.115438e18f9307a52df9006f982d3a1f.jpg"
image_path = f"Dataset/test/images/{image_name}"

# Verify file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Run prediction
results = model.predict(source=image_path, save=True, show=True)

print("Inference complete! Saved in runs/detect/predict/")
