from ultralytics import YOLO
import cv2
import os
from src.ocr.ocr import extract_text_from_plate,clean_plate_text

# Load trained YOLO model
model = YOLO("runs/detect/mh_plate_detector2/weights/best.pt")

# Path to a test image (replace with any test image name)
image_name = "video2_230_jpg.rf.115438e18f9307a52df9006f982d3a1f.jpg"
image_path = f"Dataset/test/images/{image_name}"

# Check file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load image
img = cv2.imread(image_path)

# Run YOLO inference
results = model.predict(source=image_path)[0]

print("\n--- Detection + OCR Results ---")

# Loop through detected boxes
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Crop image
    crop = img[y1:y2, x1:x2]

    # Save crop temporarily
    crop_path = "plate_crop.jpg"
    cv2.imwrite(crop_path, crop)

    # Run OCR
    raw_text = extract_text_from_plate(crop_path)
    plate_text = clean_plate_text(raw_text)

    print("Detected Plate Number:", plate_text)

    # Show cropped plate
    cv2.imshow("Number Plate", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nDone.")
