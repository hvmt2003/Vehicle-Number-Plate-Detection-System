from flask import Flask, render_template, request
import cv2
import os
from ultralytics import YOLO
from src.ocr.ocr import extract_text_from_crop, clean_plate_text, format_plate

# Load YOLO model
model = YOLO("runs/detect/mh_plate_detector2/weights/best.pt")

app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "app/static/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
    file.save(input_path)

    img = cv2.imread(input_path)
    results = model.predict(input_path)[0]

    # get all bounding boxes
    boxes = [box.xyxy[0].cpu().numpy() for box in results.boxes]

    if len(boxes) == 0:
        return render_template("home.html", result="No plate detected")

    # pick largest box
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    best_idx = areas.index(max(areas))
    x1, y1, x2, y2 = map(int, boxes[best_idx])

    # crop plate
    crop = img[y1:y2, x1:x2]
    crop_path = os.path.join(UPLOAD_FOLDER, "crop.jpg")
    cv2.imwrite(crop_path, crop)

    # Run OCR
    best_text, formatted, score = extract_text_from_crop(crop)

    result = formatted if formatted else "OCR Failed"

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
    cv2.imwrite(output_path, img)

    import time
    return render_template("home.html",
                           result=result,
                           timestamp=time.time())

if __name__ == "__main__":
    app.run(debug=True)

