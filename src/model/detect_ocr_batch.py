# src/model/detect_ocr_batch.py
import os, cv2
from ultralytics import YOLO
from src.ocr.ocr import extract_text_from_crop, format_plate

model = YOLO("runs/detect/mh_plate_detector2/weights/best.pt")
test_folder = "Dataset/test/images/"
out_crops = "runs/crops_batch/"
os.makedirs(out_crops, exist_ok=True)

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
print(f"Found {len(image_files)} images")

results_list = []
uncertain = []

for name in image_files:
    path = os.path.join(test_folder, name)
    img = cv2.imread(path)
    if img is None:
        print("Could not read:", name)
        continue

    # Run detection with a higher conf threshold to reduce false boxes
    preds = model.predict(source=path, conf=0.25, iou=0.45, max_det=5)[0]  # conf adjustable

    if len(preds.boxes) == 0:
        print(f"{name} -> No plate detected")
        results_list.append((name, None, None))
        continue

    # If multiple boxes, choose the one with largest area (likely correct)
    boxes = [box.xyxy[0].cpu().numpy() for box in preds.boxes]
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    best_idx = int(max(range(len(areas)), key=lambda i: areas[i]))
    x1,y1,x2,y2 = map(int, boxes[best_idx])

    # Add padding (10-18% of box dims)
    h = y2-y1; w = x2-x1
    pad = int(max(4, min(h,w)*0.15))
    x1p = max(0, x1-pad); y1p = max(0, y1-pad)
    x2p = min(img.shape[1], x2+pad); y2p = min(img.shape[0], y2+pad)

    crop = img[y1p:y2p, x1p:x2p]
    crop_file = os.path.join(out_crops, f"{name}_crop.jpg")
    cv2.imwrite(crop_file, crop)

    raw, formatted, score = extract_text_from_crop(crop)
    if formatted is None:
        print(f"{name} -> OCR failed")
        uncertain.append((name, crop_file, raw, score))
        results_list.append((name, None, None))
        continue

    print(f"{name} -> {formatted}  (score={score})")
    results_list.append((name, raw, formatted))

    # If low score, add to uncertain
    if score < 10:
        uncertain.append((name, crop_file, raw, score))

# Save a simple CSV of results
import csv
with open("runs/detect_batch_results.csv", "w", newline='', encoding='utf-8') as cf:
    w = csv.writer(cf)
    w.writerow(["image","raw_text","formatted"])
    for row in results_list:
        w.writerow(row)

print("Done. Crops saved to", out_crops)
print("Uncertain cases:", len(uncertain))
