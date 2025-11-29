from ultralytics import YOLO

def main():
    # Load a YOLOv8 pretrained model (Nano version)
    model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy but slower

    # Train the model
    model.train(
        data="Dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="mh_plate_detector"
    )

if __name__ == "__main__":
    main()
