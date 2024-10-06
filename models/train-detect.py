from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt") 
    model.train(
    data="D:\.Capstone\workspace\dataset\dataset.yaml", 
    epochs=200, 
    imgsz=480, 
    device="cuda",
)
