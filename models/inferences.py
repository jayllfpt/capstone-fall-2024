from ultralytics import YOLO

model = YOLO(r"runs\segment\train2\weights\best.pt")

model.predict(r"D:\.Capstone\workspace\dataset\train\images\PMC1064076_table_0.jpg", save=True)
