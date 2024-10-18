from ultralytics import YOLO

model = YOLO(r"models\segment\seg_col_v0.onnx")
model.predict(r"images", save=True)

model = YOLO(r"models\segment\seg_row_v0.onnx")
model.predict(r"images", save=True)