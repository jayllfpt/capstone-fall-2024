from ultralytics import YOLO

model = YOLO(r"models\detect\det_col_v0.onnx")
model.predict(r"images", save=True)

model = YOLO(r"models\detect\det_row_v0.onnx")
model.predict(r"images", save=True)