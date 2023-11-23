from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")
# model = YOLO("runs/segment/train/weights/best.pt")

result = model.train(data="data.yaml", epochs=1, imgsz=256)
# print(model("0000.png"))
