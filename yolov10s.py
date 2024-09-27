from ultralytics import YOLO



model = YOLO("yolov8s.pt")

source = 'http://images.cocodataset.org/val2017/000000039769.jpg'
model.predict(source=source, save=True)
