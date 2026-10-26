from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.track(source = "video.mp4", show = True, save = True)