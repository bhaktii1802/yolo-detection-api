import time
import cv2
import numpy as np
from ultralytics import YOLO

img = cv2.imread("test.jpg")

RUNS = 50
WARMUP = 5

def bench(path):
    model = YOLO(path, task = 'detect')
    for _ in range(WARMUP):
        model(img, verbose = False)
        

    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        model(img, verbose = False)
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    print(f"{path:15s} | avg {avg:7.1f} ms | median {np.median(times):7.1f} ms | {1000 / avg:5.1f} FPS")


bench("yolo11n.pt")
bench("yolo11n.onnx")

