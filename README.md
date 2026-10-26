# YOLO11 Object Detection API

A REST API that serves a YOLO11 object detection model with FastAPI, packaged in Docker. Upload an image, get back the detected objects as JSON.

## Features
- Object detection with a pretrained YOLO11n model (COCO, 80 classes)
- `/detect` endpoint built with FastAPI
- Simple HTML frontend (`index.html`) that calls the API
- Docker image (CPU) for reproducible deployment
- Benchmark script comparing PyTorch and ONNX Runtime latency

## Tech stack
Python, Ultralytics YOLO11, FastAPI, Uvicorn, OpenCV, Docker, ONNX Runtime

## Run with Docker
```bash
docker build -t yolo-api .
docker run -p 8000:8000 yolo-api
```
Open http://127.0.0.1:8000/docs to try the API, or open `index.html` in a browser.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn app:app --reload
```

## API example
`POST /detect` with an image file (form field name: `file`).

Response:
```json
{
  "count": 1,
  "detections": [
    {
      "class": "traffic light",
      "confidence": 0.94,
      "bbox": [94.1, 56.1, 169.1, 231.5]
    }
  ]
}
```
`bbox` is `[x1, y1, x2, y2]` in pixels.

## Benchmark: PyTorch vs ONNX Runtime
Measured with `benchmark.py` (50 timed runs after 5 warmup runs, one 640px image, batch size 1) on an Intel Core i5-1235U, CPU only. I repeated the benchmark 4 times.

| Format | Latency range across 4 runs | Approx. FPS |
|---|---|---|
| PyTorch (`yolo11n.pt`) | 87 to 211 ms | 5 to 11 |
| ONNX Runtime (`yolo11n.onnx`) | 113 to 277 ms | 4 to 9 |

ONNX Runtime was slower than PyTorch in all 4 runs (about 1.1x to 1.7x). Absolute latency varied a lot between runs because of CPU throttling on a laptop, so the ratio is more reliable than the raw numbers. For a model this small on a hybrid Intel CPU, ONNX Runtime gave no speed benefit.

To reproduce: run `pip install onnx onnxruntime`, then `python export.py` to create the ONNX file, then `python benchmark.py`.

## Project structure
- `app.py`: FastAPI app
- `Dockerfile`, `requirements.txt`: container setup
- `index.html`: minimal frontend
- `export.py`, `benchmark.py`: ONNX export and speed comparison
- `track.py`: detection + tracking on a video

## Next steps
- Fine-tune on a custom dataset
- Try OpenVINO for Intel CPUs
- Draw bounding boxes in the frontend
