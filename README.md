# 🎓 Automatic Student Counter using Image and Video
### Powered by YOLOv8 · OpenCV · Python

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit)](https://streamlit.io/)

---

## 📌 Project Overview

This project automatically **detects and counts students (people)** present in:

- 📷 **Static images** (JPG, PNG, BMP …)
- 🎬 **Video files** (MP4, AVI, MOV …)
- 📡 **Live webcam feed** (real-time)

The system uses **YOLOv8**, a state-of-the-art object detection model trained on the COCO dataset. It filters detections to count only the *person* class, draws bounding boxes around each student, and overlays the total count directly on the frame.

An optional **Streamlit web interface** lets non-technical users upload files through a browser without touching the command line.

---

## 🏗️ Project Structure

```
student_counter/
├── main.py              ← CLI entry point  (image / video / webcam)
├── utils.py             ← All detection & drawing helpers
├── app.py               ← Streamlit web app  (optional)
├── generate_sample.py   ← Creates a synthetic test image
├── test_setup.py        ← Verifies all packages are installed
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🛠️ Installation

### 1. Clone / download the project

```bash
git clone https://github.com/yourname/student-counter.git
cd student-counter
```

### 2. Create and activate a virtual environment *(recommended)*

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch is installed automatically by `ultralytics`.  
> If you have an NVIDIA GPU, install the CUDA-enabled PyTorch first:  
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### 4. Verify setup

```bash
python test_setup.py
```

The first run will automatically **download YOLOv8n weights** (~6 MB) from Ultralytics.

---

## 🚀 How to Run

### A) Command-Line Interface (`main.py`)

#### Count students in an image

```bash
python main.py --image path/to/classroom.jpg
```

#### Count students in a video file

```bash
python main.py --video path/to/lecture.mp4
```

#### Real-time webcam counting

```bash
python main.py --webcam
```

#### Use a specific webcam (e.g., external USB camera)

```bash
python main.py --webcam-id 1
```

#### Save annotated output

```bash
python main.py --image classroom.jpg  --output result.jpg
python main.py --video lecture.mp4    --output annotated.mp4
```

#### Enable Region-of-Interest (ROI) mode

```bash
python main.py --webcam --roi
```
A window opens — drag to select the area to count in, then press **Enter**.

#### Save per-frame count log to CSV

```bash
python main.py --video lecture.mp4 --csv counts.csv
```

#### All options at a glance

```bash
python main.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--image PATH` | — | Input image path |
| `--video PATH` | — | Input video path |
| `--webcam` | — | Default webcam (device 0) |
| `--webcam-id ID` | — | Specific device index |
| `--model MODEL` | `yolov8n.pt` | YOLOv8 variant |
| `--conf FLOAT` | `0.40` | Min detection confidence |
| `--iou FLOAT` | `0.45` | NMS IoU threshold |
| `--output PATH` | none | Save annotated output |
| `--resize WIDTH` | `640` | Inference frame width |
| `--roi` | off | Draw ROI selector |
| `--csv PATH` | none | Frame-count CSV log |
| `--no-display` | off | Suppress preview window |

---

### B) Streamlit Web App (`app.py`)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

Features in the web app:
- Drag-and-drop image or video upload
- Live student count metric
- Side-by-side original vs. annotated image
- Count-per-frame line chart for videos
- Download annotated image or CSV log

---

### C) Generate a test image

```bash
python generate_sample.py
# Creates: sample_classroom.png
python main.py --image sample_classroom.png
```

---

## 🧠 How It Works

```
INPUT (image / video / webcam)
        │
        ▼
  Resize frame (640px wide)  ← speeds up inference
        │
        ▼
  YOLOv8 Inference
  ┌─────────────────────────────────┐
  │  Detects 80 COCO classes        │
  │  We keep only class 0 = person  │
  │  Apply confidence threshold     │
  │  Apply NMS (IoU threshold)      │
  └─────────────────────────────────┘
        │
        ▼
  Optional ROI filter
  (only count people inside region)
        │
        ▼
  Draw bounding boxes + labels
  Overlay count badge
        │
        ▼
  Display / Save / Log to CSV
```

---

## ⚠️ Limitations

| Limitation | Explanation |
|-----------|-------------|
| **Occlusion** | Heavily overlapping students may be detected as one or missed |
| **Low light** | YOLOv8 struggles with dark or blurry frames |
| **Dense crowds** | Very tight groups (> 20 people/m²) may under-count |
| **Camera angle** | Top-down views may misidentify people |
| **Small figures** | People very far from camera may not be detected |
| **Non-student people** | The model detects *all* people, not just students specifically |

---

## 🔮 Future Improvements

1. **Fine-tune YOLOv8** on a classroom-specific dataset for higher accuracy
2. **Re-ID / Tracking** (DeepSORT, ByteTrack) to avoid double-counting in video
3. **Heatmap visualization** showing where students congregate
4. **Attendance system** integration — map detected faces to a student database
5. **Alert system** — notify admin when count exceeds room capacity
6. **Multi-camera** aggregation for large lecture halls
7. **Edge deployment** — optimize with TensorRT / ONNX for Raspberry Pi / Jetson Nano
8. **Mobile app** wrapping the Streamlit backend with React Native

---

## 📊 Model Comparison

| Model | Size | Speed (CPU) | mAP50 |
|-------|------|-------------|-------|
| yolov8n | 6 MB | ~45 ms/frame | 37.3 |
| yolov8s | 22 MB | ~90 ms/frame | 44.9 |
| yolov8m | 50 MB | ~200 ms/frame | 50.2 |
| yolov8l | 87 MB | ~390 ms/frame | 52.9 |
| yolov8x | 137 MB | ~680 ms/frame | 53.9 |

Start with **yolov8n** for real-time use, switch to **yolov8s/m** for higher accuracy.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| ultralytics | ≥ 8.0 | YOLOv8 inference engine |
| opencv-python | ≥ 4.8 | Image/video capture & drawing |
| numpy | ≥ 1.24 | Array processing |
| streamlit | ≥ 1.32 | Web UI |
| Pillow | ≥ 10.0 | Image format handling |

---

## 👨‍💻 Author

Built as a college AI/ML project demonstrating real-time object detection with YOLOv8.

---

## 📄 License

MIT License — free to use and modify for educational purposes.
