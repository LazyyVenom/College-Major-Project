# Driver Safety System

Real-time driver monitoring system that detects unsafe driving behavior using a webcam and alerts the driver with audio + visual warnings.

## Features

| Feature | Method | How It Works |
|---------|--------|-------------|
| **Drowsiness Detection** | CNN Eye State Classifier | Monitors eye closure using deep learning based face mesh landmarks. Triggers alert if eyes are closed for >0.7 seconds |
| **Yawning Detection** | CNN Mouth State Classifier | Detects yawning using mouth state classification model. Triggers alert when mouth is wide open for sustained period |
| **Distraction Detection** | Head Pose Estimation | Uses `cv2.solvePnP` to estimate head orientation (yaw/pitch). Alerts if driver looks away >30 degrees |
| **Phone Detection** | YOLOv8 Object Detection | Detects cell phone usage in real-time using COCO pretrained YOLOv8 nano model |
| **Seatbelt Detection** | YOLOv5 Pretrained Model | Detects seatbelt presence using pretrained weights from Kaggle. Falls back to edge detection heuristic |
| **Drunk Detection** | Multi-Signal Impairment Model | Combines eye state variance, head sway, facial asymmetry, blink irregularity, and eye closure patterns |
| **Alert System** | Audio + Visual Overlay | Per-alert cooldowns, priority-based sound playback, colored borders and on-screen warnings |

## Tech Stack

- **OpenCV** - Webcam capture, display, image processing
- **MediaPipe** - Face mesh (468 landmarks) for eye state, yawn detection, head pose
- **YOLOv8 (ultralytics)** - Object detection for phone and seatbelt
- **pygame** - Audio alert playback
- **NumPy / SciPy** - Landmark math and distance calculations
- **matplotlib** - Accuracy charts and analysis graphs

## Prerequisites

- **Python 3.10 - 3.12** (MediaPipe requires <3.13)
- **uv** package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))
- **Webcam** connected to your machine

### Install uv (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/LazyyVenom/College-Major-Project.git
cd College-Major-Project
```

### 2. Install dependencies

```bash
uv sync
```

This will automatically:
- Download Python 3.12 if needed
- Create a virtual environment
- Install all dependencies

### 3. Run the system

```bash
uv run python main.py
```

The webcam feed will open with a live HUD showing yaw, pitch, drunk score, FPS, and safety status. Press **`q`** to quit.

## Seatbelt Model Setup (Optional)

The seatbelt detector works out of the box with a heuristic fallback (edge detection). For better accuracy, download the pretrained YOLOv5 seatbelt model:

### Option A: Automatic download

```bash
# Install kaggle CLI
uv add kaggle

# Set up Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings -> API -> Create New Token
# 2. Place the downloaded kaggle.json:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Download the model
uv run python download_model.py
```

### Option B: Manual download

1. Go to https://www.kaggle.com/datasets/sachinmlwala/seatbelt3
2. Download the dataset
3. Extract `best.pt` from the zip
4. Place it at `models/seatbelt.pt`

## Analyze a Video

To run the detection pipeline on a recorded video and generate accuracy graphs:

```bash
# Place your video at input_videos/input.MOV (or edit VIDEO_PATH in analyze_video.py)
uv run python analyze_video.py
```

This generates 8 charts in `analysis_output/`:

| Chart | Description |
|-------|-------------|
| `01_eye_score_over_time.png` | Eye state detection score with drowsiness threshold |
| `02_yawn_score_over_time.png` | Yawn detection score with yawning threshold |
| `03_head_pose_over_time.png` | Yaw and pitch angles with safe zone |
| `04_drunk_score_over_time.png` | Impairment score over time with threshold |
| `05_alert_timeline.png` | When each alert type was active |
| `06_accuracy_bar_chart.png` | Accuracy, precision, recall per feature |
| `07_confusion_matrices.png` | Confusion matrices for each detector |
| `08_system_summary.png` | Frame classification pie chart + overall accuracy |

## Project Structure

```
College_Major_Project/
├── main.py                     # Entry point - webcam loop, orchestrator
├── config.py                   # All thresholds and constants (edit to tune)
├── analyze_video.py            # Video analysis + graph generation
├── download_model.py           # Download seatbelt model from Kaggle
├── pyproject.toml              # uv project config and dependencies
├── detectors/
│   ├── eye_detector.py         # CNN-based drowsiness detection
│   ├── yawn_detector.py        # CNN-based yawning detection
│   ├── head_pose_detector.py   # solvePnP head orientation
│   ├── phone_detector.py       # YOLOv8 phone detection (threaded)
│   ├── drunk_detector.py       # Multi-signal impairment detection
│   └── seatbelt_detector.py    # YOLOv5 seatbelt detection (threaded)
├── alerts/
│   ├── alert_manager.py        # Cooldowns, priority, sound + visual alerts
│   └── sounds/                 # Auto-generated alert WAV files
├── utils/
│   ├── landmarks.py            # MediaPipe FaceMesh wrapper
│   └── drawing.py              # HUD overlay rendering
├── models/                     # Place seatbelt.pt here
├── notebooks/                  # Reference Kaggle notebooks
│   ├── drowsiness-detection-system.ipynb
│   ├── yawn-detector-using-cnn.ipynb
│   ├── head-pose-estimation-using-mediapipe.ipynb
│   ├── driver-fatigue-monitoring-system-with-yolo-v8.ipynb
│   └── seatbelt-vs-noseatbelt.ipynb
├── analysis_output/            # Generated accuracy charts
└── input_videos/               # Place test videos here
```

## Configuration

All detection thresholds are in `config.py`. Key values to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EYE_SCORE_THRESHOLD` | 0.22 | Below this = eyes closed |
| `EYE_SCORE_CONSECUTIVE_FRAMES` | 20 | Frames before drowsiness alert (~0.7s) |
| `YAWN_THRESHOLD` | 0.6 | Above this = mouth open (yawning) |
| `YAW_THRESHOLD` | 30 | Degrees of head turn before distraction alert |
| `PITCH_THRESHOLD` | 25 | Degrees of head tilt before distraction alert |
| `DRUNK_THRESHOLD` | 0.45 | Combined impairment score threshold |
| `PHONE_CONFIDENCE` | 0.5 | YOLO confidence for phone detection |
| `CAMERA_INDEX` | 0 | Change if you have multiple cameras |

## Alert System

Each alert type has independent cooldowns to prevent spamming:

| Alert | Cooldown | Priority | Color |
|-------|----------|----------|-------|
| Impairment | 10s | Highest | Dark Orange |
| Drowsiness | 5s | High | Red |
| Distraction | 4s | High | Yellow |
| Phone Usage | 6s | Medium | Blue |
| Yawning | 8s | Low | Orange |
| Seatbelt | 15s | Lowest | Dark Red |

When multiple alerts are active, all are shown visually but only the highest-priority sound plays.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cannot open camera` | Check `CAMERA_INDEX` in `config.py`. Try 1 if 0 doesn't work |
| `module 'mediapipe' has no attribute 'solutions'` | You need mediapipe <0.10.20. Run `uv sync` to get the correct version |
| Phone detection is slow | Normal - YOLOv8 runs in a background thread (~50-150ms per inference on CPU) |
| Seatbelt alert keeps firing | Place `seatbelt.pt` in `models/` or adjust heuristic thresholds |
| Low FPS (<15) | Reduce `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py` |
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
