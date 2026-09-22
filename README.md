# Fire & Smoke Detection Web Application

Detecting Fire and Smoke in real-time using Computer Vision, PyTorch, and Flask.

Early fire and smoke detection plays a critical role in saving lives, reducing property damage, and minimizing operational downtime. This project provides an end-to-end deep learning pipeline and an interactive web surveillance application for automated hazard detection across images, video files, and live webcam feeds.

---

## Demo & Sample Outputs
- **Live Demo GIF**: `utils/demo.gif`
- **Model Training Curves**: `utils/accuracy.png` and `utils/trainloss.png`
- **Class Examples**: `utils/fire.png` (Fire), `utils/smoke.png` (Smoke), `utils/neutral.png` (Neutral)

---

## Key Features

1. **Interactive Web Dashboard**:
   - Modern dark-mode CCTV/surveillance interface.
   - Real-time hazard alert banner (🔴 Fire, 🟠 Smoke, 🟢 Neutral).
   - Audio siren warning with an instant mute/unmute toggle.
   - Class confidence percentage meters and probability distribution spectrum.

2. **Multi-Input Inference**:
   - **Image Classification**: Drag & drop custom image files or select from a bundled 1-click test gallery.
   - **Video Analysis**: Upload MP4, AVI, MOV, or WEBM videos for automated periodic frame sampling, hazard timelines, and statistical summaries.
   - **Live Webcam Surveillance**: Direct in-browser camera feed with low-latency base64 streaming and dynamic FPS telemetry.

3. **Intelligent False-Alarm Suppression**:
   - Real-world indoor webcam feeds often suffer from false alarms due to warm lighting, wooden textures, or plain walls.
   - The inference engine incorporates multi-factor verification:
     - **Spatial Feature Map Analysis**: Inspects the $7 \times 7$ feature activation map from ResNet-50's `layer4` to verify concentrated plume dispersion vs uniform background.
     - **Flame Chromaticity Filtering**: Checks color channel distributions ($R > 115, R > G \ge B$).
     - **Laplacian Texture Variance**: Prevents flat, featureless surfaces or dark room noise from falsely triggering smoke alerts.

4. **Production REST API**:
   - Clean, fully documented JSON endpoints for health monitoring, batch predictions, streaming webcam frames, and video processing.

---

## Model Architecture & Specifications

- **Backbone**: Pretrained `ResNet-50` (Transfer Learning on ImageNet `IMAGENET1K_V2`).
- **Classifier Head**:
  $$\text{Dropout}(0.3) \to \text{Linear}(2048 \to 128) \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(128 \to 3)$$
- **Target Classes (3)**:
  - `Fire` (Critical hazard)
  - `Neutral` (Safe / Normal ambient environment)
  - `Smoke` (Warning hazard)
- **Input Dimensions**: $224 \times 224 \times 3$ RGB normalized with ImageNet standards:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`
- **Active Checkpoint Hierarchy**:
  1. `trained-models/fire_smoke_resnet50_final.pth` *(Primary state_dict checkpoint with training metadata)*
  2. `trained-models/model_final.pth` *(Trained fallback checkpoint)*
  3. `trained-models/model_final_legacy.pth` *(Legacy serialized checkpoint)*

---

## Project Structure

```
Fire-Smoke-Detection/
├── trained-models/
│   ├── fire_smoke_resnet50_final.pth # Primary PyTorch ResNet-50 state dict + metadata
│   ├── model_final.pth               # Fallback trained weights
│   └── model_final_legacy.pth        # Legacy serialized fallback
├── test-imgs/                        # Sample evaluation images for quick testing
├── utils/                            # Performance plots and visual assets
│   ├── accuracy.png
│   ├── trainloss.png
│   ├── demo.gif
│   ├── fire.png
│   ├── smoke.png
│   └── neutral.png
│
├── model_service.py                  # Singleton inference engine & heuristic verification
├── app.py                            # Flask server, routing, and REST API controller
├── run.py                            # CLI launcher script with configurable host/port
├── requirements.txt                  # Python dependencies
│
├── templates/
│   └── index.html                    # Surveillance dashboard UI
├── static/
│   ├── css/
│   │   └── style.css                 # Dark-mode dashboard stylesheet
│   └── js/
│       └── main.js                   # Frontend controller, WebRTC webcam, audio siren
│
├── tests/
│   └── test_api.py                   # Automated unit and API test suite (10 tests)
│
├── fire_smoke_resnet50.ipynb         # ResNet-50 transfer learning & fine-tuning notebook
├── Training.ipynb                    # Original training notebook
├── Inference.ipynb                   # Original notebook inference experiments
├── Dockerfile                        # Hugging Face Spaces & Container build config
├── .dockerignore                     # Build exclusion rules
└── README.md                         # Project documentation
```

---

## Quick Start & Installation

### 1. Prerequisites
- Python 3.9+ (Python 3.10 - 3.14 supported)
- (Optional) NVIDIA GPU with CUDA for accelerated inference (CPU is automatically supported)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python run.py
```
Or specify custom host and port:
```bash
python run.py --host 0.0.0.0 --port 5000
```

Open your browser and navigate to:
```
http://localhost:5000/
```

### 4. Production Deployment (Gunicorn)
For production Linux / container environments with multi-worker scaling:
```bash
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 app:app
```
*(Note: Gunicorn targets UNIX/POSIX environments. For local Windows hosting or development, use `python run.py`)*

---

## Running Automated Tests

Run the complete 10-test automated test suite (verifying model loading, sample classifications, API endpoints, false-alarm suppression, and video processing):

```bash
python -m pytest tests/test_api.py -v
```

---

## REST API Reference

### 1. Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "Fire & Smoke Detection AI",
  "model": "ResNet-50 (Transfer Learning)",
  "classes": ["Fire", "Neutral", "Smoke"],
  "device": "cpu",
  "pytorch_version": "2.x",
  "timestamp": 1726900000.0
}
```

### 2. Predict Image (Multipart Upload or JSON)
```http
POST /api/predict/image
```
**Options:**
- Multipart form-data with file field `file` or `image`
- JSON payload: `{"image": "<base64_data>"}` or `{"sample_filename": "26.jpg"}`

**Example curl (file upload):**
```bash
curl -X POST -F "file=@test-imgs/26.jpg" http://localhost:5000/api/predict/image
```

**Response:**
```json
{
  "prediction": "Neutral",
  "confidence": 98.45,
  "is_hazard": false,
  "hazard_level": "SAFE",
  "color_hex": "#10B981",
  "probabilities": {
    "Fire": 0.52,
    "Neutral": 98.45,
    "Smoke": 1.03
  },
  "boxes": [],
  "latency_ms": 64.2,
  "width": 800,
  "height": 600,
  "resolution": "800x600",
  "source": "upload"
}
```

### 3. Predict Streaming Frame (Live Webcam)
```http
POST /api/predict/frame
Content-Type: application/json

{"frame": "data:image/jpeg;base64,..."}
```

### 4. Process Video
```http
POST /api/predict/video
```
**Parameters (Multipart form):**
- `file`: Video file (`.mp4`, `.avi`, `.mov`, `.webm`)
- `sample_interval`: Sampling interval in seconds (default `0.5`)

**Response:**
```json
{
  "status": "success",
  "duration_sec": 12.5,
  "processed_frames": 300,
  "sampled_frames": 25,
  "overall_prediction": "Neutral",
  "is_hazard": false,
  "hazard_level": "SAFE",
  "stats": {
    "fire_percent": 0.0,
    "smoke_percent": 0.0,
    "neutral_percent": 100.0,
    "counts": {"Fire": 0, "Smoke": 0, "Neutral": 25}
  },
  "timeline": [
    {
      "timestamp": 0.0,
      "frame": 0,
      "prediction": "Neutral",
      "confidence": 99.1,
      "is_hazard": false,
      "hazard_level": "SAFE"
    }
  ]
}
```

### 5. Sample Gallery Endpoints
- `GET /api/samples` — Returns metadata and URLs for bundled test images.
- `GET /api/samples/<filename>` — Serves a specific test image.

---

## Training & Model Fine-Tuning

The repository includes [`fire_smoke_resnet50.ipynb`](fire_smoke_resnet50.ipynb) for end-to-end training and checkpoint exporting:
1. **Transfer Learning (Stage 1)**: ResNet-50 backbone is frozen; only the custom classification head is trained.
2. **Fine-Tuning (Stage 2)**: Top residual layers (`layer4`) are unfrozen with a reduced learning rate to adapt spatial features to smoke plumes and flame patterns.
3. **Export**: Saves state dictionary alongside preprocessing parameters (`img_size`, `mean`, `std`, `class_names`) into `fire_smoke_resnet50_final.pth`.

---

## Dataset
- [Fire-Smoke-Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip)
- Contains 1,000 images per class across Train and Test splits (`Fire`, `Neutral`, `Smoke`).

---

## References
1. PyImageSearch — [Fire and Smoke Detection with Deep Learning](https://www.pyimagesearch.com/2019/11/18/fire-and-smoke-detection-with-keras-and-deep-learning/)
2. DeepQuestAI — [Fire-Smoke-Dataset](https://github.com/DeepQuestAI/Fire-Smoke-Dataset)
