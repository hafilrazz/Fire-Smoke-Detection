"""
Flask Application for Fire & Smoke Detection Web Project.
Provides interactive web dashboard and REST API endpoints for
image classification, video analysis, live webcam surveillance, and sample testing.
"""

import os
import io
import time
import uuid
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from model_service import FireSmokeDetector

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEST_IMGS_DIR = os.path.join(BASE_DIR, "test-imgs")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
CORS(app)

# 50 MB max upload size
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

# Initialize detector singleton
detector = FireSmokeDetector.get_instance()


def allowed_file(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# ==========================================
# Web UI Routes
# ==========================================

@app.route("/")
def index():
    """Renders the main surveillance web dashboard."""
    return render_template("index.html")


# ==========================================
# REST API Endpoints
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """System health check and model specifications."""
    import torch
    return jsonify({
        "status": "healthy",
        "service": "Fire & Smoke Detection AI",
        "model": "ResNet-50 (Transfer Learning)",
        "classes": detector.CLASS_NAMES,
        "device": str(detector.device),
        "pytorch_version": torch.__version__,
        "timestamp": time.time()
    })


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """Returns a list of bundled sample test images for quick one-click testing."""
    if not os.path.exists(TEST_IMGS_DIR):
        return jsonify([])

    files = [f for f in os.listdir(TEST_IMGS_DIR) if allowed_file(f, ALLOWED_IMAGE_EXTENSIONS)]
    files.sort()

    samples = []
    for f in files:
        # Pre-assign helpful hint categories based on dataset inspection
        hint = "Sample"
        category = "other"
        if "fire" in f.lower() or f in ["7.jpg", "94.jpg", "image_1.jpg", "image_14.jpg"]:
            hint = "Fire Sample"
            category = "fire"
        elif "smoke" in f.lower() or f in ["36.jpg", "55.jpg", "image_0.jpg", "image_13.jpg", "image_24.jpg"]:
            hint = "Smoke Sample"
            category = "smoke"
        elif "neutral" in f.lower() or f in ["26.jpg", "73.jpg", "image_12.jpg", "image_7.jpg", "image_3.jpg"]:
            hint = "Neutral Sample"
            category = "neutral"

        samples.append({
            "filename": f,
            "hint": hint,
            "category": category,
            "url": f"/api/samples/{f}"
        })

    return jsonify(samples)


@app.route("/api/samples/<filename>", methods=["GET"])
def get_sample_image(filename):
    """Serves a test image from test-imgs folder."""
    return send_from_directory(TEST_IMGS_DIR, filename)


@app.route("/api/predict/image", methods=["POST"])
def predict_image():
    """
    Classifies an uploaded image file or Base64 payload.
    Accepts:
      - Multipart/form-data with key 'file' or 'image'
      - JSON with key 'image' (base64) or 'sample_filename'
    """
    try:
        # 1. Check for sample filename in JSON or form
        if request.is_json:
            data = request.get_json()
            if "sample_filename" in data:
                filename = secure_filename(data["sample_filename"])
                filepath = os.path.join(TEST_IMGS_DIR, filename)
                if not os.path.exists(filepath):
                    return jsonify({"status": "error", "message": "Sample image not found"}), 404
                with open(filepath, "rb") as f:
                    image_bytes = f.read()
                res = detector.predict_bytes(image_bytes)
                res["source"] = f"sample: {filename}"
                return jsonify(res)

            # 2. Check for base64 in JSON
            if "image" in data:
                res = detector.predict_base64(data["image"])
                res["source"] = "base64"
                return jsonify(res)

        # 3. Check for file in multipart form-data
        file = request.files.get("file") or request.files.get("image")
        if file and file.filename != "":
            if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                return jsonify({
                    "status": "error",
                    "message": f"Unsupported file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
                }), 400

            image_bytes = file.read()
            res = detector.predict_bytes(image_bytes)
            res["filename"] = secure_filename(file.filename)
            res["source"] = "upload"
            return jsonify(res)

        return jsonify({
            "status": "error",
            "message": "No valid image provided. Send a file or base64 JSON."
        }), 400

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/frame", methods=["POST"])
def predict_frame():
    """
    Lightweight, high-speed endpoint for streaming webcam frames (base64).
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "frame" not in data:
            return jsonify({"status": "error", "message": "Missing 'frame' in request body"}), 400

        res = detector.predict_base64(data["frame"])
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict/video", methods=["POST"])
def predict_video():
    """
    Processes an uploaded video file, generating frame-by-frame hazard detection,
    event timeline, and statistics.
    """
    file = request.files.get("file") or request.files.get("video")
    if not file or file.filename == "":
        return jsonify({"status": "error", "message": "No video file uploaded."}), 400

    if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({
            "status": "error",
            "message": f"Unsupported video type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        }), 400

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex[:8]
    temp_input = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{filename}")

    try:
        file.save(temp_input)

        # Get optional sampling parameter
        interval_sec = float(request.form.get("sample_interval", 0.5))

        logger.info(f"Processing video: {filename}, sample interval: {interval_sec}s")
        res = detector.process_video(
            video_path=temp_input,
            output_path=None,
            sample_interval_sec=interval_sec,
            max_duration_sec=60.0
        )
        res["filename"] = filename
        return jsonify(res)

    except Exception as e:
        logger.error(f"Video processing error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Cleanup temp file
        if os.path.exists(temp_input):
            try:
                os.remove(temp_input)
            except Exception:
                pass


@app.route("/api/utils/<filename>", methods=["GET"])
def get_util_asset(filename):
    """Serves asset images from utils/ (accuracy plot, trainloss, etc.)."""
    utils_dir = os.path.join(BASE_DIR, "utils")
    return send_from_directory(utils_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Fire-Smoke-Detection Web Server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
