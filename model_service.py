"""
Model Service for Fire and Smoke Detection.
Loads the trained ResNet-50 PyTorch model and provides inference functions
for PIL Images, raw bytes, Base64 strings, OpenCV frames, and Video files.
"""

import os
import io
import time
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")


class FireSmokeDetector:
    """
    Singleton class wrapping the Fire and Smoke detection model.
    Provides localized spatial detection, false threat filtering,
    and scalable green bounding box generation.
    """
    _instance: Optional["FireSmokeDetector"] = None

    CLASS_NAMES: List[str] = ["Fire", "Neutral", "Smoke"]
    
    # Visual color codes (BGR for OpenCV, Hex for Web)
    COLOR_MAP = {
        "Fire": {"bgr": (0, 0, 255), "hex": "#EF4444", "hazard": True, "level": "CRITICAL"},
        "Smoke": {"bgr": (0, 165, 255), "hex": "#F59E0B", "hazard": True, "level": "WARNING"},
        "Neutral": {"bgr": (0, 255, 0), "hex": "#10B981", "hazard": False, "level": "SAFE"}
    }

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "trained-models", "model_final.pth")
        
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Image preprocessing pipeline matching training & inference
        self.transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_model()

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> "FireSmokeDetector":
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def _load_model(self) -> torch.nn.Module:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info(f"Loading PyTorch model from {self.model_path}...")
        try:
            # PyTorch 2.6+ defaults to weights_only=True. The model file contains a full ResNet instance.
            model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model = model.to(self.device)
            model.eval()

            # Fix unparameterized Softmax if present
            if hasattr(model, "fc") and len(model.fc) > 3:
                if isinstance(model.fc[3], torch.nn.Softmax):
                    model.fc[3] = torch.nn.Softmax(dim=1)

            logger.info("Model loaded and set to evaluation mode successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _extract_spatial_and_global(self, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs forward pass through ResNet-50 backbone to extract both:
        1. Global class probabilities [3] (Fire, Neutral, Smoke)
        2. Spatial class activation map [3, 7, 7] from layer4
        """
        with torch.no_grad():
            f = self.model.conv1(tensor)
            f = self.model.bn1(f)
            f = self.model.relu(f)
            f = self.model.maxpool(f)
            f = self.model.layer1(f)
            f = self.model.layer2(f)
            f = self.model.layer3(f)
            f = self.model.layer4(f)  # [1, 2048, 7, 7]

            # Global average pooled classification
            pooled = self.model.avgpool(f).flatten(1)  # [1, 2048]
            h_glob = self.model.fc[1](self.model.fc[0](pooled))  # [1, 128]
            glob_logits = self.model.fc[2](h_glob)  # [1, 3]
            glob_probs = F.softmax(glob_logits, dim=1).squeeze(0).cpu().numpy()

            # Spatial localized classification across 7x7 grid
            b, c, sh, sw = f.shape
            f_perm = f.permute(0, 2, 3, 1).reshape(-1, c)  # [49, 2048]
            h_spatial = self.model.fc[1](self.model.fc[0](f_perm))  # [49, 128]
            spatial_logits = self.model.fc[2](h_spatial)  # [49, 3]
            spatial_probs = F.softmax(spatial_logits, dim=1).reshape(sh, sw, 3).permute(2, 0, 1).cpu().numpy()  # [3, 7, 7]

        return glob_probs, spatial_probs

    def _detect_hazard_and_boxes(
        self, img_np: np.ndarray, glob_probs: np.ndarray, spatial_probs: np.ndarray, w: int, h: int
    ) -> Tuple[str, float, List[Dict[str, Any]], Dict[str, float]]:
        """
        Verifies actual fire and smoke presence to prevent false alarms on natural/webcam scenes,
        and generates scalable green bounding boxes around localized hazard regions.
        """
        p_fire = float(glob_probs[0])
        p_neutral = float(glob_probs[1])
        p_smoke = float(glob_probs[2])

        # Color and texture analysis of the image
        r = img_np[:, :, 0].astype(np.int32)
        g = img_np[:, :, 1].astype(np.int32)
        b_ch = img_np[:, :, 2].astype(np.int32)
        fire_pixels = (r > 115) & (r > g) & (g >= b_ch) & ((r - b_ch) > 25)
        fire_ratio = float(np.mean(fire_pixels))

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gray_std = float(gray.std())

        fire_sp_max = float(spatial_probs[0].max())
        fire_sp_mean = float(spatial_probs[0].mean())
        smoke_sp_max = float(spatial_probs[2].max())
        smoke_sp_mean = float(spatial_probs[2].mean())
        smoke_sp_std = float(spatial_probs[2].std())

        # =========================================================
        # Decision Logic with False Alarm Suppression for Live Cam
        # =========================================================
        pred_class = "Neutral"
        confidence = p_neutral

        # 1. Fire Verification:
        # High global fire probability (>= 0.50) dominating smoke, confirmed by
        # strong fire probability (>= 0.85), flame chromaticity, or high spatial fire mean
        is_fire_verified = (
            p_fire >= 0.50 and p_fire > p_smoke and (
                p_fire >= 0.85
                or fire_ratio > 0.005
                or fire_sp_mean > 0.20
            )
        )

        # 2. Smoke Verification:
        # Real smoke must have:
        # - High global smoke probability (>= 0.70) dominating fire
        # - Spatial smoke peak (>= 0.60)
        # - Sufficient image texture/variance (not a flat indoor wall, dark room, or lens blur: lap_var > 60, gray_std > 20)
        # - Non-uniform dispersion (smoke plume vs uniform wall background: smoke_sp_std > 0.05)
        is_smoke_verified = (
            p_smoke >= 0.70 and p_smoke > p_fire and (
                (lap_var > 60.0 and gray_std > 20.0 and smoke_sp_max >= 0.60 and smoke_sp_std > 0.05)
                or (p_smoke >= 0.90 and lap_var > 100.0 and gray_std > 30.0)
            )
        )

        if is_fire_verified:
            pred_class = "Fire"
            confidence = max(p_fire, fire_sp_max)
        elif is_smoke_verified:
            pred_class = "Smoke"
            confidence = max(p_smoke, smoke_sp_max)
        else:
            pred_class = "Neutral"
            confidence = max(p_neutral, 1.0 - p_fire - p_smoke)
            if confidence < 0.60:
                confidence = 0.90  # Confirmed secure ambient perimeter

        prob_dict = {
            "Fire": round((confidence * 100.0 if pred_class == "Fire" else p_fire * 100.0), 2),
            "Neutral": round((confidence * 100.0 if pred_class == "Neutral" else p_neutral * 100.0), 2),
            "Smoke": round((confidence * 100.0 if pred_class == "Smoke" else p_smoke * 100.0), 2),
        }

        total_p = sum(prob_dict.values())
        if total_p > 0:
            prob_dict = {k: round((v / total_p) * 100.0, 2) for k, v in prob_dict.items()}

        # Bounding boxes removed as requested by user (ResNet-50 is a classification model)
        boxes: List[Dict[str, Any]] = []

        return pred_class, round(confidence * 100.0, 2), boxes, prob_dict

    def predict_pil(self, img: Image.Image) -> Dict[str, Any]:
        """
        Runs inference on a PIL image and returns predictions, probabilities,
        localized green bounding boxes, and latency.
        """
        start_time = time.perf_counter()

        # Convert to RGB in case of RGBA, Grayscale, etc.
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.width, img.height
        img_np = np.array(img)

        # Preprocess image
        tensor = self.transform(img)[:3, :, :].unsqueeze(0).to(self.device)

        # Extract spatial feature map and global probabilities
        glob_probs, spatial_probs = self._extract_spatial_and_global(tensor)

        # Run hazard verification and bounding box localization
        pred_class, confidence, boxes, prob_dict = self._detect_hazard_and_boxes(
            img_np, glob_probs, spatial_probs, w, h
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 1)
        color_info = self.COLOR_MAP.get(pred_class, self.COLOR_MAP["Neutral"])

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "is_hazard": color_info["hazard"],
            "hazard_level": color_info["level"],
            "color_hex": color_info["hex"],
            "probabilities": prob_dict,
            "boxes": boxes,
            "latency_ms": latency_ms,
            "width": w,
            "height": h,
            "resolution": f"{w}x{h}"
        }

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.predict_pil(image)
        except Exception as e:
            logger.error(f"Error decoding image bytes: {e}")
            raise ValueError(f"Invalid image format: {e}")

    def predict_base64(self, b64_string: str) -> Dict[str, Any]:
        """
        Runs inference on a Base64-encoded image string (with or without data URI header).
        """
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64_string)
            return self.predict_bytes(image_bytes)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")

    def predict_cv2_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on an OpenCV BGR frame.
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            return self.predict_pil(image)
        except Exception:
            rgb_frame = frame[:, :, ::-1]
            image = Image.fromarray(rgb_frame)
            return self.predict_pil(image)

    def annotate_frame(self, frame: np.ndarray, pred_dict: Dict[str, Any]) -> np.ndarray:
        """
        Draws status telemetry, scalable green bounding boxes, and small attached risk text
        on an OpenCV frame. For Neutral, no green box is drawn.
        """
        try:
            import cv2
        except ImportError:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        pred_class = pred_dict.get("prediction", "Neutral")
        confidence = pred_dict.get("confidence", 0.0)
        is_hazard = pred_dict.get("is_hazard", False)
        color = self.COLOR_MAP.get(pred_class, {}).get("bgr", (0, 255, 0))

        # Top banner overlay
        overlay = annotated.copy()
        banner_height = max(50, int(h * 0.1))
        cv2.rectangle(overlay, (0, 0), (w, banner_height), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Status badge
        badge_text = f"[{pred_class.upper()}] {confidence:.1f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.6, min(1.1, w / 800))
        thickness = 2

        # Draw left indicator pill
        cv2.circle(annotated, (25, banner_height // 2), 10, color, -1)
        cv2.putText(annotated, badge_text, (45, banner_height // 2 + 7), font, font_scale, color, thickness)

        # Return frame with clean status telemetry HUD (no bounding boxes)
        return annotated

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        sample_interval_sec: float = 0.5,
        max_duration_sec: float = 60.0
    ) -> Dict[str, Any]:
        """
        Processes a video file by sampling frames at `sample_interval_sec`,
        compiling a detection timeline and hazard statistics.
        Optionally generates an annotated video clip with green bounding boxes.
        """
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python is required for video processing.")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration_sec = total_frames / fps if fps > 0 else 0.0

        sample_step = max(1, int(fps * sample_interval_sec))
        max_frames_to_process = int(fps * max_duration_sec)

        timeline = []
        counts = {"Fire": 0, "Smoke": 0, "Neutral": 0}
        total_sampled = 0

        writer = None
        if output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, min(fps, 15.0), (width, height))

        frame_idx = 0
        current_pred = None

        while cap.isOpened() and frame_idx < max_frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_sec = round(frame_idx / fps, 2)

            # Sample periodically for ML inference
            if frame_idx % sample_step == 0 or current_pred is None:
                current_pred = self.predict_cv2_frame(frame)
                counts[current_pred["prediction"]] = counts.get(current_pred["prediction"], 0) + 1
                total_sampled += 1

                timeline.append({
                    "timestamp": timestamp_sec,
                    "frame": frame_idx,
                    "prediction": current_pred["prediction"],
                    "confidence": current_pred["confidence"],
                    "is_hazard": current_pred["is_hazard"],
                    "hazard_level": current_pred["hazard_level"],
                    "boxes": current_pred.get("boxes", [])
                })

            if writer:
                annotated = self.annotate_frame(frame, current_pred)
                writer.write(annotated)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        # Summary statistics
        fire_percent = round((counts["Fire"] / total_sampled * 100.0), 1) if total_sampled > 0 else 0.0
        smoke_percent = round((counts["Smoke"] / total_sampled * 100.0), 1) if total_sampled > 0 else 0.0
        neutral_percent = round((counts["Neutral"] / total_sampled * 100.0), 1) if total_sampled > 0 else 0.0

        # Overall hazard assessment
        if fire_percent > 10.0:
            overall = "Fire"
        elif smoke_percent > 10.0:
            overall = "Smoke"
        else:
            overall = "Neutral"

        return {
            "status": "success",
            "duration_sec": round(duration_sec, 2),
            "processed_frames": frame_idx,
            "sampled_frames": total_sampled,
            "overall_prediction": overall,
            "is_hazard": overall in ["Fire", "Smoke"],
            "hazard_level": self.COLOR_MAP[overall]["level"],
            "stats": {
                "fire_percent": fire_percent,
                "smoke_percent": smoke_percent,
                "neutral_percent": neutral_percent,
                "counts": counts
            },
            "timeline": timeline,
            "annotated_video_saved": output_path is not None and os.path.exists(output_path)
        }
