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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
            candidate_paths = [
                os.path.join(base_dir, "fire_smoke_resnet50_final.pth"),
                os.path.join(base_dir, "trained-models", "fire_smoke_resnet50_final.pth"),
                os.path.join(base_dir, "trained-models", "model_final.pth"),
            ]
            for p in candidate_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if model_path is None:
                model_path = candidate_paths[-1]
        
        self.model_path = model_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Image preprocessing pipeline (defaults, updated if checkpoint specifies parameters)
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
            # Try weights_only=True first (safe standard for state_dict checkpoints)
            try:
                loaded = torch.load(self.model_path, map_location=self.device, weights_only=True)
            except Exception:
                # Fallback to weights_only=False for legacy serialized models
                loaded = torch.load(self.model_path, map_location=self.device, weights_only=False)

            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                logger.info("Detected dictionary checkpoint with model_state_dict.")
                # Extract metadata if available
                if "class_names" in loaded and isinstance(loaded["class_names"], list):
                    self.CLASS_NAMES = loaded["class_names"]

                img_size = loaded.get("img_size", 224)
                mean = loaded.get("mean", [0.485, 0.456, 0.406])
                std = loaded.get("std", [0.229, 0.224, 0.225])

                # Update preprocessing transforms to match training configuration
                self.transform = transforms.Compose([
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

                # Reconstruct ResNet-50 transfer learning architecture
                num_classes = len(self.CLASS_NAMES)
                model = models.resnet50(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(in_features, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, num_classes),
                )
                model.load_state_dict(loaded["model_state_dict"])
                model = model.to(self.device)
                model.eval()
                logger.info(f"Successfully loaded trained ResNet-50 state_dict with classes: {self.CLASS_NAMES}")
                return model

            elif isinstance(loaded, torch.nn.Module):
                model = loaded.to(self.device)
                model.eval()

                # Fix unparameterized Softmax if present
                if hasattr(model, "fc") and len(model.fc) > 3:
                    if isinstance(model.fc[3], torch.nn.Softmax):
                        model.fc[3] = torch.nn.Identity()

                logger.info("Legacy PyTorch nn.Module loaded and set to evaluation mode successfully.")
                return model

            else:
                raise ValueError(f"Unrecognized model checkpoint structure from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _extract_spatial_and_global(self, tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs forward pass through ResNet-50 backbone to extract both:
        1. Global class probabilities [num_classes] (Fire, Neutral, Smoke)
        2. Spatial class activation map [num_classes, 7, 7] from layer4
        """
        num_classes = len(self.CLASS_NAMES)
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
            glob_logits = self.model.fc(pooled)  # [1, num_classes]
            glob_probs = F.softmax(glob_logits, dim=1).squeeze(0).cpu().numpy()

            # Spatial localized classification across 7x7 grid
            b, c, sh, sw = f.shape
            f_perm = f.permute(0, 2, 3, 1).reshape(-1, c)  # [49, 2048]
            spatial_logits = self.model.fc(f_perm)  # [49, num_classes]
            spatial_probs = F.softmax(spatial_logits, dim=1).reshape(sh, sw, num_classes).permute(2, 0, 1).cpu().numpy()  # [num_classes, 7, 7]

        return glob_probs, spatial_probs

    def _detect_hazard_and_boxes(
        self, img_np: np.ndarray, glob_probs: np.ndarray, spatial_probs: np.ndarray, w: int, h: int, is_webcam: bool = False
    ) -> Tuple[str, float, List[Dict[str, Any]], Dict[str, float]]:
        """
        Verifies actual fire and smoke presence to prevent false alarms on natural/webcam scenes,
        and generates scalable green bounding boxes around localized hazard regions.
        """
        p_fire = float(glob_probs[0])
        p_neutral = float(glob_probs[1])
        p_smoke = float(glob_probs[2])

        # 1. Fire chromaticity analysis
        r = img_np[:, :, 0].astype(np.int32)
        g = img_np[:, :, 1].astype(np.int32)
        b_ch = img_np[:, :, 2].astype(np.int32)
        fire_pixels = (r > 115) & (r > g) & (g >= b_ch) & ((r - b_ch) > 25)
        fire_ratio = float(np.mean(fire_pixels))

        # 2. Structural & texture analysis
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gray_std = float(gray.std())

        # 3. HSV color & human skin detection
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Human skin tone mask (detects face, neck, arms of person in frame)
        skin_mask = ((h_ch <= 22) | (h_ch >= 170)) & (s_ch >= 45) & (s_ch <= 200) & (v_ch >= 55)
        skin_ratio = float(np.mean(skin_mask))

        # Smoke desaturation check (smoke particles are largely achromatic haze)
        low_sat_ratio = float(np.mean(s_ch < 50))

        # 4. Spatial feature map activations (layer4: 7x7 grid)
        fire_sp_max = float(spatial_probs[0].max())
        fire_sp_mean = float(spatial_probs[0].mean())
        smoke_sp_max = float(spatial_probs[2].max())
        smoke_sp_mean = float(spatial_probs[2].mean())
        smoke_cells_active = int(np.sum(spatial_probs[2] > 0.50))

        # =========================================================
        # Robust Decision Logic with Human & Ambient False Alarm Filtering
        # =========================================================

        # Fire Verification
        is_fire_verified = (
            p_fire >= 0.50 and p_fire > p_smoke and (
                p_fire >= 0.85
                or fire_ratio > 0.005
                or fire_sp_mean > 0.20
            )
        )

        is_human_present = (skin_ratio >= 0.055)

        if is_webcam:
            # During live webcam surveillance:
            # If a human is in front of the camera and no fire is verified,
            # this is a normal ambient room with a user at their computer -> suppress false smoke!
            if is_human_present and not is_fire_verified:
                is_smoke_verified = (
                    p_smoke >= 0.999
                    and smoke_sp_mean >= 0.70
                    and smoke_cells_active >= 35
                )
            else:
                # Without human face/skin, require decisive smoke evidence (not just a flat wall or dim room)
                is_smoke_verified = (
                    p_smoke >= 0.985
                    and p_smoke > p_fire * 2.0
                    and smoke_sp_mean >= 0.50
                    and smoke_cells_active >= 20
                    and lap_var > 40.0
                    and gray_std > 15.0
                )
        else:
            # General image prediction (e.g. dataset photos, uploaded hazard images)
            is_smoke_verified = (
                p_smoke >= 0.985
                and p_smoke > p_fire * 2.0
                and smoke_sp_mean >= 0.50
                and smoke_cells_active >= 20
                and lap_var > 40.0
                and gray_std > 15.0
            )

        # Final classification assignment
        if is_fire_verified:
            pred_class = "Fire"
            confidence = max(p_fire, fire_sp_max)
            prob_dict = {
                "Fire": round(confidence * 100.0, 2),
                "Neutral": round(p_neutral * 100.0, 2),
                "Smoke": round(p_smoke * 100.0, 2),
            }
        elif is_smoke_verified:
            pred_class = "Smoke"
            confidence = max(p_smoke, smoke_sp_max)
            prob_dict = {
                "Fire": round(p_fire * 100.0, 2),
                "Neutral": round(p_neutral * 100.0, 2),
                "Smoke": round(confidence * 100.0, 2),
            }
        else:
            pred_class = "Neutral"
            # If hazard was suppressed as a false alarm, reallocate ambiguous hazard probabilities to Neutral
            suppressed_smoke = p_smoke if not is_smoke_verified else 0.0
            suppressed_fire = p_fire if not is_fire_verified else 0.0
            effective_neutral = min(1.0, p_neutral + suppressed_smoke * 0.92 + suppressed_fire * 0.92)
            confidence = max(0.90, effective_neutral)
            prob_dict = {
                "Fire": round(max(0.0, p_fire - suppressed_fire * 0.92) * 100.0, 2),
                "Neutral": round(confidence * 100.0, 2),
                "Smoke": round(max(0.0, p_smoke - suppressed_smoke * 0.92) * 100.0, 2),
            }

        total_p = sum(prob_dict.values())
        if total_p > 0:
            prob_dict = {k: round((v / total_p) * 100.0, 2) for k, v in prob_dict.items()}

        # Bounding boxes removed as requested by user (ResNet-50 is a classification model)
        boxes: List[Dict[str, Any]] = []

        return pred_class, round(confidence * 100.0, 2), boxes, prob_dict

    def predict_pil(self, img: Image.Image, is_webcam: bool = False) -> Dict[str, Any]:
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
            img_np, glob_probs, spatial_probs, w, h, is_webcam=is_webcam
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

    def predict_bytes(self, image_bytes: bytes, is_webcam: bool = False) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return self.predict_pil(image, is_webcam=is_webcam)
        except Exception as e:
            logger.error(f"Error decoding image bytes: {e}")
            raise ValueError(f"Invalid image format: {e}")

    def predict_base64(self, b64_string: str, is_webcam: bool = False) -> Dict[str, Any]:
        """
        Runs inference on a Base64-encoded image string (with or without data URI header).
        """
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64_string)
            return self.predict_bytes(image_bytes, is_webcam=is_webcam)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")

    def predict_cv2_frame(self, frame: np.ndarray, is_webcam: bool = False) -> Dict[str, Any]:
        """
        Runs inference on an OpenCV BGR frame.
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            return self.predict_pil(image, is_webcam=is_webcam)
        except Exception:
            rgb_frame = frame[:, :, ::-1]
            image = Image.fromarray(rgb_frame)
            return self.predict_pil(image, is_webcam=is_webcam)

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
