"""One small, shared static-sign pipeline for the internship application."""
from __future__ import annotations

import json
import string
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


# These are the only labels that exist in the selected subset of the supplied
# A-Z image dataset. They are deliberately visually distinct static signs.
# The final internship subset is deliberately limited to static ASL signs for
# which the supplied A-Z dataset has 3,000 labelled images per class.
SUPPORTED_SIGNS = ["A", "B", "C", "D", "L", "V", "W", "Y"]
ALPHABET_SIGNS = list(string.ascii_uppercase)


@dataclass(frozen=True)
class SignResult:
    label: str | None
    confidence: float
    recognized: bool
    bbox: tuple[int, int, int, int] | None
    top5: tuple[tuple[str, float], ...] = ()
    reason: str | None = None


def landmark_features(landmarks: np.ndarray, handedness: str) -> np.ndarray:
    """Translation/scale normalise MediaPipe's 21 hand landmarks.

    Left hands are mirrored into the same coordinate orientation as right
    hands. This exact function is used during both training and inference.
    """
    points = np.asarray(landmarks, dtype=np.float32).copy()
    points -= points[0]  # wrist origin
    if handedness.lower() == "left":
        points[:, 0] *= -1.0
    scale = float(np.linalg.norm(points[9]))  # middle-finger MCP distance
    if scale < 1e-6:
        scale = max(float(np.linalg.norm(points, axis=1).max()), 1e-6)
    return (points / scale).reshape(-1).astype(np.float32)


def hand_bbox(landmarks: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    xs = np.clip(landmarks[:, 0], 0.0, 1.0)
    ys = np.clip(landmarks[:, 1], 0.0, 1.0)
    x1, x2 = int(np.floor(xs.min() * width)), int(np.ceil(xs.max() * width))
    y1, y2 = int(np.floor(ys.min() * height)), int(np.ceil(ys.max() * height))
    margin = max(3, int(round(max(x2 - x1, y2 - y1) * 0.10)))
    return max(0, x1 - margin), max(0, y1 - margin), min(width - 1, x2 + margin), min(height - 1, y2 + margin)


class HandLandmarkExtractor:
    """The only hand detector/preprocessor used by training and the app."""

    def __init__(self) -> None:
        # First pass: robust for backlit hands such as A_test.jpg.
        self._enhanced_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.10,
        )
        # Some high-contrast silhouette signs are found by the full detector
        # on the original pixels, but not after local-contrast enhancement.
        # This pass is shared by training, uploads, and WebRTC frames.
        self._original_full_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            # A very low fallback threshold is deliberate: the classifier's
            # genuine probability still rejects unsupported/non-sign poses,
            # while dark silhouette test images remain detectable.
            min_detection_confidence=0.01,
        )
        # Fallback: some thin-finger signs (V/W) are recognised more reliably
        # from the original pixels by MediaPipe's lightweight model.
        self._original_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.10,
        )
        self._lock = threading.RLock()

    @staticmethod
    def _detector_image(image_bgr: np.ndarray) -> np.ndarray:
        """Apply one mild, shared local-contrast step before MediaPipe.

        This preserves geometry while making dark hands against bright windows
        detectable. Training, upload, and webcam all call this same function.
        """
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        return cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    def extract(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
        if image_bgr is None or image_bgr.size == 0:
            return None
        detector_bgr = self._detector_image(image_bgr)
        with self._lock:
            result = self._enhanced_hands.process(cv2.cvtColor(detector_bgr, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                result = self._original_hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if not result.multi_hand_landmarks:
                result = self._original_full_hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None
        landmarks = result.multi_hand_landmarks[0]
        handedness = result.multi_handedness[0].classification[0].label
        points = np.asarray([[point.x, point.y, point.z] for point in landmarks.landmark], dtype=np.float32)
        return landmark_features(points, handedness), hand_bbox(points, image_bgr.shape)

    def close(self) -> None:
        self._enhanced_hands.close()
        self._original_full_hands.close()
        self._original_hands.close()


class InternshipSignPipeline:
    """MediaPipe hand landmarks -> one CPU SVM -> genuine probabilities."""

    def __init__(self, model_path: str | Path, labels_path: str | Path, threshold: float = 0.70) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = float(threshold)
        self.error: str | None = None
        self.model: Any | None = None
        self.labels: list[str] = []
        self.extractor = HandLandmarkExtractor()
        try:
            import joblib

            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
                raise ValueError("Model label file must contain unique class labels.")
            self.model = joblib.load(self.model_path)
            model_labels = list(self.model.named_steps["svc"].classes_)
            if model_labels != self.labels:
                raise ValueError("SVM class labels do not match the saved label order.")
        except Exception as exc:
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, image_bgr: np.ndarray) -> SignResult:
        if not self.available:
            return SignResult(None, 0.0, False, None, reason=self.error or "Model unavailable")
        hand = self.extractor.extract(image_bgr)
        if hand is None:
            return SignResult(None, 0.0, False, None, reason="No hand detected")
        features, bbox = hand
        probabilities = np.asarray(self.model.predict_proba(features[None, :])[0], dtype=np.float32)
        index = int(np.argmax(probabilities))
        top_indices = np.argsort(probabilities)[-5:][::-1]
        confidence = float(probabilities[index])
        return SignResult(
            label=self.labels[index] if confidence >= self.threshold else None,
            confidence=confidence,
            recognized=confidence >= self.threshold,
            bbox=bbox,
            top5=tuple((self.labels[int(item)], float(probabilities[int(item)])) for item in top_indices),
            reason=None if confidence >= self.threshold else "Unknown Sign",
        )

    def close(self) -> None:
        self.extractor.close()
