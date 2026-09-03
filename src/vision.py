from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np


@dataclass
class HandObservation:
    landmarks: np.ndarray
    handedness: str
    bbox: tuple[int, int, int, int]


@dataclass
class Prediction:
    label: str | None
    confidence: float
    is_known: bool
    top_predictions: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class HandRoiQuality:
    """The one authoritative quality decision for a single hand ROI."""
    sharpness: float
    minimum_sharpness: float

    @property
    def is_blurry(self) -> bool:
        return self.sharpness < self.minimum_sharpness


def landmark_features(landmarks: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Make scale/translation-normalised, handedness-invariant landmark features."""
    points = np.asarray(landmarks, dtype=np.float32).copy()
    points -= points[0]
    # Put left and right hands into one canonical orientation.
    if handedness.lower() == "left":
        points[:, 0] *= -1
    scale = np.linalg.norm(points[9])
    if scale < 1e-6:
        scale = max(float(np.linalg.norm(points, axis=1).max()), 1e-6)
    return (points / scale).reshape(-1)


def sequence_features(observations: list[HandObservation]) -> np.ndarray:
    """Return fixed left/right landmark slots for one temporal video frame."""
    output = np.zeros(126, dtype=np.float32)
    for observation in observations:
        offset = 0 if observation.handedness.lower() == "left" else 63
        output[offset:offset + 63] = landmark_features(observation.landmarks, observation.handedness)
    return output


class HandDetector:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_hands: int = 2,
        detection_confidence: float = 0.40,
        tracking_confidence: float = 0.40,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            model_complexity=0,
            # Still photos and WLASL clips can contain smaller/darker hands.
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[HandObservation]:
        height, width = frame_bgr.shape[:2]
        result = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return []
        observations: list[HandObservation] = []
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            points = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
            observations.append(HandObservation(
                landmarks=points,
                handedness=hand_info.classification[0].label,
                bbox=tight_hand_bbox(points, frame_bgr.shape),
            ))
        return observations

    def close(self) -> None:
        self._hands.close()


def tight_hand_bbox(landmarks: np.ndarray, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Return a tight landmark box in a target frame's coordinate system.

    MediaPipe landmarks are normalised, so the same points can be measured on
    a CPU-sized detector frame and mapped precisely to the original HD frame
    without scaling an already rounded detector rectangle.
    """
    height, width = frame_shape[:2]
    x_values = np.clip(landmarks[:, 0], 0.0, 1.0)
    y_values = np.clip(landmarks[:, 1], 0.0, 1.0)
    x1 = int(np.floor(x_values.min() * width))
    y1 = int(np.floor(y_values.min() * height))
    x2 = int(np.ceil(x_values.max() * width))
    y2 = int(np.ceil(y_values.max() * height))
    # Keep a tiny safety margin for landmark rounding only. Classifier context
    # is added separately by ``crop_hand_square`` so faces/backgrounds are not
    # accidentally pulled into the visible hand box.
    margin = max(2, int(round(max(x2 - x1, y2 - y1) * 0.02)))
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width - 1, x2 + margin),
        min(height - 1, y2 + margin),
    )


def detect_live_hands(
    detector: HandDetector,
    frame_bgr: np.ndarray,
    processing_width: int = 640,
) -> list[HandObservation]:
    """Detect on a compact copy, then map tight boxes to the original frame.

    The original browser frame is never upscaled or recompressed. At 1280x720
    this halves the MediaPipe pixel workload while all crop, sharpness, and
    model operations still use the actual camera-resolution pixels.
    """
    height, width = frame_bgr.shape[:2]
    if width <= processing_width:
        return detector.detect(frame_bgr)
    scale = processing_width / float(width)
    detector_height = max(1, int(round(height * scale)))
    detector_frame = cv2.resize(frame_bgr, (processing_width, detector_height), interpolation=cv2.INTER_AREA)
    detections = detector.detect(detector_frame)
    return [
        HandObservation(
            landmarks=observation.landmarks,
            handedness=observation.handedness,
            bbox=tight_hand_bbox(observation.landmarks, frame_bgr.shape),
        )
        for observation in detections
    ]


class SignClassifier:
    def __init__(self, model_path: str | Path, threshold: float) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.model: Any | None = None
        self.error: str | None = None
        try:
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            self.error = f"Model file not found: {self.model_path.name}"
        except Exception as exc:  # Surface a helpful message without breaking detection.
            self.error = f"Could not load model: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, observation: HandObservation) -> Prediction:
        if self.model is None:
            return Prediction(None, 0.0, False)
        features = landmark_features(observation.landmarks, observation.handedness).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = str(self.model.classes_[index])
        return Prediction(label, confidence, confidence >= self.threshold)


class LandmarkAlphabetClassifier:
    """Optional background-invariant A–Z classifier using MediaPipe geometry."""
    def __init__(self, model_path: str | Path, labels_path: str | Path, threshold: float) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = threshold
        self.model: Any | None = None
        self.error: str | None = None
        try:
            import json
            labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            if labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                raise ValueError("landmark label file is not the exact A–Z class order")
            self.model = joblib.load(self.model_path)
            if list(self.model.classes_) != labels:
                raise ValueError("landmark model classes do not match its label file")
        except FileNotFoundError:
            self.error = f"Model file not found: {self.model_path.name}"
        except Exception as exc:
            self.error = f"Could not load A–Z landmark model: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, observation: HandObservation) -> Prediction:
        if self.model is None:
            return Prediction(None, 0.0, False)
        features = landmark_features(observation.landmarks, observation.handedness).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = str(self.model.classes_[index])
        return Prediction(label, confidence, confidence >= self.threshold)


class TemporalWordClassifier:
    """Separate WLASL landmark-sequence classifier for known dynamic words."""
    def __init__(self, model_path: str | Path, labels_path: str | Path, threshold: float) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = threshold
        self.model: Any | None = None
        self.labels: list[str] = []
        self.sequence_length = 16
        self.error: str | None = None
        try:
            import json
            import tensorflow as tf
            metadata = json.loads(self.labels_path.read_text(encoding="utf-8"))
            self.labels = list(metadata["classes"])
            self.sequence_length = int(metadata["sequence_length"])
            features_per_frame = int(metadata.get("features_per_frame", 126))
            if not 2 <= len(self.labels) <= 10 or len(set(self.labels)) != len(self.labels):
                raise ValueError("word label file must contain 2–10 unique classes")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            if self.model.input_shape != (None, self.sequence_length, features_per_frame):
                raise ValueError(f"unexpected known-word model input: {self.model.input_shape}")
            if self.model.output_shape != (None, len(self.labels)):
                raise ValueError(f"known-word output does not match labels: {self.model.output_shape}")
        except FileNotFoundError:
            self.error = f"Model file not found: {self.model_path.name}"
        except Exception as exc:
            self.error = f"Could not load known-word model: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict_sequence(self, sequence: np.ndarray) -> Prediction:
        if self.model is None or sequence.shape != (self.sequence_length, 126):
            return Prediction(None, 0.0, False)
        probabilities = np.asarray(self.model(sequence[None, ...], training=False).numpy()[0], dtype=np.float32)
        if probabilities.shape != (len(self.labels),) or not np.isfinite(probabilities).all():
            return Prediction(None, 0.0, False)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        top_indices = np.argsort(probabilities)[-min(5, len(self.labels)):][::-1]
        return Prediction(
            self.labels[index],
            confidence,
            confidence >= self.threshold,
            tuple((self.labels[int(item)], float(probabilities[int(item)])) for item in top_indices),
        )


def padded_hand_roi_bounds(
    frame_shape: tuple[int, ...],
    bbox: tuple[int, int, int, int],
    hand_coverage: float = 0.75,
) -> tuple[int, int, int, int]:
    """Return the un-clipped square bounds used to make a hand ROI."""
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return bbox
    if not 0.1 < hand_coverage <= 1.0:
        raise ValueError("hand_coverage must be between 0.1 and 1.0")
    side = max(1, int(round(max(width, height) / hand_coverage)))
    center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - side / 2.0))
    crop_y1 = int(round(center_y - side / 2.0))
    return crop_x1, crop_y1, crop_x1 + side, crop_y1 + side


def visible_roi_bounds(frame_shape: tuple[int, ...], bounds: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Clip padded ROI bounds only for drawing them on the original frame."""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bounds
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def crop_hand_square(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    hand_coverage: float = 0.75,
) -> np.ndarray:
    """Return one padded square BGR hand ROI without resizing or re-encoding.

    ``hand_coverage=0.75`` adds roughly 17% space per edge around a tight
    landmark box. It retains fingertips and wrist while preventing a face or
    background from dominating a live classifier crop. Replicate padding
    preserves a hand near a frame edge.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = padded_hand_roi_bounds(frame_bgr.shape, bbox, hand_coverage)
    side = max(1, crop_x2 - crop_x1)
    pad_left, pad_top = max(0, -crop_x1), max(0, -crop_y1)
    pad_right = max(0, crop_x2 - frame_bgr.shape[1])
    pad_bottom = max(0, crop_y2 - frame_bgr.shape[0])
    source = cv2.copyMakeBorder(
        frame_bgr,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )
    crop_x1 += pad_left
    crop_y1 += pad_top
    return source[crop_y1:crop_y1 + side, crop_x1:crop_x1 + side]


def hand_crop_sharpness(hand_roi_bgr: np.ndarray) -> float:
    """Return a lightweight motion-focus score without modifying the ROI."""
    if hand_roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(hand_roi_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_hand_roi_quality(hand_roi_bgr: np.ndarray, minimum_sharpness: float) -> HandRoiQuality:
    """Calculate the live blur decision once and share it with all consumers."""
    return HandRoiQuality(
        sharpness=hand_crop_sharpness(hand_roi_bgr),
        minimum_sharpness=float(minimum_sharpness),
    )


def prepare_alphabet_image(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    size: int = 96,
    hand_coverage: float = 0.70,
) -> np.ndarray:
    """Prepare the exact RGB 0–255 tensor used by training and inference."""
    image = frame_bgr
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        if width > 0 and height > 0:
            image = crop_hand_square(frame_bgr, bbox, hand_coverage)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Training uses ``tf.image.resize`` after decoding an RGB JPEG. Use the
    # same interpolation/runtime here instead of OpenCV's INTER_AREA so an
    # uploaded image and a saved webcam ROI enter the CNN with the same RGB
    # channel order, 96x96 shape, and 0–255 pixel scale.
    import tensorflow as tf

    tensor = tf.image.resize(tf.convert_to_tensor(image), (size, size), antialias=False)
    # The saved Keras model owns the only Rescaling layer. Do not normalise or
    # apply softmax here; model.predict() already returns 26 softmax scores.
    return tensor.numpy().astype(np.float32)


def has_hand_like_content(frame_bgr: np.ndarray) -> bool:
    """Conservative guard for full-image fallback when landmarks are absent.

    The still-image ASL test set contains a few valid signs for which
    MediaPipe cannot obtain 21 landmarks.  This check lets those photos use
    the training-consistent full image, while rejecting blank/near-blank
    uploads so a classifier never supplies a random-looking default letter.
    """
    if frame_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # Broad warm-skin range plus an edge-content requirement. The thresholds
    # deliberately avoid treating a white/black/flat background as a sign.
    warm_tones = (((hue <= 28) | (hue >= 165)) & (saturation >= 18) & (value >= 45))
    edges = cv2.Canny(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 60, 140)
    return float(np.mean(warm_tones)) >= 0.012 and float(np.mean(edges > 0)) >= 0.004


class AlphabetImageClassifier:
    """Keras A–Z image classifier operating on the MediaPipe hand ROI."""
    def __init__(self, model_path: str | Path, labels_path: str | Path, threshold: float) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.threshold = threshold
        self.model: Any | None = None
        self.labels: list[str] = []
        self.error: str | None = None
        try:
            import json
            import tensorflow as tf
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
            if self.labels != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                raise ValueError("label file is not the exact A–Z class order")
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            if self.model.input_shape != (None, 96, 96, 3) or self.model.output_shape != (None, len(self.labels)):
                raise ValueError(f"unexpected A-Z model shape: {self.model.input_shape} -> {self.model.output_shape}")
            output_layer = self.model.get_layer("letter_probabilities")
            if getattr(output_layer.activation, "__name__", None) != "softmax":
                raise ValueError("A-Z output layer must expose one softmax probability vector")
        except FileNotFoundError:
            self.error = f"Model file not found: {self.model_path.name}"
        except Exception as exc:
            self.error = f"Could not load A–Z model: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def _predict_prepared(self, image: np.ndarray) -> Prediction:
        """Run the saved network on an already preprocessed RGB sign image."""
        if self.model is None:
            return Prediction(None, 0.0, False)
        # Calling the cached Keras model directly avoids the per-frame
        # DataAdapter/progress machinery of ``model.predict``. This is the
        # same trained graph and output tensor, just lower-overhead for WebRTC.
        probabilities = np.asarray(self.model(image[None, ...], training=False).numpy()[0], dtype=np.float32)
        if probabilities.shape != (len(self.labels),) or not np.isfinite(probabilities).all():
            return Prediction(None, 0.0, False)
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = self.labels[index]
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_predictions = tuple((self.labels[int(item)], float(probabilities[int(item)])) for item in top_indices)
        return Prediction(label, confidence, confidence >= self.threshold, top_predictions)

    def predict_from_bgr_image(self, image_bgr: np.ndarray) -> Prediction:
        """The one shared pixel-model path for uploads and final webcam ROIs."""
        return self._predict_prepared(prepare_alphabet_image(image_bgr))

    def predict_full_image(self, frame_bgr: np.ndarray) -> Prediction:
        """Classify a complete uploaded sign image using the training-time view."""
        return self.predict_from_bgr_image(frame_bgr)

    def predict(self, frame_bgr: np.ndarray, observation: HandObservation) -> Prediction:
        if self.model is None:
            return Prediction(None, 0.0, False)
        # The supplied A–Z model was trained on complete sign images.  Keep
        # upload inference in that exact training-time view; a hand crop may
        # still be drawn for the user, but must not override a correct letter.
        full_image = self.predict_full_image(frame_bgr)
        if full_image.is_known:
            return full_image
        # Retain a genuine crop fallback for unusually framed uploads.
        hand_crop = crop_hand_square(frame_bgr, observation.bbox, hand_coverage=0.75)
        return self.predict_from_bgr_image(hand_crop)

    def predict_webcam(
        self,
        frame_bgr: np.ndarray,
        observation: HandObservation,
        hand_roi_bgr: np.ndarray | None = None,
    ) -> Prediction:
        """Classify the padded hand ROI only for live camera frames.

        Webcam backgrounds are not part of the supplied alphabet dataset, so
        allowing a full-frame score to override the hand crop caused confident
        but unrelated letters to persist in the live overlay.
        """
        # Use the already measured ROI when the live processor supplies it;
        # this avoids a second crop/resize pass on every WebRTC frame.
        roi = hand_roi_bgr if hand_roi_bgr is not None else crop_hand_square(frame_bgr, observation.bbox, hand_coverage=0.75)
        return self.predict_from_bgr_image(roi)


class PredictionSmoother:
    """Require the same confident result in several adjacent frames."""
    def __init__(self, frames: int, threshold: float) -> None:
        self.frames = frames
        self.threshold = threshold
        self._candidate: str | None = None
        self._count = 0
        self._confidences: deque[float] = deque(maxlen=frames)

    def reset(self) -> None:
        """Discard a partial/stable result after uncertainty or a sign change."""
        self._candidate = None
        self._count = 0
        self._confidences.clear()

    @property
    def progress(self) -> int:
        """Number of adjacent valid frames collected for the current label."""
        return self._count

    def update(self, prediction: Prediction) -> Prediction:
        # Low-confidence outputs are never evidence for a stable label.  This
        # prevents five old W frames plus one uncertain W-like frame from
        # continuing to display W.
        if not prediction.is_known or not prediction.label:
            self.reset()
            return Prediction(None, prediction.confidence, False)
        if prediction.label != self._candidate:
            self._candidate = prediction.label
            self._count = 0
            self._confidences.clear()
        self._count += 1
        self._confidences.append(prediction.confidence)
        if self._count < self.frames:
            return Prediction(None, prediction.confidence, False)
        confidence = float(np.mean(self._confidences))
        return Prediction(self._candidate, confidence, confidence >= self.threshold)
