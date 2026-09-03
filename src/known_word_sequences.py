"""Shared, auditable preprocessing for the four collected dynamic words.

The collector, trainer, uploaded-video path, and WebRTC processor all use this
module.  A sequence always consists of 30 chronological MediaPipe feature
frames, uniformly resampled to the 16 frames consumed by the classifier.
"""
from __future__ import annotations

import hashlib

import numpy as np


KNOWN_WORD_CLASSES = ("Hello", "Help", "Yes", "No")
RAW_SEQUENCE_FRAMES = 30
MODEL_SEQUENCE_FRAMES = 16
FEATURES_PER_FRAME = 126
MIN_VALID_FRAMES = 10


def empty_feature_frame() -> np.ndarray:
    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)


def resample_sequence(raw_features: np.ndarray) -> np.ndarray:
    """Uniformly resample a chronological 30 x 126 landmark window to 16 x 126.

    Missing MediaPipe detections are deliberately represented by zero rows.
    This is the same representation used live, so a short tracking miss does
    not silently change preprocessing between collection and inference.
    """
    raw_features = np.asarray(raw_features, dtype=np.float32)
    if raw_features.shape != (RAW_SEQUENCE_FRAMES, FEATURES_PER_FRAME):
        raise ValueError(
            f"Expected {(RAW_SEQUENCE_FRAMES, FEATURES_PER_FRAME)} raw features, got {raw_features.shape}."
        )
    positions = np.linspace(0, RAW_SEQUENCE_FRAMES - 1, MODEL_SEQUENCE_FRAMES, dtype=np.int32)
    return np.ascontiguousarray(raw_features[positions], dtype=np.float32)


def landmark_sequence_fingerprint(sequence: np.ndarray) -> str:
    """Stable content hash used to reject an exact re-recorded sequence."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.shape != (MODEL_SEQUENCE_FRAMES, FEATURES_PER_FRAME):
        raise ValueError(f"Expected {(MODEL_SEQUENCE_FRAMES, FEATURES_PER_FRAME)}, got {sequence.shape}.")
    # Landmark outputs have harmless floating-point jitter. Quantising makes
    # an immediately repeated capture identifiable without treating genuine
    # independent performances as identical.
    return hashlib.sha256(np.round(sequence, 4).tobytes()).hexdigest()
