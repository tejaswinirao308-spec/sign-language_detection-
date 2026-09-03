"""Collect real, operator-confirmed landmark sequences for Hello/Help/Yes/No.

No image labels are generated and no reference screenshot is copied into the
dataset.  The operator must select one word and explicitly confirm that every
recorded performance is that word.  Press R for one 2-second sequence, or A to
let the collector capture the requested number of separate takes with a pause
between them.  Only MediaPipe landmarks and audit metadata are saved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.known_word_sequences import (
    FEATURES_PER_FRAME,
    KNOWN_WORD_CLASSES,
    MIN_VALID_FRAMES,
    RAW_SEQUENCE_FRAMES,
    empty_feature_frame,
    landmark_sequence_fingerprint,
    resample_sequence,
)
from src.vision import HandDetector, sequence_features


DATASET_ROOT = ROOT / "data" / "known_words_hello_help_yes_no"


def slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def sample_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid.uuid4().hex[:10]}"


def setup_layout() -> None:
    for label in KNOWN_WORD_CLASSES:
        (DATASET_ROOT / "raw" / slug(label)).mkdir(parents=True, exist_ok=True)
    (DATASET_ROOT / "manifests").mkdir(parents=True, exist_ok=True)
    classes_file = DATASET_ROOT / "classes.json"
    if not classes_file.exists():
        classes_file.write_text(
            json.dumps(
                {
                    "classes": list(KNOWN_WORD_CLASSES),
                    "raw_sequence_frames": RAW_SEQUENCE_FRAMES,
                    "model_sequence_frames": 16,
                    "features_per_frame": FEATURES_PER_FRAME,
                    "minimum_valid_frames": MIN_VALID_FRAMES,
                    "policy": "Only operator-confirmed webcam performances; reference images are never training samples.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def load_fingerprints() -> dict[str, dict[str, str]]:
    path = DATASET_ROOT / "manifests" / "fingerprints.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_sample(label: str, raw_features: np.ndarray, valid_frames: int, fps: float) -> bool:
    sequence = resample_sequence(raw_features)
    fingerprint = landmark_sequence_fingerprint(sequence)
    fingerprints = load_fingerprints()
    if fingerprint in fingerprints:
        print(f"Skipped exact duplicate of {fingerprints[fingerprint]['sample']}.")
        return False
    capture_id = sample_id()
    target = DATASET_ROOT / "raw" / slug(label) / f"{capture_id}.npz"
    np.savez_compressed(
        target,
        sequence=sequence,
        raw_features=np.asarray(raw_features, dtype=np.float32),
        label=np.asarray(label),
        valid_frames=np.asarray(valid_frames, dtype=np.int16),
        capture_id=np.asarray(capture_id),
        preprocessing_version=np.asarray("mediapipe_two_hand_30_to_16_v1"),
    )
    fingerprints[fingerprint] = {"label": label, "sample": str(target.relative_to(DATASET_ROOT))}
    (DATASET_ROOT / "manifests" / "fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True), encoding="utf-8"
    )
    record = {
        "capture_id": capture_id,
        "label": label,
        "file": str(target.relative_to(DATASET_ROOT)),
        "valid_frames": valid_frames,
        "fps": fps,
        "fingerprint": fingerprint,
        "operator_confirmed": True,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (DATASET_ROOT / "manifests" / "metadata.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Saved {label} sample: {target.name} ({valid_frames}/{RAW_SEQUENCE_FRAMES} detected frames)")
    return True


def purge_label(label: str) -> int:
    """Delete one label's collected landmark samples and their audit entries."""
    target = DATASET_ROOT / "raw" / slug(label)
    files = list(target.glob("*.npz"))
    for path in files:
        path.unlink()
    metadata_path = DATASET_ROOT / "manifests" / "metadata.jsonl"
    if metadata_path.exists():
        kept = [line for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("label") != label]
        metadata_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    fingerprints = load_fingerprints()
    kept_fingerprints = {key: value for key, value in fingerprints.items() if value.get("label") != label}
    (DATASET_ROOT / "manifests" / "fingerprints.json").write_text(
        json.dumps(kept_fingerprints, indent=2, sort_keys=True), encoding="utf-8"
    )
    return len(files)


def capture_take(camera: cv2.VideoCapture, detector: HandDetector, fps: float, label: str) -> tuple[np.ndarray, int] | None:
    frames: list[np.ndarray] = []
    valid_frames = 0
    while len(frames) < RAW_SEQUENCE_FRAMES:
        ok, image_bgr = camera.read()
        if not ok:
            return None
        observations = detector.detect(image_bgr)
        if observations:
            frames.append(sequence_features(observations))
            valid_frames += 1
        else:
            frames.append(empty_feature_frame())
        preview = image_bgr.copy()
        cv2.putText(preview, f"{label}: recording {len(frames)}/{RAW_SEQUENCE_FRAMES}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.imshow("Known-word landmark collector", preview)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            return None
    return np.asarray(frames, dtype=np.float32), valid_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect genuine MediaPipe word-sign sequences.")
    parser.add_argument("--label", choices=KNOWN_WORD_CLASSES)
    parser.add_argument("--samples", type=int, default=40, help="Target recordings in this collection session (30–50 recommended).")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--confirm", action="store_true", help="Required: confirm the performer will make exactly --label for every take.")
    parser.add_argument("--automatic", action="store_true", help="After pressing A, capture takes automatically with a short transition pause.")
    parser.add_argument("--status", action="store_true", help="Create the four class folders and print current valid sample counts without opening a camera.")
    parser.add_argument("--purge-label", choices=KNOWN_WORD_CLASSES, help="Permanently remove one label's collected sequences and metadata; requires --confirm.")
    args = parser.parse_args()
    setup_layout()
    if args.status:
        for label in KNOWN_WORD_CLASSES:
            count = sum(1 for path in (DATASET_ROOT / "raw" / slug(label)).glob("*.npz") if path.is_file())
            print(f"{label}: {count} landmark sequences")
        return
    if args.purge_label:
        if not args.confirm:
            raise SystemExit("Refusing to remove collected data without --confirm.")
        removed = purge_label(args.purge_label)
        print(f"Removed {removed} collected {args.purge_label} sequences and their manifest entries.")
        return
    if args.label is None:
        parser.error("--label is required unless --status is used")
    if not args.confirm:
        raise SystemExit("Refusing to label data without --confirm.")
    if not 1 <= args.samples <= 100:
        raise SystemExit("--samples must be between 1 and 100.")
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit("Could not open the webcam. Stop Streamlit/WebRTC before collecting.")
    detector = HandDetector(static_image_mode=False, max_hands=2, detection_confidence=0.20, tracking_confidence=0.20)
    saved = 0
    automatic = False
    next_take_at = 0.0
    try:
        while saved < args.samples:
            ok, image_bgr = camera.read()
            if not ok:
                raise RuntimeError("Could not read a camera frame.")
            preview = image_bgr.copy()
            text = f"{args.label} | saved {saved}/{args.samples} | R: record one  A: auto  Q: quit"
            cv2.putText(preview, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
            cv2.imshow("Known-word landmark collector", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("a"), ord("A")) and args.automatic:
                automatic = True
                next_take_at = time.monotonic() + 1.0
            record_now = key in (ord("r"), ord("R")) or (automatic and time.monotonic() >= next_take_at)
            if not record_now:
                continue
            captured = capture_take(camera, detector, args.fps, args.label)
            if captured is None:
                automatic = False
                continue
            features, valid_frames = captured
            if valid_frames < MIN_VALID_FRAMES:
                print(f"Skipped: only {valid_frames}/{RAW_SEQUENCE_FRAMES} frames contained a detected hand.")
            elif save_sample(args.label, features, valid_frames, args.fps):
                saved += 1
            next_take_at = time.monotonic() + 0.8
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()
    print(f"Collection finished: {saved} new {args.label} sequences.")


if __name__ == "__main__":
    main()
