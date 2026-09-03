"""Train/evaluate the final eight-class static-sign internship recogniser.

All available images for A, B, C, D, L, V, W and Y are assigned once to a
saved group-aware train/validation/test manifest *before* MediaPipe feature
extraction.  Exact duplicate image hashes are kept in one split, preventing
duplicate leakage into validation or the untouched test partition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.internship_pipeline import HandLandmarkExtractor, SUPPORTED_SIGNS

DATA = ROOT / "data" / "asl_alphabet" / "train"
MODELS = ROOT / "models"
SEED = 42
MODEL_STEM = "internship_static_signs_abcdlvwy_v2"
MANIFEST_PATH = MODELS / f"{MODEL_STEM}_split_manifest.json"
TRAIN_AUGMENT_RATIO = 0.30


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split_manifest() -> dict[str, object]:
    """Split duplicate groups within each label: 80% train / 10% / 10%."""
    records: list[dict[str, str]] = []
    for label in SUPPORTED_SIGNS:
        files = sorted((DATA / label).glob("*.jpg"))
        if not files:
            raise RuntimeError(f"No JPG images available for class {label}.")
        print(f"{label}: {len(files)} available images", flush=True)
        records.extend(
            {"path": str(path.relative_to(ROOT)), "label": label, "sha256": file_hash(path)}
            for path in files
        )

    # A pixel-identical file with two different labels would be corrupt data.
    labels_by_hash: dict[str, set[str]] = {}
    for record in records:
        labels_by_hash.setdefault(record["sha256"], set()).add(record["label"])
    conflicts = {digest: labels for digest, labels in labels_by_hash.items() if len(labels) > 1}
    if conflicts:
        raise RuntimeError(f"Found {len(conflicts)} duplicate hashes assigned to different labels.")

    split_records: dict[str, list[dict[str, str]]] = {"train": [], "validation": [], "test": []}
    for label in SUPPORTED_SIGNS:
        class_records = [record for record in records if record["label"] == label]
        groups = np.asarray([record["sha256"] for record in class_records])
        indices = np.arange(len(class_records))
        train_index, held_index = next(
            GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + ord(label)).split(indices, groups=groups)
        )
        validation_relative, test_relative = next(
            GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED + 100 + ord(label)).split(
                held_index, groups=groups[held_index]
            )
        )
        split_records["train"].extend(class_records[index] for index in train_index)
        split_records["validation"].extend(class_records[held_index[index]] for index in validation_relative)
        split_records["test"].extend(class_records[held_index[index]] for index in test_relative)

    hash_sets = {name: {record["sha256"] for record in entries} for name, entries in split_records.items()}
    if hash_sets["train"] & hash_sets["validation"] or hash_sets["train"] & hash_sets["test"] or hash_sets["validation"] & hash_sets["test"]:
        raise RuntimeError("Duplicate leakage detected while creating the split manifest.")
    return {
        "seed": SEED,
        "labels": SUPPORTED_SIGNS,
        "split_policy": "GroupShuffleSplit by SHA-256 image hash, 80/10/10 per class",
        "available_per_class": {label: sum(record["label"] == label for record in records) for label in SUPPORTED_SIGNS},
        "unique_hashes": len(labels_by_hash),
        "duplicate_images": len(records) - len(labels_by_hash),
        "splits": split_records,
    }


def split_paths() -> dict[str, list[tuple[str, str]]]:
    """Load the immutable manifest; create it only for a new training run."""
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("labels") != SUPPORTED_SIGNS:
            raise RuntimeError(f"Existing manifest labels do not match {SUPPORTED_SIGNS}.")
    else:
        manifest = build_split_manifest()
        MODELS.mkdir(exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        name: [(str(ROOT / record["path"]), record["label"]) for record in manifest["splits"][name]]
        for name in ("train", "validation", "test")
    }


def extract(split: list[tuple[str, str]], name: str) -> tuple[np.ndarray, np.ndarray, int]:
    extractor = HandLandmarkExtractor()
    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0
    try:
        for number, (path, label) in enumerate(split, start=1):
            image = cv2.imread(path)
            result = extractor.extract(image) if image is not None else None
            if result is None:
                skipped += 1
                continue
            feature, _ = result
            features.append(feature)
            labels.append(label)
            if number % 250 == 0:
                print(f"{name}: {number}/{len(split)} processed, {len(labels)} usable", flush=True)
    finally:
        extractor.close()
    if set(labels) != set(SUPPORTED_SIGNS):
        raise RuntimeError(f"{name} has incomplete detected-class coverage: {sorted(set(labels))}")
    return np.stack(features), np.asarray(labels), skipped


def augment_training_features(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add only mild landmark-space augmentation to the training partition."""
    rng = np.random.default_rng(SEED)
    count = int(round(len(features) * TRAIN_AUGMENT_RATIO))
    selected = rng.choice(len(features), size=count, replace=False)
    points = features[selected].reshape(-1, 21, 3).copy()
    angles = rng.normal(0.0, np.deg2rad(3.0), size=count)
    scales = rng.normal(1.0, 0.025, size=count)
    cosine, sine = np.cos(angles), np.sin(angles)
    x, y = points[:, :, 0].copy(), points[:, :, 1].copy()
    points[:, :, 0] = scales[:, None] * (cosine[:, None] * x - sine[:, None] * y)
    points[:, :, 1] = scales[:, None] * (sine[:, None] * x + cosine[:, None] * y)
    points += rng.normal(0.0, 0.004, size=points.shape).astype(np.float32)
    points[:, 0, :] = 0.0  # retain the wrist-origin convention.
    return np.vstack((features, points.reshape(count, -1))), np.concatenate((labels, labels[selected]))


def main() -> None:
    random.seed(SEED); np.random.seed(SEED)
    splits = split_paths()
    extracted = {name: extract(items, name) for name, items in splits.items()}
    x_train, y_train, skipped_train = extracted["train"]
    x_val, y_val, skipped_val = extracted["validation"]
    x_test, y_test, skipped_test = extracted["test"]
    x_train_augmented, y_train_augmented = augment_training_features(x_train, y_train)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=SEED)),
    ])
    model.fit(x_train_augmented, y_train_augmented)
    validation_accuracy = float(model.score(x_val, y_val))
    predicted = model.predict(x_test)
    test_accuracy = float(np.mean(predicted == y_test))
    report = classification_report(y_test, predicted, labels=SUPPORTED_SIGNS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predicted, labels=SUPPORTED_SIGNS)
    per_class_accuracy = {
        label: {
            "test_samples": int(np.sum(y_test == label)),
            "correct": int(np.sum((y_test == label) & (predicted == label))),
            "accuracy": float(np.mean(predicted[y_test == label] == label)),
        }
        for label in SUPPORTED_SIGNS
    }
    MODELS.mkdir(exist_ok=True)
    model_path = MODELS / f"{MODEL_STEM}.joblib"
    labels_path = MODELS / f"{MODEL_STEM}_labels.json"
    metrics_path = MODELS / f"{MODEL_STEM}_metrics.json"
    joblib.dump(model, model_path)
    labels_path.write_text(json.dumps(SUPPORTED_SIGNS, indent=2), encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metrics = {
        "labels": SUPPORTED_SIGNS,
        "available_per_class": manifest["available_per_class"],
        "all_valid_images_used": True,
        "split_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "split_policy": manifest["split_policy"],
        "duplicate_images": manifest["duplicate_images"],
        "split_before_detection": {name: len(items) for name, items in splits.items()},
        "usable_after_detection": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "skipped_no_hand": {"train": skipped_train, "validation": skipped_val, "test": skipped_test},
        "training_augmentation": f"{TRAIN_AUGMENT_RATIO:.0%} mild landmark rotation/scale/jitter on training samples only",
        "training_samples_after_augmentation": len(y_train_augmented),
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "per_class_accuracy": per_class_accuracy,
        "model": "StandardScaler + RBF SVC on canonical MediaPipe hand landmarks",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODELS / f"{MODEL_STEM}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (MODELS / f"{MODEL_STEM}_confusion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["actual/predicted", *SUPPORTED_SIGNS])
        for label, row in zip(SUPPORTED_SIGNS, matrix): writer.writerow([label, *row])
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
