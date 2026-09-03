"""Train the four-word webcam landmark model from genuine collected samples.

This script intentionally refuses to train from the four reference pictures.
It requires independent, MediaPipe-detected webcam recordings for each label
and creates stratified 70/15/15 train/validation/untouched-test partitions.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.known_word_sequences import (
    FEATURES_PER_FRAME,
    KNOWN_WORD_CLASSES,
    MIN_VALID_FRAMES,
    MODEL_SEQUENCE_FRAMES,
    landmark_sequence_fingerprint,
)


DATASET_ROOT = ROOT / "data" / "known_words_hello_help_yes_no"
DEFAULT_OUTPUT = ROOT / "models" / "known_words_hello_help_yes_no_v1.keras"
SEED = 20260902


def slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def load_records(dataset_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_fingerprints: set[str] = set()
    for label in KNOWN_WORD_CLASSES:
        directory = dataset_root / "raw" / slug(label)
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as sample:
                sequence = np.asarray(sample["sequence"], dtype=np.float32)
                stored_label = str(sample["label"].item())
                valid_frames = int(sample["valid_frames"].item())
                capture_id = str(sample["capture_id"].item())
            if stored_label != label:
                raise RuntimeError(f"Label mismatch in {path}: directory is {label}, file says {stored_label}.")
            if sequence.shape != (MODEL_SEQUENCE_FRAMES, FEATURES_PER_FRAME) or not np.isfinite(sequence).all():
                raise RuntimeError(f"Invalid landmark sequence: {path}")
            if valid_frames < MIN_VALID_FRAMES:
                continue
            fingerprint = landmark_sequence_fingerprint(sequence)
            if fingerprint in seen_fingerprints:
                raise RuntimeError(f"Duplicate landmark sequence found: {path}")
            seen_fingerprints.add(fingerprint)
            records.append({"label": label, "path": path, "capture_id": capture_id, "sequence": sequence, "valid_frames": valid_frames, "fingerprint": fingerprint})
    return records


def stratified_split(records: list[dict[str, object]]) -> dict[str, list[int]]:
    """Split independent capture IDs, never frames, into 70/15/15 partitions."""
    rng = np.random.default_rng(SEED)
    groups: dict[str, list[int]] = {label: [] for label in KNOWN_WORD_CLASSES}
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        capture_id = str(record["capture_id"])
        if capture_id in seen_ids:
            raise RuntimeError(f"Capture ID appears more than once: {capture_id}")
        seen_ids.add(capture_id)
        groups[str(record["label"])].append(index)
    split = {"train": [], "validation": [], "test": []}
    for label, indices in groups.items():
        if len(indices) < 30:
            raise RuntimeError(f"{label} has {len(indices)} valid independent samples; collect at least 30 before training.")
        shuffled = np.asarray(indices, dtype=np.int32)
        rng.shuffle(shuffled)
        train_count = max(1, round(len(shuffled) * 0.70))
        validation_count = max(1, round(len(shuffled) * 0.15))
        test_count = len(shuffled) - train_count - validation_count
        if test_count < 1:
            raise RuntimeError(f"{label} cannot form a held-out test split.")
        split["train"].extend(shuffled[:train_count].tolist())
        split["validation"].extend(shuffled[train_count:train_count + validation_count].tolist())
        split["test"].extend(shuffled[train_count + validation_count:].tolist())
    return {name: sorted(indices) for name, indices in split.items()}


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(MODEL_SEQUENCE_FRAMES, FEATURES_PER_FRAME), name="hand_landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    # Applied only while fitting. It is modest landmark jitter, not a fake
    # sample generator, and helps the CPU model tolerate natural tracking noise.
    x = tf.keras.layers.GaussianNoise(0.008)(x)
    x = tf.keras.layers.GRU(64, dropout=0.15)(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(KNOWN_WORD_CLASSES), activation="softmax", name="word_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="collected_four_word_landmark_gru")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(SEED)
    records = load_records(args.dataset)
    counts = Counter(str(record["label"]) for record in records)
    print("Valid independent sequence counts:")
    for label in KNOWN_WORD_CLASSES:
        print(f"  {label}: {counts[label]}")
    # Failing before model creation is intentional: it prevents a misleading
    # tiny-data checkpoint from replacing a real validated model.
    split_indices = stratified_split(records)
    X = np.asarray([record["sequence"] for record in records], dtype=np.float32)
    y = np.asarray([KNOWN_WORD_CLASSES.index(str(record["label"])) for record in records], dtype=np.int32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_name(args.output.stem + ".best.keras")
    model = build_model()
    history = model.fit(
        X[split_indices["train"]], y[split_indices["train"]],
        validation_data=(X[split_indices["validation"]], y[split_indices["validation"]]),
        batch_size=16,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_accuracy", mode="max", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
        ],
        verbose=2,
    )
    model = tf.keras.models.load_model(checkpoint, compile=False)
    probabilities = np.asarray(model(X[split_indices["test"]], training=False).numpy(), dtype=np.float32)
    predicted = probabilities.argmax(axis=1)
    actual = y[split_indices["test"]]
    accuracy = float(accuracy_score(actual, predicted))
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=np.arange(len(KNOWN_WORD_CLASSES)), zero_division=0)
    macro_f1 = float(f1_score(actual, predicted, average="macro", zero_division=0))
    matrix = confusion_matrix(actual, predicted, labels=np.arange(len(KNOWN_WORD_CLASSES)))
    report = classification_report(actual, predicted, labels=np.arange(len(KNOWN_WORD_CLASSES)), target_names=list(KNOWN_WORD_CLASSES), output_dict=True, zero_division=0)
    model.save(args.output)
    checkpoint.unlink(missing_ok=True)
    labels_path = args.output.with_name(args.output.stem + "_labels.json")
    metrics_path = args.output.with_name(args.output.stem + "_metrics.json")
    report_path = args.output.with_name(args.output.stem + "_classification_report.json")
    matrix_path = args.output.with_name(args.output.stem + "_confusion_matrix.csv")
    manifest_path = args.output.with_name(args.output.stem + "_manifest.csv")
    split_of_index = {index: split for split, indices in split_indices.items() for index in indices}
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "split", "capture_id", "path", "valid_frames", "fingerprint"])
        writer.writeheader()
        writer.writerows({key: (str(value) if key == "path" else value) for key, value in record.items() if key != "sequence"} | {"split": split_of_index[index]} for index, record in enumerate(records))
    split_counts = {
        split: {label: sum(1 for index in indices if records[index]["label"] == label) for label in KNOWN_WORD_CLASSES}
        for split, indices in split_indices.items()
    }
    metrics = {
        "classes": list(KNOWN_WORD_CLASSES),
        "dataset": str(args.dataset),
        "preprocessing": "BGR webcam frame -> MediaPipe Hands -> handedness-canonical two-hand 126-D features -> chronological 30 frame window -> uniform 16 frame sample",
        "sequence_length": MODEL_SEQUENCE_FRAMES,
        "features_per_frame": FEATURES_PER_FRAME,
        "minimum_valid_frames": MIN_VALID_FRAMES,
        "split_policy": "independent capture IDs only; 70/15/15 stratified train/validation/untouched test",
        "split_counts": split_counts,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "held_out_test_accuracy": accuracy,
        "held_out_macro_f1": macro_f1,
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "test_samples": int(support[i]), "accuracy": float(matrix[i, i] / max(1, matrix[i].sum()))} for i, label in enumerate(KNOWN_WORD_CLASSES)},
        "ready": bool(accuracy >= 0.70 and macro_f1 >= 0.65),
    }
    labels_path.write_text(json.dumps({"classes": list(KNOWN_WORD_CLASSES), "sequence_length": MODEL_SEQUENCE_FRAMES, "features_per_frame": FEATURES_PER_FRAME, "threshold": 0.50}, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with matrix_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *KNOWN_WORD_CLASSES])
        for label, row in zip(KNOWN_WORD_CLASSES, matrix):
            writer.writerow([label, *row.tolist()])
    print(f"Saved model: {args.output}")
    print(f"Held-out test accuracy: {accuracy:.2%}")
    print(f"Held-out macro F1: {macro_f1:.4f}")
    print("READY" if metrics["ready"] else "NOT READY: held-out metrics did not meet the 70% accuracy / 0.65 macro-F1 gate")


if __name__ == "__main__":
    main()
