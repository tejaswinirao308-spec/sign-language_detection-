"""Train a small, auditable five-word ASL recogniser from video clips.

The public source archive is intentionally kept outside the repository.  This
script reads only these verified labels from its extracted ``Video_Data``
folder: hello, yes, no, help, stop.  A single MediaPipe two-hand landmark
preprocessor is shared with runtime inference through ``src.vision``.

Splits are group-safe: filename families such as ``12.mp4``, ``12 (2).mp4``
and ``12 (3).mp4`` are kept wholly in one split.  This avoids evaluating on a
nearby re-recording of a training gesture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision import HandDetector, sequence_features


CLASSES = ("Hello", "Yes", "No", "Help", "Stop")
SOURCE_DIRS = {label: label.lower() for label in CLASSES}
SEQUENCE_LENGTH = 16
FEATURES_PER_FRAME = 126
MIN_DETECTED_FRAMES = 10
SEED = 20260902


def clip_group(path: Path) -> str:
    """Return the conservative group shared by same-numbered file variants."""
    match = re.match(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot derive a group id from {path.name}")
    return match.group(1)


def clip_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_sequence(path: Path, detector: HandDetector) -> tuple[np.ndarray, int, int]:
    """Uniformly sample one video into the shared 30 x 126 landmark tensor."""
    capture = cv2.VideoCapture(str(path))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32), 0, total_frames
    positions = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH, dtype=np.int32)
    sequence = np.zeros((SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)
    detected = 0
    for output_index, position in enumerate(positions):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        observations = detector.detect(frame_bgr)
        if observations:
            sequence[output_index] = sequence_features(observations)
            detected += 1
    capture.release()
    return sequence, detected, total_frames


def split_by_group(records: list[dict[str, object]]) -> dict[str, str]:
    """Make deterministic 70/15/15 class-stratified, group-disjoint splits."""
    rng = np.random.default_rng(SEED)
    assignments: dict[str, str] = {}
    by_label: dict[str, set[str]] = {label: set() for label in CLASSES}
    for record in records:
        by_label[str(record["label"])].add(str(record["group"]))
    for label, groups in by_label.items():
        ordered = np.asarray(sorted(groups), dtype=object)
        if len(ordered) < 30:
            raise RuntimeError(f"{label} has only {len(ordered)} independent capture groups; need at least 30.")
        rng.shuffle(ordered)
        train_count = round(len(ordered) * 0.70)
        validation_count = round(len(ordered) * 0.15)
        train_count = max(train_count, 1)
        validation_count = max(validation_count, 1)
        test_count = len(ordered) - train_count - validation_count
        if test_count < 1:
            raise RuntimeError(f"{label} cannot form a non-empty held-out test split.")
        for group in ordered[:train_count]:
            assignments[f"{label}:{group}"] = "train"
        for group in ordered[train_count:train_count + validation_count]:
            assignments[f"{label}:{group}"] = "validation"
        for group in ordered[train_count + validation_count:]:
            assignments[f"{label}:{group}"] = "test"
    return assignments


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURES_PER_FRAME), name="hand_landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.GaussianNoise(0.008)(x)
    x = tf.keras.layers.GRU(64, dropout=0.15, recurrent_dropout=0.0)(x)
    x = tf.keras.layers.Dense(48, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="word_probabilities")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="five_word_landmark_gru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True, help="Extracted Video_Data directory")
    parser.add_argument("--cache", type=Path, default=PROJECT_ROOT / "data" / "known_words_5_sequences.npz")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "known_words_hello_yes_no_help_stop_v1.keras")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--max-groups-per-class",
        type=int,
        default=30,
        help="Deterministic random capture-group cap per class (30 groups = 90 clips here).",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    if not args.videos.is_dir():
        raise SystemExit(f"Video directory not found: {args.videos}")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    manifest_path = args.output.with_name(args.output.stem + "_manifest.csv")
    labels_path = args.output.with_name(args.output.stem + "_labels.json")
    metrics_path = args.output.with_name(args.output.stem + "_metrics.json")
    report_path = args.output.with_name(args.output.stem + "_classification_report.json")
    confusion_path = args.output.with_name(args.output.stem + "_confusion_matrix.csv")
    checkpoint_path = args.output.with_name(args.output.stem + ".best.keras")

    records: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    for label in CLASSES:
        folder = args.videos / SOURCE_DIRS[label]
        clips = sorted(folder.glob("*.mp4"))
        if not clips:
            raise SystemExit(f"No clips found for {label}: {folder}")
        for clip in clips:
            sha = clip_sha256(clip)
            if sha in seen_hashes:
                continue
            seen_hashes.add(sha)
            records.append({"label": label, "path": str(clip), "group": clip_group(clip), "sha256": sha})

    raw_counts = Counter(str(item["label"]) for item in records)
    raw_groups = {label: len({str(item["group"]) for item in records if item["label"] == label}) for label in CLASSES}
    print("Unique clip audit:")
    for label in CLASSES:
        print(f"  {label:<5} clips={raw_counts[label]:3d} independent_groups={raw_groups[label]:2d}")
        if raw_counts[label] < 30 or raw_groups[label] < 30:
            raise SystemExit(f"Insufficient genuine data for {label}: {raw_counts[label]} clips / {raw_groups[label]} groups.")

    # The complete archive offers 60 capture groups per class.  A deterministic
    # 30-group subset is ample for this five-word CPU model (90 clips/class),
    # preserves a meaningful untouched test partition, and avoids spending an
    # hour repeatedly running MediaPipe on near-identical source conditions.
    if args.max_groups_per_class:
        selected_groups: dict[str, set[str]] = {}
        selector = np.random.default_rng(SEED)
        for label in CLASSES:
            groups = np.asarray(sorted({str(item["group"]) for item in records if item["label"] == label}), dtype=object)
            if len(groups) < args.max_groups_per_class:
                raise SystemExit(f"{label} has only {len(groups)} groups; cannot select {args.max_groups_per_class}.")
            selector.shuffle(groups)
            selected_groups[label] = set(groups[:args.max_groups_per_class].tolist())
        records = [record for record in records if str(record["group"]) in selected_groups[str(record["label"])]]
        print(f"Using deterministic subset: {args.max_groups_per_class} capture groups per class / {len(records)} clips total")

    assignments = split_by_group(records)
    for record in records:
        record["split"] = assignments[f"{record['label']}:{record['group']}"]

    cache_exists = args.cache.is_file() and not args.rebuild_cache
    if cache_exists:
        cached = np.load(args.cache, allow_pickle=False)
        X = cached["X"]
        detected_frames = cached["detected_frames"]
        total_frames = cached["total_frames"]
        cache_paths = cached["paths"].astype(str).tolist()
        if cache_paths != [str(record["path"]) for record in records]:
            raise RuntimeError("Sequence cache does not match the audited manifest; rerun with --rebuild-cache.")
    else:
        tensors: list[np.ndarray] = []
        detected_list: list[int] = []
        total_list: list[int] = []
        detector = HandDetector(static_image_mode=False, max_hands=2, detection_confidence=0.20, tracking_confidence=0.20)
        try:
            for index, record in enumerate(records, start=1):
                tensor, detected, total = sample_sequence(Path(str(record["path"])), detector)
                tensors.append(tensor)
                detected_list.append(detected)
                total_list.append(total)
                print(f"{index:>3}/{len(records)} {record['label']:<5} landmarks={detected:>2}/{SEQUENCE_LENGTH}", end="\r", flush=True)
        finally:
            detector.close()
        print()
        X = np.asarray(tensors, dtype=np.float32)
        detected_frames = np.asarray(detected_list, dtype=np.int16)
        total_frames = np.asarray(total_list, dtype=np.int16)
        np.savez_compressed(args.cache, X=X, detected_frames=detected_frames, total_frames=total_frames, paths=np.asarray([str(record["path"]) for record in records]))

    valid = detected_frames >= MIN_DETECTED_FRAMES
    for index, record in enumerate(records):
        record["detected_frames"] = int(detected_frames[index])
        record["source_frames"] = int(total_frames[index])
        record["valid"] = bool(valid[index])
    usable = [record for record in records if bool(record["valid"])]
    usable_counts = Counter(str(record["label"]) for record in usable)
    usable_groups = {label: len({str(record["group"]) for record in usable if record["label"] == label}) for label in CLASSES}
    print("Valid landmark-sequence audit:")
    for label in CLASSES:
        print(f"  {label:<5} clips={usable_counts[label]:3d} independent_groups={usable_groups[label]:2d}")
        # The acceptance requirement is at least 30 distinct valid clips per
        # word.  We also need enough capture groups for a 70/15/15 split; 15
        # is the practical lower bound (10/2/3 groups).  Requiring every one
        # of the 30 selected source groups to survive MediaPipe would wrongly
        # reject a valid dataset merely because a single three-take burst is
        # backlit or off-frame.
        if usable_counts[label] < 30 or usable_groups[label] < 15:
            raise SystemExit(f"Insufficient valid hand sequences for {label}: {usable_counts[label]} clips / {usable_groups[label]} groups.")

    # Re-split after detection filtering: a dropped capture never leaves its
    # neighbours in another split, and every resulting split is class-balanced.
    usable_assignments = split_by_group(usable)
    for record in usable:
        record["split"] = usable_assignments[f"{record['label']}:{record['group']}"]
    y_all = np.asarray([CLASSES.index(str(record["label"])) for record in usable], dtype=np.int32)
    x_all = X[valid]
    split_indices = {
        split: np.asarray([index for index, record in enumerate(usable) if record["split"] == split], dtype=np.int32)
        for split in ("train", "validation", "test")
    }
    if not all(len(indices) for indices in split_indices.values()):
        raise RuntimeError("A required split is empty after landmark validation.")

    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "split", "group", "path", "sha256", "detected_frames", "source_frames", "valid"])
        writer.writeheader()
        writer.writerows(usable)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    history = model.fit(
        x_all[split_indices["train"]], y_all[split_indices["train"]],
        validation_data=(x_all[split_indices["validation"]], y_all[split_indices["validation"]]),
        batch_size=16,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model = tf.keras.models.load_model(checkpoint_path, compile=False)
    probabilities = np.asarray(model(x_all[split_indices["test"]], training=False).numpy(), dtype=np.float32)
    predicted = probabilities.argmax(axis=1)
    actual = y_all[split_indices["test"]]
    accuracy = float(accuracy_score(actual, predicted))
    precision, recall, f1, support = precision_recall_fscore_support(actual, predicted, labels=np.arange(len(CLASSES)), zero_division=0)
    macro_f1 = float(f1_score(actual, predicted, average="macro", zero_division=0))
    report = classification_report(actual, predicted, labels=np.arange(len(CLASSES)), target_names=list(CLASSES), output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=np.arange(len(CLASSES)))

    model.save(args.output)
    checkpoint_path.unlink(missing_ok=True)
    counts_by_split = {
        split: {label: int(sum(1 for record in usable if record["split"] == split and record["label"] == label)) for label in CLASSES}
        for split in ("train", "validation", "test")
    }
    metrics = {
        "classes": list(CLASSES),
        "source": "Realtime_ASL_RasPi public training_data.tar; extracted five labels only",
        "source_video_root": str(args.videos),
        "preprocessing": f"OpenCV BGR frames -> MediaPipe Hands (two hand slots) -> handedness-canonical, wrist/scale-normalized 126-D features -> uniformly sampled {SEQUENCE_LENGTH}-frame sequence",
        "grouping": "same leading numeric filename group kept in one split",
        "sequence_length": SEQUENCE_LENGTH,
        "features_per_frame": FEATURES_PER_FRAME,
        "minimum_detected_frames": MIN_DETECTED_FRAMES,
        "confidence_threshold": 0.50,
        "raw_unique_clip_counts": {label: int(raw_counts[label]) for label in CLASSES},
        "raw_independent_group_counts": raw_groups,
        "usable_clip_counts": {label: int(usable_counts[label]) for label in CLASSES},
        "usable_independent_group_counts": usable_groups,
        "split_counts": counts_by_split,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "held_out_test_accuracy": accuracy,
        "held_out_macro_f1": macro_f1,
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "test_samples": int(support[i]), "accuracy": float(matrix[i, i] / max(1, matrix[i].sum()))}
            for i, label in enumerate(CLASSES)
        },
        "ready": bool(accuracy >= 0.70 and macro_f1 >= 0.65),
    }
    # Kept separate from the A–Z confidence cutoff: validation contains
    # correct dynamic Stop/No sequences in the 0.55–0.70 range.
    labels_path.write_text(json.dumps({"classes": list(CLASSES), "sequence_length": SEQUENCE_LENGTH, "features_per_frame": FEATURES_PER_FRAME, "threshold": 0.50}, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with confusion_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual/predicted", *CLASSES])
        for label, row in zip(CLASSES, matrix):
            writer.writerow([label, *row.tolist()])

    print(f"Saved model: {args.output}")
    print(f"Held-out test accuracy: {accuracy:.2%}")
    print(f"Held-out macro F1: {macro_f1:.4f}")
    print("READY" if metrics["ready"] else "NOT READY: model did not meet the 70% accuracy / 0.65 macro-F1 gate")


if __name__ == "__main__":
    main()
