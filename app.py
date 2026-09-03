"""A focused Streamlit sign-language internship project.

The validated A–Z/static model and the separately validated temporal known-word
model are deliberately isolated: neither can overwrite the other's labels or
preprocessing pipeline.
"""

from __future__ import annotations

import json
import threading
import tempfile
from collections import Counter, deque
from datetime import datetime, time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    WEBRTC_AVAILABLE = True
except ImportError:
    av = None
    WEBRTC_AVAILABLE = False

from src.internship_pipeline import InternshipSignPipeline, SignResult
from src.known_word_sequences import MIN_VALID_FRAMES, RAW_SEQUENCE_FRAMES, empty_feature_frame, resample_sequence
from src.vision import HandDetector, TemporalWordClassifier, sequence_features


ROOT = Path(__file__).resolve().parent
# The only promoted static-sign checkpoint.  The incomplete WLASL word model
# is deliberately not loaded by this application.
MODEL_PATH = ROOT / "models" / "internship_static_signs_abcdlvwy_v1.joblib"
LABELS_PATH = ROOT / "models" / "internship_static_signs_abcdlvwy_v1_labels.json"
METRICS_PATH = ROOT / "models" / "internship_static_signs_abcdlvwy_v1_metrics.json"
WORD_MODEL_PATH = ROOT / "models" / "known_words_hello_yes_no_help_stop_v1.keras"
WORD_LABELS_PATH = ROOT / "models" / "known_words_hello_yes_no_help_stop_v1_labels.json"
WORD_METRICS_PATH = ROOT / "models" / "known_words_hello_yes_no_help_stop_v1_metrics.json"
COLLECTED_WORD_MODEL_PATH = ROOT / "models" / "known_words_hello_help_yes_no_v1.keras"
COLLECTED_WORD_LABELS_PATH = ROOT / "models" / "known_words_hello_help_yes_no_v1_labels.json"
COLLECTED_WORD_METRICS_PATH = ROOT / "models" / "known_words_hello_help_yes_no_v1_metrics.json"
CONFIDENCE_THRESHOLD = 0.70
# This sequence model's held-out correct scores include a genuine Stop at
# 54.5%; keep its independently calibrated threshold separate from A–Z.
KNOWN_WORD_CONFIDENCE_THRESHOLD = 0.50
REQUIRED_STABLE_FRAMES = 6

st.set_page_config(page_title="Sign Language Detection", page_icon="🤟", layout="wide")


def prediction_active() -> bool:
    """Keep the required final operating period: 6 PM until 10 PM local time."""
    now = datetime.now().time()
    return time(18, 0) <= now < time(22, 0)


def current_time_text() -> str:
    return datetime.now().strftime("%I:%M:%S %p")


@st.cache_resource(show_spinner="Loading the trained sign model…")
def load_pipeline() -> InternshipSignPipeline:
    return InternshipSignPipeline(
        model_path=MODEL_PATH,
        labels_path=LABELS_PATH,
        threshold=CONFIDENCE_THRESHOLD,
    )


def known_word_artifacts() -> tuple[Path, Path, Path]:
    """Prefer the locally collected four-word checkpoint once it exists.

    The existing verified five-word model remains a fallback so adding the
    collection workflow never breaks the current Known Words feature.
    """
    if COLLECTED_WORD_MODEL_PATH.exists() and COLLECTED_WORD_LABELS_PATH.exists() and COLLECTED_WORD_METRICS_PATH.exists():
        return COLLECTED_WORD_MODEL_PATH, COLLECTED_WORD_LABELS_PATH, COLLECTED_WORD_METRICS_PATH
    return WORD_MODEL_PATH, WORD_LABELS_PATH, WORD_METRICS_PATH


@st.cache_resource(show_spinner="Loading the validated known-word model…")
def load_word_classifier(model_path: Path, labels_path: Path) -> TemporalWordClassifier:
    return TemporalWordClassifier(
        model_path=model_path,
        labels_path=labels_path,
        threshold=KNOWN_WORD_CONFIDENCE_THRESHOLD,
    )


def load_metrics_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_metrics() -> dict:
    return load_metrics_file(METRICS_PATH)


def model_is_ready(pipeline: InternshipSignPipeline, metrics: dict) -> bool:
    accuracy = metrics.get("end_to_end_test_accuracy", metrics.get("test_accuracy"))
    return pipeline.available and isinstance(accuracy, (int, float)) and accuracy >= 0.80


def word_model_is_ready(classifier: TemporalWordClassifier, metrics: dict) -> bool:
    accuracy = metrics.get("held_out_test_accuracy")
    macro_f1 = metrics.get("held_out_macro_f1")
    return (
        classifier.available
        and metrics.get("ready") is True
        and isinstance(accuracy, (int, float))
        and isinstance(macro_f1, (int, float))
        and accuracy >= 0.70
        and macro_f1 >= 0.65
    )


def draw_result(image_bgr: np.ndarray, result: SignResult, stable_label: str | None = None) -> np.ndarray:
    """Draw the same hand box produced by the shared inference pipeline."""
    annotated = image_bgr.copy()
    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        color = (40, 190, 80) if result.recognized else (0, 170, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if stable_label:
            caption = f"Detected: {stable_label}"
        elif result.recognized:
            caption = f"Detected: {result.label} ({result.confidence * 100:.1f}%)"
        else:
            caption = f"Unknown Sign ({result.confidence * 100:.1f}%)"
        cv2.putText(
            annotated,
            caption,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def render_prediction(result: SignResult) -> None:
    st.subheader("Prediction result")
    # A missing MediaPipe hand has no classifier probability. Keep it distinct
    # from a genuine low-confidence model prediction.
    if result.bbox is None:
        st.warning("No hand detected. Please upload a clear sign-language image.")
        return
    if not result.recognized:
        st.warning("Detected Sign: Unknown / Please try again")
        st.write(f"Highest confidence: {result.confidence * 100:.1f}%")
    else:
        st.success(f"Detected Sign: {result.label}")
        st.write(f"Confidence: {result.confidence * 100:.1f}%")
    if result.top5:
        st.caption("Top predictions: " + " · ".join(f"{label} {score * 100:.1f}%" for label, score in result.top5))


class FrameSmoother:
    """Confirms a label only after several consecutive real model predictions."""

    def __init__(self, required_frames: int = REQUIRED_STABLE_FRAMES) -> None:
        self.required_frames = required_frames
        self.labels: deque[str] = deque(maxlen=required_frames)
        self.stable_label: str | None = None

    def update(self, result: SignResult) -> str | None:
        if not result.recognized:
            self.labels.clear()
            self.stable_label = None
            return None
        self.labels.append(result.label)
        if len(self.labels) == self.required_frames and len(set(self.labels)) == 1:
            self.stable_label = result.label
        elif self.stable_label is not None and result.label != self.stable_label:
            self.stable_label = None
        return self.stable_label


class VideoProcessor:
    """WebRTC owns the camera; each received frame calls the shared pipeline."""

    def __init__(self, pipeline: InternshipSignPipeline) -> None:
        self.pipeline = pipeline
        self.smoother = FrameSmoother()
        self.lock = threading.Lock()
        self.frames_received = 0
        self.hands_detected = 0
        self.last_result = SignResult(
            label=None,
            confidence=0.0,
            recognized=False,
            bbox=None,
            reason="waiting",
        )
        self.stable_label: str | None = None

    def recv(self, frame):  # type: ignore[no-untyped-def]
        image_bgr = frame.to_ndarray(format="bgr24")
        result = self.pipeline.predict(image_bgr)
        stable = self.smoother.update(result)
        annotated = draw_result(image_bgr, result, stable)
        with self.lock:
            self.frames_received += 1
            if result.bbox is not None:
                self.hands_detected += 1
            self.last_result = result
            self.stable_label = stable
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def status(self) -> tuple[int, int, SignResult, str | None]:
        with self.lock:
            return self.frames_received, self.hands_detected, self.last_result, self.stable_label


class KnownWordVideoProcessor:
    """A separate temporal processor for Hello/Yes/No/Help/Stop.

    Each WebRTC frame follows the same BGR -> MediaPipe two-hand landmarks ->
    handedness-canonical 126-D feature step used by ``train_known_words_5``.
    A rolling 30-frame live window is uniformly sampled to the model's 16
    landmark frames, exactly matching the training-video sampler.  An image is
    therefore never misrepresented as a dynamic word prediction.
    """

    def __init__(self, classifier: TemporalWordClassifier) -> None:
        self.classifier = classifier
        self.detector = HandDetector(
            static_image_mode=False,
            max_hands=2,
            detection_confidence=0.20,
            tracking_confidence=0.20,
        )
        self.capture_frames = RAW_SEQUENCE_FRAMES
        self.frames: deque[np.ndarray] = deque(maxlen=self.capture_frames)
        self.valid_frame_flags: deque[bool] = deque(maxlen=self.capture_frames)
        self.smoother = FrameSmoother()
        self.lock = threading.Lock()
        self.frames_received = 0
        self.hands_detected = 0
        self.last_result = SignResult(None, 0.0, False, None, reason="waiting for a sign sequence")
        self.stable_label: str | None = None
        self.no_hand_streak = 0
        self.last_bbox: tuple[int, int, int, int] | None = None

    @staticmethod
    def _combined_bbox(observations: list) -> tuple[int, int, int, int] | None:
        if not observations:
            return None
        return (
            min(observation.bbox[0] for observation in observations),
            min(observation.bbox[1] for observation in observations),
            max(observation.bbox[2] for observation in observations),
            max(observation.bbox[3] for observation in observations),
        )

    def recv(self, frame):  # type: ignore[no-untyped-def]
        image_bgr = frame.to_ndarray(format="bgr24")
        observations = self.detector.detect(image_bgr)
        bbox = self._combined_bbox(observations)
        # Training uses zero vectors for individual sampled video frames where
        # MediaPipe sees no hand.  Preserve that exact behaviour live instead
        # of discarding an otherwise valid temporal gesture after one brief
        # tracking miss (which used to prevent several genuine signs reaching
        # the classifier at all).
        self.frames.append(sequence_features(observations) if observations else np.zeros(126, dtype=np.float32))
        self.valid_frame_flags.append(bool(observations))
        if observations:
            self.no_hand_streak = 0
            self.last_bbox = bbox
        else:
            self.no_hand_streak += 1

        valid_frames = sum(self.valid_frame_flags)
        if self.no_hand_streak >= self.capture_frames:
            self.frames.clear()
            self.valid_frame_flags.clear()
            self.last_bbox = None
            self.smoother.update(SignResult(None, 0.0, False, None, reason="No hand detected"))
            result = SignResult(None, 0.0, False, None, reason="No hand detected")
            stable = None
        elif len(self.frames) < self.capture_frames or valid_frames < 10:
            result = SignResult(None, 0.0, False, bbox or self.last_bbox, reason="Collecting a 30-frame sign window")
            stable = None
        else:
            live_window = np.asarray(self.frames, dtype=np.float32)
            prediction = self.classifier.predict_sequence(resample_sequence(live_window))
            result = SignResult(
                label=prediction.label if prediction.is_known else None,
                confidence=prediction.confidence,
                recognized=prediction.is_known,
                bbox=bbox or self.last_bbox,
                top5=prediction.top_predictions,
                reason=None if prediction.is_known else "Unknown Sign",
            )
            stable = self.smoother.update(result)
        annotated = draw_result(image_bgr, result, stable)
        with self.lock:
            self.frames_received += 1
            if observations:
                self.hands_detected += 1
            self.last_result = result
            self.stable_label = stable
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def status(self) -> tuple[int, int, SignResult, str | None]:
        with self.lock:
            return self.frames_received, self.hands_detected, self.last_result, self.stable_label


def upload_tab(pipeline: InternshipSignPipeline, ready: bool, active: bool) -> None:
    st.subheader("Upload image")
    uploaded = st.file_uploader(
        "Browse / Upload Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image containing one supported hand sign.",
    )
    if uploaded is None:
        st.info("Upload a JPG, JPEG, or PNG image to begin.")
        return

    raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("The selected file could not be read as an image.")
        return

    preview_col, result_col = st.columns([1, 1])
    with preview_col:
        st.markdown("**Uploaded image**")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), width=460)
    with result_col:
        if not active:
            st.info("Sign Language Detection is available only between 6:00 PM and 10:00 PM.")
            return
        if not ready:
            st.error("The selected sign model is not yet validated. Complete training before prediction.")
            return
        result = pipeline.predict(image_bgr)
        render_prediction(result)
    if active and ready:
        st.markdown("**Detected hand region**")
        st.image(cv2.cvtColor(draw_result(image_bgr, result), cv2.COLOR_BGR2RGB), width=460)


def webcam_tab(pipeline: InternshipSignPipeline, ready: bool, active: bool) -> None:
    st.subheader("Real-time video")
    if not active:
        st.info("Webcam prediction is disabled outside the active period.")
        return
    if not ready:
        st.error("The selected sign model is not yet validated. Complete training before starting the webcam.")
        return
    if not WEBRTC_AVAILABLE:
        st.error("Webcam support needs PyAV and streamlit-webrtc in this project environment.")
        st.code(".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt", language="powershell")
        return

    st.info("Click Start and allow camera access. Hold one supported sign steady for six frames.")
    context = webrtc_streamer(
        key="internship-sign-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 20}},
            "audio": False,
        },
        video_processor_factory=lambda: VideoProcessor(pipeline),
        async_processing=True,
    )
    if context.video_processor is not None:
        received, hands, result, stable = context.video_processor.status()
        st.caption(f"Camera connected: {'YES' if received else 'NO'} · Frames received: {received} · Hands detected: {hands}")
        if stable:
            st.success(f"Stable detected sign: {stable} · latest confidence: {result.confidence * 100:.1f}%")
        elif result.bbox is None:
            st.info("No hand detected. Keep one hand clearly inside the camera frame.")
        elif result.recognized:
            st.info(f"Current sign: {result.label} ({result.confidence * 100:.1f}%). Waiting for stable frames.")
        else:
            st.warning("Unknown / Please try again")


def predict_known_word_video(video_bytes: bytes, suffix: str, classifier: TemporalWordClassifier) -> tuple[SignResult, np.ndarray | None]:
    """Apply the exact 30-frame landmark preprocessing to one uploaded video."""
    temporary = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(temporary.name)
    try:
        temporary.write(video_bytes)
        temporary.close()
        capture = cv2.VideoCapture(str(path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return SignResult(None, 0.0, False, None, reason="The uploaded video contains no readable frames."), None
        positions = np.linspace(0, total_frames - 1, RAW_SEQUENCE_FRAMES, dtype=np.int32)
        detector = HandDetector(static_image_mode=False, max_hands=2, detection_confidence=0.20, tracking_confidence=0.20)
        features: list[np.ndarray] = []
        valid_frames = 0
        last_bbox: tuple[int, int, int, int] | None = None
        preview: np.ndarray | None = None
        try:
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                ok, frame_bgr = capture.read()
                if not ok:
                    features.append(empty_feature_frame())
                    continue
                preview = frame_bgr
                observations = detector.detect(frame_bgr)
                if observations:
                    features.append(sequence_features(observations))
                    valid_frames += 1
                    last_bbox = (
                        min(item.bbox[0] for item in observations), min(item.bbox[1] for item in observations),
                        max(item.bbox[2] for item in observations), max(item.bbox[3] for item in observations),
                    )
                else:
                    features.append(empty_feature_frame())
        finally:
            detector.close()
            capture.release()
        if valid_frames < MIN_VALID_FRAMES:
            return SignResult(None, 0.0, False, last_bbox, reason="No clear hand sequence detected"), preview
        prediction = classifier.predict_sequence(resample_sequence(np.asarray(features, dtype=np.float32)))
        return SignResult(
            prediction.label if prediction.is_known else None,
            prediction.confidence,
            prediction.is_known,
            last_bbox,
            top5=prediction.top_predictions,
            reason=None if prediction.is_known else "Unknown Sign",
        ), preview
    finally:
        temporary.close()
        path.unlink(missing_ok=True)


def known_words_upload_tab(classifier: TemporalWordClassifier, ready: bool, active: bool) -> None:
    """Preview images and genuinely classify uploaded motion videos only."""
    st.subheader("Upload image")
    uploaded_image = st.file_uploader(
        "Browse / Upload Image",
        type=["jpg", "jpeg", "png"],
        key="known_words_upload",
        help="Preview a sign image. Dynamic known words require video or the webcam for a genuine prediction.",
    )
    if uploaded_image is not None:
        raw = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("The selected file could not be read as an image.")
        else:
            st.markdown("**Uploaded image**")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), width=460)
            st.info("A single image cannot genuinely classify these motion words. Upload a short MP4 below or use Real-time Video.")

    uploaded_video = st.file_uploader(
        "Upload a short sign video",
        type=["mp4", "avi", "mov"],
        key="known_words_video_upload",
        help="The video uses the same 30-frame MediaPipe landmark pipeline as the webcam.",
    )
    if uploaded_video is None:
        return
    st.video(uploaded_video.getvalue())
    if not active:
        st.info("Sign Language Detection is available only between 6:00 PM and 10:00 PM.")
        return
    if not ready:
        st.error("The known-word model is unavailable or has not passed validation.")
        return
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    result, preview = predict_known_word_video(uploaded_video.getvalue(), suffix, classifier)
    if preview is not None:
        st.markdown("**Detected hand region**")
        st.image(cv2.cvtColor(draw_result(preview, result), cv2.COLOR_BGR2RGB), width=460)
    st.subheader("Prediction result")
    if result.bbox is None:
        st.warning("No clear hand sequence detected. Upload a well-lit video showing the complete sign motion.")
    elif result.recognized:
        st.success(f"Detected Word: {result.label}")
        st.write(f"Confidence: {result.confidence * 100:.1f}%")
    else:
        st.warning("Detected Word: Unknown / Please try again")
        st.write(f"Highest confidence: {result.confidence * 100:.1f}%")
    if result.top5:
        st.caption("Top predictions: " + " · ".join(f"{label} {score * 100:.1f}%" for label, score in result.top5))


def known_words_webcam_tab(classifier: TemporalWordClassifier, ready: bool, active: bool) -> None:
    st.subheader("Real-time video — Known Words")
    if not active:
        st.info("Webcam prediction is disabled outside the active period.")
        return
    if not ready:
        st.error("The known-word model is unavailable or has not passed validation.")
        return
    if not WEBRTC_AVAILABLE:
        st.error("Webcam support needs PyAV and streamlit-webrtc in this project environment.")
        return
    st.info("Click Start and perform one sign steadily. The model uses a real 16-frame landmark sequence and confirms a word after six consistent predictions.")
    context = webrtc_streamer(
        key="known-words-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": []},
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 20}},
            "audio": False,
        },
        video_processor_factory=lambda: KnownWordVideoProcessor(classifier),
        async_processing=True,
    )
    if context.video_processor is not None:
        received, hands, result, stable = context.video_processor.status()
        st.caption(f"Camera connected: {'YES' if received else 'NO'} · Frames received: {received} · Hands detected: {hands}")
        if stable:
            st.success(f"Detected Word: {stable} · Confidence: {result.confidence * 100:.1f}%")
        elif result.bbox is None:
            st.info("No hand detected. Keep the signing hands clearly inside the camera frame.")
        elif result.recognized:
            st.info(f"Current word: {result.label} ({result.confidence * 100:.1f}%). Waiting for stable frames.")
        elif result.reason == "Collecting a 30-frame sign window":
            st.info("Collecting motion frames…")
        else:
            st.warning("Detected Word: Unknown / Please try again")
            st.write(f"Highest confidence: {result.confidence * 100:.1f}%")
        if result.top5:
            st.caption("Top predictions: " + " · ".join(f"{label} {score * 100:.1f}%" for label, score in result.top5))


def main() -> None:
    pipeline = load_pipeline()
    metrics = load_metrics()
    word_model_path, word_labels_path, word_metrics_path = known_word_artifacts()
    word_classifier = load_word_classifier(word_model_path, word_labels_path)
    word_metrics = load_metrics_file(word_metrics_path)
    active = prediction_active()
    ready = model_is_ready(pipeline, metrics)
    word_ready = word_model_is_ready(word_classifier, word_metrics)

    st.title("🤟 Sign Language Detection System")
    st.caption("A validated static-sign model plus a separate temporal known-word recogniser.")
    status_col, time_col, period_col, model_col, word_col = st.columns(5)
    status_col.metric("System Status", "ACTIVE" if active else "INACTIVE")
    time_col.metric("Current local time", current_time_text())
    period_col.metric("Active period", "6:00 PM – 10:00 PM")
    model_col.metric("A–Z model", "Ready" if ready else "Training required")
    word_col.metric("Known Words", "Ready" if word_ready else "Unavailable")

    if not active:
        st.warning("Sign Language Detection is available only between 6:00 PM and 10:00 PM.")
    if not pipeline.available:
        st.error("No validated internship model is available. Run the training command shown below.")
    elif not ready:
        st.warning("A model exists but has not met the 80% held-out test readiness requirement.")
    else:
        accuracy = metrics.get("end_to_end_test_accuracy", metrics["test_accuracy"])
        st.success(f"Validated held-out test accuracy: {accuracy * 100:.2f}%")

    st.markdown("**Trained alphabet classes:** " + ", ".join(pipeline.labels))
    if word_ready:
        st.markdown("**Trained known words:** " + ", ".join(word_classifier.labels))
        st.caption(
            f"Known-word held-out accuracy: {word_metrics['held_out_test_accuracy'] * 100:.2f}% · "
            f"macro F1: {word_metrics['held_out_macro_f1']:.3f}"
        )
    else:
        st.warning("Known Words mode is unavailable until its separate video model passes validation. Collect 30–50 genuine webcam sequences per word, then run the four-word trainer.")
    if not ready:
        st.code(".\\.venv\\Scripts\\python.exe scripts\\train_internship_static_signs.py", language="powershell")

    mode = st.radio("Recognition mode", ["A–Z Alphabet", "Known Words"], horizontal=True)
    upload, webcam = st.tabs(["Upload Image", "Real-time Video"])
    with upload:
        if mode == "A–Z Alphabet":
            upload_tab(pipeline, ready, active)
        else:
            known_words_upload_tab(word_classifier, word_ready, active)
    with webcam:
        if mode == "A–Z Alphabet":
            webcam_tab(pipeline, ready, active)
        else:
            known_words_webcam_tab(word_classifier, word_ready, active)


if __name__ == "__main__":
    main()
