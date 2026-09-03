# Internship Project Report: Sign Language Detection

## Objective

Develop a Windows CPU-compatible sign-language detection application that
supports image upload and real-time webcam input, recognises selected signs,
shows genuine confidence, and operates only from 6:00 PM to 10:00 PM local
time.

## Implementation

- **GUI:** Streamlit interface with separate static-sign and known-word modes.
- **Image upload:** JPG, JPEG, and PNG images are processed using MediaPipe
  hand landmarks. The interface displays the detected hand region, predicted
  sign, and model confidence.
- **Real-time video:** Browser webcam access is provided through WebRTC.
  Predictions are smoothed over multiple frames to reduce flicker.
- **Static signs:** A, B, C, D, L, V, W, and Y.
- **Known words:** Hello, Yes, No, Help, and Stop. The word model uses a
  16-frame MediaPipe landmark sequence; therefore word predictions are made
  from webcam/video input rather than a single still image.
- **Unknown handling:** Unsupported, low-confidence, and no-hand inputs are
  reported as `Unknown Sign` rather than assigned a fabricated label.
- **Time control:** Prediction and webcam processing are disabled outside the
  required 6:00 PM–10:00 PM local-time window while the GUI remains visible.

## Models and Evaluation

| Model | Method | Held-out evaluation |
| --- | --- | --- |
| Static-sign subset | MediaPipe canonical hand landmarks + StandardScaler + RBF SVC | 98.72% end-to-end accuracy on 1,956 detected untouched test images; 99.22% classifier accuracy after hand detection |
| Known-word model | 16-frame MediaPipe two-hand landmark sequence + GRU | 100.00% held-out accuracy and 1.000 macro F1 on 64 held-out source clips |

For the static model, source images were split by SHA-256 image hash into
80% training, 10% validation, and 10% untouched test sets. The known-word
model kept independent source-video groups in a single split to prevent
leakage. Exact labels, metrics, reports, and confusion matrices are committed
under `models/`.

## Technology

Python, Streamlit, Streamlit WebRTC, OpenCV, MediaPipe, scikit-learn, and
TensorFlow. The project runs on Windows CPU and does not require CUDA or an
NVIDIA GPU.

## Run Command

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_app.ps1
```

## Limitation

The static model is a validated eight-letter subset, not all 26 alphabet
letters. The known-word test metric is measured on held-out source clips;
webcam results may vary with signer, camera position, background, and
lighting. The application deliberately returns `Unknown Sign` when confidence
is insufficient.
