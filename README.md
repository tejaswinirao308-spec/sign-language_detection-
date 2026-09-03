# Sign Language Detection

Windows CPU-friendly Streamlit application prepared for the internship task:
train a sign-language model, recognise selected known words, provide uploaded
image and real-time video inputs, and restrict prediction to 6:00 PM–10:00 PM
local time.

## Features

- **A–Z Alphabet mode:** a validated static-sign subset: **A, B, C, D, L, V,
  W, Y**. These are the eight labels with adequate images in the selected
  training subset; unsupported letters are reported as `Unknown Sign`.
- **Known Words mode:** temporal recognition of **Hello, Yes, No, Help, Stop**.
  It uses real MediaPipe hand-landmark sequences, so a word prediction is made
  only from a webcam sequence or uploaded video—not from a single screenshot.
- **Upload image:** detects a hand, draws its bounding box, and returns the
  predicted static sign and genuine model confidence.
- **Real-time webcam:** browser WebRTC video, hand bounding box, confidence,
  and multi-frame smoothing to reduce flicker.
- **Unknown handling:** low-confidence or unsupported gestures are never
  forced into a class.
- **Access window:** prediction and webcam processing are enabled only from
  **6:00 PM to 10:00 PM** local time. The GUI stays visible outside that window
  and shows the availability message.

## Models and measured evaluation

| Mode | Checkpoint | Held-out result |
| --- | --- | --- |
| Static alphabet subset | `models/internship_static_signs_abcdlvwy_v1.joblib` | 98.72% end-to-end accuracy on 1,956 detected untouched test images; 99.22% classifier accuracy after hand detection |
| Known words | `models/known_words_hello_yes_no_help_stop_v1.keras` | 100.00% held-out accuracy / 1.000 macro F1 on 64 held-out source clips |

The known-word score is a result on the source dataset's held-out clips. It is
not a promise of identical accuracy for every webcam, signer, background, or
lighting condition. The app displays `Unknown Sign` when the genuine
probability does not meet the saved confidence criterion.

## Data and leakage controls

The static-sign trainer uses all valid images from the selected eight A–Z
classes (3,000 source images per class). It applies a SHA-256 image-hash split
before detection: 80% train, 10% validation, and 10% untouched test. Duplicate
images across the partitions are rejected.

Known-word training uses 16 uniformly sampled MediaPipe landmark frames per
clip and keeps independent source-video groups in only one partition. Its
metrics, class labels, and confusion matrix are stored alongside the checkpoint.

## Run on Windows

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run_app.ps1
```

Or run directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

No CUDA or NVIDIA GPU is required.

## Retraining

Static alphabet subset:

```powershell
.\.venv\Scripts\python.exe scripts\train_internship_static_signs.py
```

For a separate four-word self-collected model (**Hello, Help, Yes, No**), first
capture at least 30 independent real sequences per word using
`scripts/collect_four_word_landmarks.py`, then run:

```powershell
.\.venv\Scripts\python.exe scripts\train_collected_known_words.py
```

The collector stores landmark sequences only; it does not manufacture labels
or use reference screenshots as training data.

## Project layout

```text
app.py                         Streamlit interface
src/internship_pipeline.py     Shared static sign detection/inference
src/vision.py                  Temporal word-sequence preprocessing
scripts/                       Collection and training scripts
models/                        Versioned trained checkpoints and metrics
requirements.txt               Windows CPU dependencies
run_app.ps1                    Project launcher
```
