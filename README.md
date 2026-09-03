# Sign Language Detection System

A compact Windows CPU project built for the internship task. It recognises a
small, verified set of static signs with one MediaPipe-landmark + SVM model.
The available local dataset contains alphabet labels only, so the app supports
the eight real labels selected from it: **A, B, F, I, L, V, W, Y**. It does not
pretend to recognise word labels that are not present in the dataset.

## What it does

- Upload a JPG, JPEG, or PNG and receive a hand box, sign prediction, and real confidence.
- Use the browser webcam through Streamlit WebRTC. The same inference function
  used for uploads handles every video frame, then confirms a label after six
  consecutive confident frames.
- Shows `Unknown Sign` for low-confidence input and a no-hand message when
  MediaPipe cannot find a hand.
- Enforces the required local-time operating window: **6:00 PM–10:00 PM**.
  Outside that window the interface remains visible but prediction and webcam
  detection are disabled.

## Dataset and split

Training uses only:

`data/asl_alphabet/train/{A,B,F,I,L,V,W,Y}`

The trainer counts every source image, then chooses 500 deterministic images
per class and makes stratified 80% / 10% / 10% train, validation, and untouched
test partitions before extracting landmarks. It writes the exact sample counts,
held-out metrics, classification report, and confusion matrix to `models/`.

## Setup and training (Windows PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\train_internship_static_signs.py
```

The final trainer only promotes a checkpoint after it has evaluated the
untouched test partition. Its artifacts are:

- `models/internship_static_signs.joblib`
- `models/internship_static_signs_labels.json`
- `models/internship_static_signs_metrics.json`
- `models/internship_static_signs_report.json`
- `models/internship_static_signs_confusion.csv`

Start the dashboard with the project interpreter:

```powershell
.\run_app.ps1
```

or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app uses no NVIDIA/CUDA requirement. PyAV is included because
`streamlit-webrtc` receives webcam frames as `av.VideoFrame` objects.

## Four-word landmark collection: Hello, Help, Yes, No

The supplied reference pictures are deliberately **not** copied into the
training set. They are single screenshots with visible word text, so training
on them would leak labels and would not teach a model the sign motion.

The dedicated collector stores only 30-frame MediaPipe two-hand landmark
sequences in `data/known_words_hello_help_yes_no/`. Each saved sample has a
unique capture ID, an operator-confirmed label, a preprocessing version, and a
content fingerprint that rejects exact duplicate sequences. Stop Streamlit
before collecting because the collector must own the webcam.

Collect 30–50 independent performances for each word (40 is a good target):

```powershell
.\.venv\Scripts\python.exe scripts\collect_four_word_landmarks.py --label Hello --samples 40 --confirm --automatic
.\.venv\Scripts\python.exe scripts\collect_four_word_landmarks.py --label Help --samples 40 --confirm --automatic
.\.venv\Scripts\python.exe scripts\collect_four_word_landmarks.py --label Yes --samples 40 --confirm --automatic
.\.venv\Scripts\python.exe scripts\collect_four_word_landmarks.py --label No --samples 40 --confirm --automatic
```

Press `A` in the collector to start the automatic takes, change the gesture or
camera position a little between takes, and press `Q` to stop. It refuses to
save a take with fewer than 10 detected hand frames.

After every class has at least 30 valid independent sequences, train and
evaluate the separate CPU GRU model:

```powershell
.\.venv\Scripts\python.exe scripts\train_collected_known_words.py
```

The trainer makes a stratified 70/15/15 train/validation/untouched-test split
by capture ID. It saves the model, labels, manifest, metrics, classification
report, and confusion matrix under `models/known_words_hello_help_yes_no_v1*`.
It will not start with missing or duplicate data, and the dashboard only
enables this four-word checkpoint after its saved held-out metrics pass the
readiness gate. A static image is previewed but never
used to make a fake dynamic-word prediction; upload a short video or use the
webcam for the landmark-sequence model.
