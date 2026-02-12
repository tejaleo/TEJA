# Real-Time Sign Language Recognition System (Python)

A complete static **single-hand** sign language recognition pipeline using:
- Webcam + OpenCV
- MediaPipe Hands (21 landmarks)
- Random Forest (scikit-learn)
- Real-time prediction stabilization (majority voting)
- Text-to-speech output (pyttsx3)
- Tkinter GUI wrapper

> Designed for controlled lighting conditions and no specialized hardware.

---

## Project Structure

- `data_collection.py` – Collects normalized landmark feature vectors and labels into CSV.
- `train_model.py` – Trains and saves a Random Forest classifier.
- `predict.py` – Real-time webcam prediction + stabilization + speech output.
- `gui_app.py` – Tkinter interface to run recognition and build recognized text.
- `requirements.txt` – Python dependencies.

---

## 1) Prerequisites

- Python **3.10+**
- A standard webcam
- Controlled lighting background for better landmark tracking

---

## 2) Install Required Libraries

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

`tkinter` is typically bundled with standard Python installations.

---

## 3) Collect Training Data

Run:

```bash
python data_collection.py --output dataset/landmarks.csv
```

Controls in the data collection window:
- `n` → next label
- `p` → previous label
- `c` → capture one sample for current label
- `q` → quit

Default labels include:
- Alphabets: `A-Z`
- Digits: `0-9`
- Special characters: `SPACE`, `DEL`, `?`, `!`

> Capture many samples per class (e.g., 150–300/class) for better accuracy.

---

## 4) Train the Model

```bash
python train_model.py --data dataset/landmarks.csv --model-out models/sign_rf.joblib
```

This prints test accuracy and saves:
- Trained Random Forest
- Label encoder

into `models/sign_rf.joblib`.

---

## 5) Run Real-Time CLI Predictor

```bash
python predict.py --model models/sign_rf.joblib --camera 0
```

Features:
- Live webcam gesture detection
- Raw + stabilized prediction display
- Speech output for stable recognized labels
- `q` to quit

---

## 6) Run GUI App

```bash
python gui_app.py
```

GUI includes:
- Start/Stop recognition
- Current stable prediction
- Recognized text accumulation
- Text controls (`SPACE`, `DEL` supported)

A camera window opens for landmark visualization and prediction overlays.

---

## How Normalization Works

For each frame with one detected hand:
1. Extract 21 `(x, y)` landmarks from MediaPipe.
2. Translate all points by subtracting the wrist point (landmark `0`).
3. Scale by the max Euclidean distance from wrist to achieve scale invariance.
4. Flatten into a 42-value feature vector for classification.

This improves robustness to hand position and size differences.

---

## Performance Notes

- Uses a lightweight feature vector (42 values) and Random Forest inference for real-time performance.
- Majority voting over recent frames reduces flickering predictions.
- Keep only one hand visible and ensure good front lighting for stable results.

---

## Troubleshooting

- **Webcam not opening**: check camera index (`--camera 0`, `--camera 1`, etc.).
- **Poor accuracy**: collect more balanced data for each class and retrain.
- **Speech issues**: verify OS audio backend and pyttsx3 installation.
- **No hand detection**: improve lighting, reduce background clutter, keep hand in frame.

