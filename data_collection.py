"""Data collection utility for static single-hand sign language gestures.

This script captures webcam frames, extracts 21 hand landmarks with MediaPipe,
normalizes coordinates relative to the wrist, and stores feature vectors in CSV.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np


# Default labels: A-Z, 0-9 and selected special characters.
DEFAULT_LABELS: List[str] = [*(chr(i) for i in range(ord("A"), ord("Z") + 1)), *(str(i) for i in range(10)), "SPACE", "DEL", "?", "!"]


class LandmarkExtractor:
    """Extract and normalize single-hand landmark vectors."""

    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return normalized feature vector (42 values) or None if no hand."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)

        # Normalize coordinates relative to wrist (landmark 0).
        wrist = coords[0]
        coords -= wrist

        # Scale normalization for size invariance.
        max_norm = np.max(np.linalg.norm(coords, axis=1))
        if max_norm > 1e-6:
            coords /= max_norm

        return coords.flatten()

    def close(self) -> None:
        self.hands.close()


def ensure_dataset_file(csv_path: Path) -> None:
    """Create CSV with header if it does not already exist."""
    if csv_path.exists():
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"f{i}" for i in range(42)] + ["label"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_sample(csv_path: Path, features: np.ndarray, label: str) -> None:
    """Append one labeled sample to CSV."""
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*features.tolist(), label])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sign gesture landmarks into a CSV dataset.")
    parser.add_argument("--output", type=Path, default=Path("dataset/landmarks.csv"), help="Output CSV file path.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS), help="Comma-separated label list.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()]

    ensure_dataset_file(args.output)
    extractor = LandmarkExtractor()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    label_idx = 0
    counts = {label: 0 for label in labels}

    print("Controls:")
    print("  n: next label | p: previous label | c: capture sample | q: quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_label = labels[label_idx]
        features = extractor.extract(frame)

        display = frame.copy()
        status = "Hand detected" if features is not None else "No hand"
        cv2.putText(display, f"Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, f"Captured: {counts[current_label]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(display, "n/p: change label | c: capture | q: quit", (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Data Collection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("n"):
            label_idx = (label_idx + 1) % len(labels)
        elif key == ord("p"):
            label_idx = (label_idx - 1) % len(labels)
        elif key == ord("c"):
            if features is not None:
                append_sample(args.output, features, current_label)
                counts[current_label] += 1
                print(f"Captured sample for '{current_label}'. Total: {counts[current_label]}")
            else:
                print("No hand detected; sample not saved.")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()
