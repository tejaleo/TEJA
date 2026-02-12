"""Real-time prediction module for static sign language recognition."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3


class SpeechEngine:
    """Non-blocking text-to-speech wrapper around pyttsx3."""

    def __init__(self, rate: int = 160) -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self.engine.say(text)
            self.engine.runAndWait()

    def speak(self, text: str) -> None:
        if text:
            self.queue.put(text)

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=1.0)


class GesturePredictor:
    """Capture webcam, classify gesture, and stabilize predictions."""

    def __init__(
        self,
        model_path: Path = Path("models/sign_rf.joblib"),
        camera_index: int = 0,
        vote_window: int = 10,
        vote_threshold: int = 6,
    ) -> None:
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.encoder = payload["encoder"]

        self.camera_index = camera_index
        self.vote_window = vote_window
        self.vote_threshold = vote_threshold

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        self.speech = SpeechEngine()
        self.recent_predictions: deque[str] = deque(maxlen=vote_window)
        self.last_spoken = ""
        self.last_spoken_time = 0.0
        self.cooldown_s = 1.0

        self.running = False

    @staticmethod
    def normalize_landmarks(hand_landmarks) -> np.ndarray:
        coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)

        # Translate to wrist origin.
        coords -= coords[0]

        # Scale to normalize hand size.
        max_norm = np.max(np.linalg.norm(coords, axis=1))
        if max_norm > 1e-6:
            coords /= max_norm

        return coords.flatten()

    def predict_from_frame(self, frame_bgr: np.ndarray) -> tuple[Optional[str], np.ndarray]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        frame_drawn = frame_bgr.copy()

        if not results.multi_hand_landmarks:
            return None, frame_drawn

        hand_landmarks = results.multi_hand_landmarks[0]
        self.mp_draw.draw_landmarks(frame_drawn, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        features = self.normalize_landmarks(hand_landmarks).reshape(1, -1)
        pred_idx = self.model.predict(features)[0]
        pred_label = self.encoder.inverse_transform([pred_idx])[0]
        return pred_label, frame_drawn

    def stabilize_prediction(self, raw_pred: Optional[str]) -> str:
        if raw_pred is None:
            self.recent_predictions.clear()
            return ""

        self.recent_predictions.append(raw_pred)
        counts = Counter(self.recent_predictions)
        label, count = counts.most_common(1)[0]
        return label if count >= self.vote_threshold else ""

    def maybe_speak(self, label: str) -> None:
        # Speak only when a stable label changes and cooldown has passed.
        now = time.time()
        if label and label != self.last_spoken and (now - self.last_spoken_time) >= self.cooldown_s:
            speech_text = "space" if label == "SPACE" else "delete" if label == "DEL" else label
            self.speech.speak(speech_text)
            self.last_spoken = label
            self.last_spoken_time = now

    def run(
        self,
        on_prediction: Optional[Callable[[str], None]] = None,
        window_name: str = "Sign Language Recognition",
        show_window: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Unable to open webcam")

        self.running = True
        while self.running:
            ok, frame = cap.read()
            if not ok:
                break

            raw_pred, drawn = self.predict_from_frame(frame)
            stable_pred = self.stabilize_prediction(raw_pred)
            if stable_pred:
                self.maybe_speak(stable_pred)

            if on_prediction is not None:
                on_prediction(stable_pred)

            cv2.putText(drawn, f"Raw: {raw_pred or '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(drawn, f"Stable: {stable_pred or '-'}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(drawn, "Press q to quit", (10, drawn.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if show_window:
                cv2.imshow(window_name, drawn)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        cap.release()
        if show_window:
            cv2.destroyWindow(window_name)

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.hands.close()
        self.speech.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time sign language prediction.")
    parser.add_argument("--model", type=Path, default=Path("models/sign_rf.joblib"), help="Trained model path")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = GesturePredictor(model_path=args.model, camera_index=args.camera)
    try:
        predictor.run()
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
