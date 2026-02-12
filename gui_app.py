"""Tkinter GUI wrapper for real-time sign language recognition."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from predict import GesturePredictor


class SignGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Real-time Sign Language Recognition")
        self.root.geometry("520x300")

        self.model_path_var = tk.StringVar(value="models/sign_rf.joblib")
        self.current_prediction_var = tk.StringVar(value="-")
        self.text_var = tk.StringVar(value="")

        self.predictor: GesturePredictor | None = None
        self.predict_thread: threading.Thread | None = None
        self.last_added = ""

        self._build_ui()

    def _build_ui(self) -> None:
        tk.Label(self.root, text="Model Path:").pack(anchor="w", padx=12, pady=(12, 2))
        tk.Entry(self.root, textvariable=self.model_path_var, width=60).pack(anchor="w", padx=12)

        button_frame = tk.Frame(self.root)
        button_frame.pack(anchor="w", padx=12, pady=10)

        tk.Button(button_frame, text="Start Recognition", command=self.start_recognition).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(button_frame, text="Stop", command=self.stop_recognition).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(button_frame, text="Clear Text", command=self.clear_text).pack(side=tk.LEFT)

        tk.Label(self.root, text="Stable Prediction:", font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 2))
        tk.Label(self.root, textvariable=self.current_prediction_var, font=("Arial", 18), fg="green").pack(anchor="w", padx=12)

        tk.Label(self.root, text="Recognized Text:", font=("Arial", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        tk.Entry(self.root, textvariable=self.text_var, width=60, font=("Arial", 12)).pack(anchor="w", padx=12)

        help_text = (
            "Tips: Keep your hand within camera view and use controlled lighting.\n"
            "Supported classes are static single-hand signs collected in your dataset."
        )
        tk.Label(self.root, text=help_text, justify="left", fg="gray25").pack(anchor="w", padx=12, pady=12)

    def _on_prediction(self, label: str) -> None:
        if not label:
            return

        def update_ui() -> None:
            self.current_prediction_var.set(label)
            text = self.text_var.get()

            # Avoid appending the same label repeatedly on consecutive stable frames.
            if label == self.last_added:
                return

            if label == "SPACE":
                text += " "
            elif label == "DEL":
                text = text[:-1]
            else:
                text += label

            self.text_var.set(text)
            self.last_added = label

        self.root.after(0, update_ui)

    def start_recognition(self) -> None:
        if self.predict_thread and self.predict_thread.is_alive():
            return

        model_path = Path(self.model_path_var.get())
        if not model_path.exists():
            messagebox.showerror("Missing model", f"Model file not found: {model_path}")
            return

        self.last_added = ""
        self.current_prediction_var.set("-")

        self.predictor = GesturePredictor(model_path=model_path)

        def worker() -> None:
            assert self.predictor is not None
            try:
                self.predictor.run(on_prediction=self._on_prediction, show_window=True)
            except Exception as exc:  # keep GUI alive and report runtime errors
                self.root.after(0, lambda: messagebox.showerror("Runtime error", str(exc)))
            finally:
                if self.predictor is not None:
                    self.predictor.close()
                    self.predictor = None

        self.predict_thread = threading.Thread(target=worker, daemon=True)
        self.predict_thread.start()

    def stop_recognition(self) -> None:
        if self.predictor is not None:
            self.predictor.stop()

    def clear_text(self) -> None:
        self.text_var.set("")
        self.last_added = ""


if __name__ == "__main__":
    root = tk.Tk()
    app = SignGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_recognition(), root.destroy()))
    root.mainloop()
