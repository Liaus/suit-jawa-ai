"""Run the Suit Jawa webcam classifier locally without network access."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib

from src.features.landmark_normalizer import FEATURE_COLUMNS, normalize_hand_landmarks
from train_classifier import CLASS_ORDER, DEFAULT_MODEL


def load_model(model_path: Path) -> Any:
    """Load a compatible local model bundle."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. Jalankan train_classifier.py "
            "terlebih dahulu."
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Format model {model_path} tidak valid.")
    if bundle.get("feature_columns") != list(FEATURE_COLUMNS):
        raise ValueError("Fitur model tidak cocok dengan landmark normalizer.")
    if bundle.get("labels") != list(CLASS_ORDER):
        raise ValueError("Urutan label model tidak cocok dengan demo.")
    model = bundle.get("model")
    if (
        model is None
        or not callable(getattr(model, "predict", None))
        or not callable(getattr(model, "predict_proba", None))
    ):
        raise ValueError("Model tersimpan tidak memiliki fungsi prediksi yang valid.")
    return model


def _load_vision_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            f"OpenCV tidak tersedia untuk Python yang sedang dipakai "
            f"({sys.executable}). Pasang dependensi dari requirements.txt."
        ) from error
    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            f"MediaPipe tidak tersedia untuk Python yang sedang dipakai "
            f"({sys.executable}). Pasang dependensi dari requirements.txt."
        ) from error
    return cv2, mp


def run_demo(
    model: Any,
    camera_index: int = 0,
) -> None:
    """Predict hand gestures from a webcam using local MediaPipe/model files."""
    cv2, mp = _load_vision_dependencies()
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Webcam dengan indeks {camera_index} tidak dapat dibuka. "
            "Periksa izin kamera dan indeks perangkat."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands_api = mp.solutions.hands
    try:
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Gagal membaca frame dari webcam.")
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)
                landmarks = (
                    result.multi_hand_landmarks[0]
                    if result.multi_hand_landmarks
                    else None
                )

                if landmarks is None:
                    prediction = "Tidak ada tangan"
                    confidence_text = ""
                else:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        landmarks,
                        hands_api.HAND_CONNECTIONS,
                    )
                    features = normalize_hand_landmarks(landmarks)
                    prediction = str(model.predict([features])[0])
                    probabilities = model.predict_proba([features])[0]
                    confidence_text = f" ({max(probabilities):.0%})"

                cv2.putText(
                    frame,
                    f"Prediksi: {prediction}{confidence_text}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if prediction == "unknown" else (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Mode offline - tekan Q untuk keluar",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                )
                cv2.imshow("Suit Jawa vs AI - Demo Offline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("offline",),
        default="offline",
        help="Jalankan inferensi lokal; tidak memerlukan koneksi internet.",
    )
    args = parser.parse_args(argv)
    try:
        model = load_model(args.model)
        run_demo(model, camera_index=args.camera_index)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"Demo gagal: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
