"""Game-engine-facing webcam interface for hand gesture predictions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import joblib

from src.features.landmark_normalizer import FEATURE_COLUMNS, normalize_hand_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "suit_jawa_classifier.joblib"
CLASS_ORDER = ("Gajah", "Orang", "Semut", "unknown")
CORE_LABELS = frozenset({"Gajah", "Orang", "Semut"})
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


class CameraDisconnectedError(RuntimeError):
    """Raised when an opened camera can no longer provide frames."""


@dataclass(frozen=True)
class GesturePrediction:
    """A prediction; confidence is the probability of the returned label."""

    label: str
    confidence: float
    is_valid: bool
    hand_detected: bool


def _validate_classifier(model: Any) -> tuple[str, ...]:
    if not callable(getattr(model, "predict", None)):
        raise ValueError("Artifact classifier tidak memiliki fungsi predict.")
    if not callable(getattr(model, "predict_proba", None)):
        raise ValueError(
            "Artifact classifier harus menyediakan predict_proba untuk confidence."
        )

    raw_classes = getattr(model, "classes_", None)
    if raw_classes is None:
        raise ValueError("Artifact classifier tidak menyediakan classes_.")
    classes = tuple(str(label) for label in raw_classes)
    if len(classes) != len(CLASS_ORDER) or set(classes) != set(CLASS_ORDER):
        raise ValueError(
            "Classifier harus memiliki kelas Gajah, Orang, Semut, dan unknown."
        )

    feature_count = getattr(model, "n_features_in_", len(FEATURE_COLUMNS))
    if feature_count != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Classifier memerlukan {feature_count} fitur; "
            f"{len(FEATURE_COLUMNS)} fitur landmark tersedia."
        )
    return classes


def _load_model_artifact(model_path: Path) -> tuple[Any, tuple[str, ...], float | None]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}. Jalankan train_classifier.py."
        )

    artifact = joblib.load(model_path)
    if isinstance(artifact, Mapping):
        model = artifact.get("model")
        if model is None:
            raise ValueError("Artifact model berbentuk dictionary tanpa key 'model'.")
        feature_columns = artifact.get("feature_columns")
        if (
            feature_columns is not None
            and tuple(feature_columns) != FEATURE_COLUMNS
        ):
            raise ValueError(
                "Urutan fitur artifact berbeda dari normalisasi landmark training."
            )
        declared_classes = artifact.get("classes", artifact.get("labels"))
        confidence_threshold = artifact.get("confidence_threshold")
    else:
        model = artifact
        declared_classes = None
        confidence_threshold = None

    classes = _validate_classifier(model)
    if declared_classes is not None:
        declared = tuple(str(label) for label in declared_classes)
        if len(declared) != len(CLASS_ORDER) or set(declared) != set(classes):
            raise ValueError("Daftar kelas artifact tidak cocok dengan classifier.")
    return model, classes, confidence_threshold


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


class GestureInterface:
    """Own webcam/MediaPipe resources and expose one prediction per call.

    The game engine only needs :meth:`get_prediction`; landmark extraction and
    normalization stay inside the vision package.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        camera_index: int = 0,
        confidence_threshold: float | None = None,
        *,
        model: Any | None = None,
        capture: Any | None = None,
        hand_processor: Any | None = None,
        cv2_module: Any | None = None,
        preview_enabled: bool = False,
    ) -> None:
        if model is None:
            model, classes, artifact_threshold = _load_model_artifact(model_path)
        else:
            classes = _validate_classifier(model)
            artifact_threshold = None

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else artifact_threshold
            if artifact_threshold is not None
            else DEFAULT_CONFIDENCE_THRESHOLD
        )
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("confidence_threshold harus berupa angka antara 0 dan 1.")

        self._model = model
        self._classes = classes
        self.confidence_threshold = float(threshold)
        self.camera_index = camera_index
        self._capture = capture
        self._hand_processor = hand_processor
        self._cv2 = cv2_module
        self.preview_enabled = preview_enabled
        self._latest_frame_jpeg: bytes | None = None
        self._owns_capture = capture is None
        self._owns_hand_processor = hand_processor is None
        self._started = False

    def start(self) -> None:
        """Open the configured camera and initialize MediaPipe Hands."""
        if self._started:
            return

        media_pipe = None
        if self._cv2 is None or self._capture is None or self._hand_processor is None:
            cv2, media_pipe = _load_vision_dependencies()
            if self._cv2 is None:
                self._cv2 = cv2

        if self._cv2 is None:
            raise RuntimeError("OpenCV belum diinisialisasi.")
        if self._capture is None:
            self._capture = self._cv2.VideoCapture(self.camera_index)
        capture = self._capture
        if capture is None:
            raise RuntimeError("Webcam belum diinisialisasi.")
        if not capture.isOpened():
            if self._owns_capture:
                capture.release()
                self._capture = None
            raise RuntimeError(
                f"Webcam dengan indeks {self.camera_index} tidak dapat dibuka. "
                "Periksa izin kamera dan indeks perangkat."
            )

        capture.set(self._cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(self._cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if self._hand_processor is None:
            if media_pipe is None:
                raise RuntimeError("MediaPipe Hands belum diinisialisasi.")
            try:
                self._hand_processor = media_pipe.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                if self._owns_capture:
                    capture.release()
                    self._capture = None
                raise
        self._started = True

    def get_prediction(self) -> GesturePrediction:
        """Capture a frame and return a game-ready gesture result."""
        if not self._started:
            self.start()

        capture = self._capture
        cv2 = self._cv2
        hand_processor = self._hand_processor
        model = self._model
        if capture is None or cv2 is None or hand_processor is None or model is None:
            raise RuntimeError("GestureInterface belum siap membaca frame.")

        success, frame = capture.read()
        if not success or frame is None:
            raise CameraDisconnectedError(
                f"Kamera dengan indeks {self.camera_index} terputus "
                "atau tidak dapat membaca frame."
            )

        frame = cv2.flip(frame, 1)
        if self.preview_enabled:
            encoded, image = cv2.imencode(".jpg", frame)
            if not encoded:
                raise RuntimeError("Gagal membuat preview JPEG dari kamera.")
            self._latest_frame_jpeg = image.tobytes()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand_processor.process(rgb_frame)
        hand_landmarks = getattr(result, "multi_hand_landmarks", None)
        if not hand_landmarks:
            return GesturePrediction("unknown", 0.0, False, False)

        features = normalize_hand_landmarks(hand_landmarks[0])
        predicted_label = str(model.predict([features])[0])
        if predicted_label not in self._classes:
            raise ValueError(
                f"Classifier mengembalikan label yang tidak dikenal: {predicted_label!r}."
            )

        probabilities = [
            float(value) for value in model.predict_proba([features])[0]
        ]
        if len(probabilities) != len(self._classes) or not all(
            math.isfinite(value) and 0 <= value <= 1 for value in probabilities
        ):
            raise ValueError("Classifier menghasilkan nilai confidence yang tidak valid.")
        unknown_confidence = probabilities[self._classes.index("unknown")]
        confidence = probabilities[self._classes.index(predicted_label)]
        if predicted_label not in CORE_LABELS or confidence < self.confidence_threshold:
            return GesturePrediction("unknown", unknown_confidence, False, True)
        return GesturePrediction(predicted_label, confidence, True, True)

    def close(self) -> None:
        """Release camera and MediaPipe resources created by this interface."""
        try:
            if self._owns_hand_processor and self._hand_processor is not None:
                self._hand_processor.close()
        finally:
            try:
                if self._owns_capture and self._capture is not None:
                    self._capture.release()
            finally:
                if self._owns_hand_processor:
                    self._hand_processor = None
                if self._owns_capture:
                    self._capture = None
                self._latest_frame_jpeg = None
                self._started = False

    @property
    def latest_frame_jpeg(self) -> bytes | None:
        """Return the most recent JPEG when camera preview is enabled."""
        return self._latest_frame_jpeg

    def __enter__(self) -> GestureInterface:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "GestureInterface",
    "GesturePrediction",
    "CameraDisconnectedError",
]
