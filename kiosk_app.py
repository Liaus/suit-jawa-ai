"""Serve the local, offline Suit Jawa kiosk."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
import threading
from typing import Protocol

from flask import Flask, Response, abort, jsonify, send_from_directory

from src.game import GameState, GameLogger, SuitJawaEngine
from src.game.game_logger import DEFAULT_LOG_PATH
from src.vision import (
    CameraDisconnectedError,
    GestureInterface,
    GesturePrediction,
)
from src.vision.gesture_interface import DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
IMAGE_ROOT = PROJECT_ROOT / "src" / "Img"
GESTURE_IMAGES = {
    "gajah": "Gajah.jpg",
    "orang": "Manusia.jpg",
    "semut": "Semut.jpg",
}


class KioskGestureSource(Protocol):
    confidence_threshold: float

    def start(self) -> None: ...

    def get_prediction(self) -> GesturePrediction: ...

    def close(self) -> None: ...

    @property
    def latest_frame_jpeg(self) -> bytes | None: ...


class CameraUnavailableError(RuntimeError):
    """Raised when a game is requested before its camera is ready."""


class KioskController:
    """Own the webcam worker and synchronize the engine with the kiosk API."""

    def __init__(
        self,
        *,
        camera_index: int = 0,
        model_path: Path = DEFAULT_MODEL,
        log_path: Path = DEFAULT_LOG_PATH,
        interface: KioskGestureSource | None = None,
        engine: SuitJawaEngine | None = None,
        poll_interval: float = 1 / 30,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval harus lebih besar dari 0.")

        self.interface = (
            interface
            if interface is not None
            else GestureInterface(
                model_path=model_path,
                camera_index=camera_index,
                preview_enabled=True,
            )
        )
        self.engine = (
            engine
            if engine is not None
            else SuitJawaEngine(
                gesture_interface=self.interface,
                logger=GameLogger(log_path),
            )
        )
        self.poll_interval = poll_interval
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._camera_error: str | None = None
        self._game_error: str | None = None
        self._last_prediction = GesturePrediction("unknown", 0.0, False, False)

    @property
    def camera_error(self) -> str | None:
        with self._lock:
            return self._camera_error

    @property
    def latest_frame_jpeg(self) -> bytes | None:
        return self.interface.latest_frame_jpeg

    @property
    def camera_ready(self) -> bool:
        with self._lock:
            return (
                self._camera_error is None
                and self._worker is not None
                and self._worker.is_alive()
            )

    def start(self) -> None:
        """Initialize the local camera and start its prediction worker."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._camera_error = None
            self._game_error = None
            try:
                self.interface.start()
            except (OSError, RuntimeError, ValueError) as error:
                self._camera_error = str(error)
                return

            self._stop_event.clear()
            self._worker = threading.Thread(
                target=self._capture_loop,
                name="suit-jawa-kiosk-camera",
                daemon=True,
            )
            self._worker.start()

    def start_game(self) -> None:
        with self._lock:
            if not self.camera_ready:
                raise CameraUnavailableError(
                    self._camera_error or "Webcam belum siap."
                )
            self._game_error = None
            self.engine.start()

    def state_snapshot(self) -> dict[str, object]:
        with self._lock:
            statistics = self.engine.prediction_statistics
            forecast = self.engine.forecast
            last_result = self.engine.last_result
            return {
                "state": self.engine.state.value,
                "message": self.engine.message,
                "current_round": self.engine.current_round,
                "total_rounds": self.engine.total_rounds,
                "scores": {
                    "player": self.engine.player_score,
                    "ai": self.engine.ai_score,
                },
                "gesture": {
                    "label": self._last_prediction.label,
                    "confidence": self._last_prediction.confidence,
                    "is_valid": self._last_prediction.is_valid,
                    "hand_detected": self._last_prediction.hand_detected,
                },
                "forecast": (
                    {
                        "predicted_gesture": forecast.predicted_gesture,
                        "probabilities": forecast.probabilities,
                        "confidence": forecast.confidence,
                        "strategy": forecast.strategy,
                    }
                    if forecast is not None
                    else None
                ),
                "ai_move": self.engine.planned_ai_gesture,
                "prediction_statistics": {
                    "correct": statistics.correct_predictions,
                    "total": statistics.evaluated_rounds,
                    "accuracy": statistics.accuracy,
                },
                "last_result": (
                    {
                        "round_number": last_result.round_number,
                        "player_gesture": last_result.player_gesture,
                        "ai_gesture": last_result.ai_gesture,
                        "winner": last_result.winner,
                        "message": last_result.message,
                    }
                    if last_result is not None
                    else None
                ),
                "history": [
                    {
                        "round_number": result.round_number,
                        "player_gesture": result.player_gesture,
                        "ai_gesture": result.ai_gesture,
                        "predicted_player_gesture": (
                            result.predicted_player_gesture
                        ),
                        "winner": result.winner,
                        "message": result.message,
                    }
                    for result in self.engine.round_history
                ],
                "camera": {
                    "ready": self.camera_ready,
                    "error": self._camera_error,
                },
                "game_error": self._game_error,
            }

    def mjpeg_frames(self) -> Iterator[bytes]:
        while not self._stop_event.is_set():
            frame = self.latest_frame_jpeg
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            elif self.camera_error is not None:
                return
            self._stop_event.wait(1 / 24)

    def close(self) -> None:
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=2)
        self.interface.close()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                prediction = self.interface.get_prediction()
            except CameraDisconnectedError as error:
                self._set_camera_error(str(error))
                return
            except (OSError, RuntimeError, ValueError) as error:
                self._set_camera_error(str(error))
                return

            try:
                with self._lock:
                    self._last_prediction = prediction
                    if self.engine.state not in {
                        GameState.READY,
                        GameState.FINISHED,
                    }:
                        self.engine.update(prediction)
            except (OSError, RuntimeError, ValueError) as error:
                with self._lock:
                    self._game_error = str(error)
                self._stop_event.set()
                return

            self._stop_event.wait(self.poll_interval)

    def _set_camera_error(self, message: str) -> None:
        with self._lock:
            self._camera_error = message
        self._stop_event.set()


def create_app(controller: KioskController | None = None) -> Flask:
    kiosk = controller if controller is not None else KioskController()
    app = Flask(__name__, static_folder=str(WEB_ROOT), static_url_path="/static")

    @app.get("/")
    def index() -> Response:
        return send_from_directory(WEB_ROOT, "index.html")

    @app.get("/images/<gesture>")
    def gesture_image(gesture: str) -> Response:
        filename = GESTURE_IMAGES.get(gesture.lower())
        if filename is None:
            abort(404)
        return send_from_directory(IMAGE_ROOT, filename, max_age=3600)

    @app.get("/api/state")
    def state() -> Response:
        return jsonify(kiosk.state_snapshot())

    @app.post("/api/start")
    def start_game() -> tuple[Response, int] | Response:
        try:
            kiosk.start_game()
        except CameraUnavailableError as error:
            return jsonify({"error": str(error)}), 503
        except RuntimeError as error:
            return jsonify({"error": str(error)}), 409
        return jsonify(kiosk.state_snapshot())

    @app.get("/video_feed")
    def video_feed() -> Response | tuple[Response, int]:
        if not kiosk.camera_ready:
            return (
                jsonify(
                    {
                        "error": kiosk.camera_error
                        or "Webcam belum siap untuk menampilkan preview."
                    }
                ),
                503,
            )
        if kiosk.latest_frame_jpeg is None:
            return jsonify({"error": "Frame kamera belum tersedia."}), 503
        return Response(
            kiosk.mjpeg_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    app.config["KIOSK_CONTROLLER"] = kiosk
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Jalankan kiosk Suit Jawa vs AI secara lokal dan offline."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--port", type=int, default=5000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.camera_index < 0:
        parser.error("--camera-index tidak boleh negatif.")
    if not 1 <= args.port <= 65535:
        parser.error("--port harus berada pada rentang 1–65535.")

    controller = KioskController(
        camera_index=args.camera_index,
        model_path=args.model,
        log_path=args.log_file,
    )
    controller.start()
    print(f"Kiosk offline: http://127.0.0.1:{args.port}")
    if controller.camera_error:
        print(f"Kamera tidak siap: {controller.camera_error}")
    app = create_app(controller)
    try:
        app.run(
            host="127.0.0.1",
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
