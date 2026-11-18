from pathlib import Path
import tempfile
import time
import unittest

from kiosk_app import KioskController, create_app
from src.game import GameLogger, SuitJawaEngine
from src.vision import GesturePrediction


class FakeGestureInterface:
    confidence_threshold = 0.60
    latest_frame_jpeg: bytes | None = b"fake-jpeg-frame"

    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def get_prediction(self) -> GesturePrediction:
        return GesturePrediction("unknown", 0.0, False, False)

    def close(self) -> None:
        self.closed = True


class FailingGestureInterface(FakeGestureInterface):
    def start(self) -> None:
        raise RuntimeError("izin kamera ditolak")


class KioskAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self._controllers: list[KioskController] = []

    def tearDown(self) -> None:
        for controller in self._controllers:
            controller.close()
        self._temporary_directory.cleanup()

    def make_controller(
        self,
        interface: FakeGestureInterface | None = None,
    ) -> KioskController:
        gesture_interface = interface or FakeGestureInterface()
        engine = SuitJawaEngine(
            gesture_interface=gesture_interface,
            total_rounds=1,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            gesture_timeout=0.04,
            stable_seconds=0,
            result_seconds=0,
            logger=GameLogger(
                Path(self._temporary_directory.name) / "game_logs.csv"
            ),
        )
        controller = KioskController(
            interface=gesture_interface,
            engine=engine,
            poll_interval=0.003,
        )
        self._controllers.append(controller)
        controller.start()
        return controller

    def test_serves_offline_page_and_local_gesture_art(self) -> None:
        app = create_app(self.make_controller())
        app.testing = True
        client = app.test_client()

        page = client.get("/")
        try:
            self.assertEqual(page.status_code, 200)
            self.assertIn(b"/static/styles.css", page.data)
            self.assertNotIn(b"https://", page.data)
        finally:
            page.close()

        for gesture in ("gajah", "orang", "semut"):
            with self.subTest(gesture=gesture):
                image = client.get(f"/images/{gesture}")
                try:
                    self.assertEqual(image.status_code, 200)
                    self.assertEqual(image.mimetype, "image/jpeg")
                finally:
                    image.close()

        self.assertEqual(client.get("/images/not-a-gesture").status_code, 404)
        script = client.get("/static/app.js")
        try:
            self.assertEqual(script.status_code, 200)
        finally:
            script.close()

    def test_game_api_streams_camera_and_reports_completed_rounds(self) -> None:
        controller = self.make_controller()
        app = create_app(controller)
        app.testing = True
        client = app.test_client()

        initial = client.get("/api/state").get_json()
        self.assertTrue(initial["camera"]["ready"])
        self.assertEqual(initial["prediction_statistics"]["total"], 0)

        stream = client.get("/video_feed", buffered=False)
        try:
            self.assertEqual(stream.status_code, 200)
            self.assertIn(b"--frame", next(iter(stream.response)))
        finally:
            stream.close()

        started = client.post("/api/start")
        self.assertEqual(started.status_code, 200)
        deadline = time.monotonic() + 1
        state = started.get_json()
        while state["state"] != "finished" and time.monotonic() < deadline:
            time.sleep(0.01)
            state = client.get("/api/state").get_json()

        self.assertEqual(state["state"], "finished")
        self.assertEqual(state["history"][0]["message"], "Gestur tidak stabil")
        self.assertEqual(state["prediction_statistics"]["total"], 0)
        self.assertEqual(state["prediction_statistics"]["accuracy"], None)

    def test_camera_start_failure_is_reported_by_api(self) -> None:
        controller = self.make_controller(FailingGestureInterface())
        app = create_app(controller)
        app.testing = True
        client = app.test_client()

        state = client.get("/api/state").get_json()
        self.assertFalse(state["camera"]["ready"])
        self.assertEqual(state["camera"]["error"], "izin kamera ditolak")
        self.assertEqual(client.post("/api/start").status_code, 503)
        self.assertEqual(client.get("/video_feed").status_code, 503)


if __name__ == "__main__":
    unittest.main()
