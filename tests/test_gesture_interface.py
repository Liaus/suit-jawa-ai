from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.vision.gesture_interface import (
    CLASS_ORDER,
    CameraDisconnectedError,
    GestureInterface,
)


class FakeModel:
    classes_ = CLASS_ORDER
    n_features_in_ = 63

    def __init__(self, label: str, probabilities: list[float]) -> None:
        self.label = label
        self.probabilities = probabilities

    def predict(self, features: list[list[float]]) -> list[str]:
        _ = features
        return [self.label]

    def predict_proba(self, features: list[list[float]]) -> list[list[float]]:
        _ = features
        return [self.probabilities]


class FakeCapture:
    def __init__(self) -> None:
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: int) -> None:
        _ = property_id, value

    def read(self) -> tuple[bool, str | None]:
        return True, "frame"

    def release(self) -> None:
        self.opened = False


class DisconnectedCapture(FakeCapture):
    def read(self) -> tuple[bool, str | None]:
        return False, None


class FakeCV2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    COLOR_BGR2RGB = 42

    def __init__(self) -> None:
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, camera_index: int) -> FakeCapture:
        _ = camera_index
        capture = FakeCapture()
        self.captures.append(capture)
        return capture

    def flip(self, frame: str, flip_code: int) -> str:
        _ = flip_code
        return frame

    def cvtColor(self, frame: str, conversion: int) -> str:
        _ = conversion
        return frame

    def imencode(self, extension: str, frame: str) -> tuple[bool, "FakeEncodedImage"]:
        _ = extension, frame
        return True, FakeEncodedImage()


class FakeEncodedImage:
    def tobytes(self) -> bytes:
        return b"jpeg-frame"


class FakeHandProcessor:
    def __init__(self, has_hand: bool = True) -> None:
        self.has_hand = has_hand

    def process(self, frame: str) -> SimpleNamespace:
        _ = frame
        landmarks = [object()] if self.has_hand else None
        return SimpleNamespace(multi_hand_landmarks=landmarks)

    def close(self) -> None:
        pass


class FakeHandsFactory:
    def Hands(
        self,
        *,
        static_image_mode: bool,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> FakeHandProcessor:
        _ = (
            static_image_mode,
            max_num_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )
        return FakeHandProcessor()


def make_interface(
    label: str,
    probabilities: list[float],
    has_hand: bool = True,
    preview_enabled: bool = False,
) -> GestureInterface:
    return GestureInterface(
        model=FakeModel(label, probabilities),
        capture=FakeCapture(),
        hand_processor=FakeHandProcessor(has_hand),
        cv2_module=FakeCV2(),
        preview_enabled=preview_enabled,
    )


class GestureInterfaceTests(unittest.TestCase):
    @patch("src.vision.gesture_interface.normalize_hand_landmarks", return_value=[0.0] * 63)
    def test_valid_game_gesture_is_returned_with_confidence(self, normalize) -> None:
        interface = make_interface("Gajah", [0.92, 0.03, 0.03, 0.02])

        prediction = interface.get_prediction()

        self.assertEqual(prediction.label, "Gajah")
        self.assertAlmostEqual(prediction.confidence, 0.92)
        self.assertTrue(prediction.is_valid)
        self.assertTrue(prediction.hand_detected)
        normalize.assert_called_once()

    @patch("src.vision.gesture_interface.normalize_hand_landmarks", return_value=[0.0] * 63)
    def test_unknown_prediction_is_not_valid_for_the_game(self, normalize) -> None:
        interface = make_interface("unknown", [0.02, 0.03, 0.05, 0.90])

        prediction = interface.get_prediction()

        self.assertEqual(prediction.label, "unknown")
        self.assertFalse(prediction.is_valid)
        self.assertTrue(prediction.hand_detected)
        normalize.assert_called_once()

    @patch("src.vision.gesture_interface.normalize_hand_landmarks", return_value=[0.0] * 63)
    def test_low_confidence_known_prediction_becomes_unknown(self, normalize) -> None:
        interface = make_interface("Semut", [0.12, 0.12, 0.56, 0.20])

        prediction = interface.get_prediction()

        self.assertEqual(prediction.label, "unknown")
        self.assertAlmostEqual(prediction.confidence, 0.20)
        self.assertFalse(prediction.is_valid)
        self.assertTrue(prediction.hand_detected)
        normalize.assert_called_once()

    @patch("src.vision.gesture_interface.normalize_hand_landmarks")
    def test_no_detected_hand_immediately_returns_unknown(self, normalize) -> None:
        interface = make_interface("Gajah", [0.92, 0.03, 0.03, 0.02], has_hand=False)

        prediction = interface.get_prediction()

        self.assertEqual(prediction.label, "unknown")
        self.assertEqual(prediction.confidence, 0.0)
        self.assertFalse(prediction.is_valid)
        self.assertFalse(prediction.hand_detected)
        normalize.assert_not_called()

    def test_camera_disconnect_raises_specific_error(self) -> None:
        interface = GestureInterface(
            model=FakeModel("Gajah", [0.92, 0.03, 0.03, 0.02]),
            capture=DisconnectedCapture(),
            hand_processor=FakeHandProcessor(),
            cv2_module=FakeCV2(),
            camera_index=2,
        )

        with self.assertRaisesRegex(CameraDisconnectedError, "indeks 2"):
            interface.get_prediction()

    @patch("src.vision.gesture_interface.normalize_hand_landmarks", return_value=[0.0] * 63)
    def test_preview_enabled_caches_the_latest_jpeg_frame(self, normalize) -> None:
        interface = make_interface(
            "Gajah",
            [0.92, 0.03, 0.03, 0.02],
            preview_enabled=True,
        )

        interface.get_prediction()

        self.assertEqual(interface.latest_frame_jpeg, b"jpeg-frame")
        normalize.assert_called_once()

    def test_confidence_threshold_must_be_between_zero_and_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "antara 0 dan 1"):
            GestureInterface(
                model=FakeModel("Gajah", [1.0, 0.0, 0.0, 0.0]),
                confidence_threshold=1.5,
            )

    def test_close_releases_owned_resources_and_allows_restart(self) -> None:
        cv2 = FakeCV2()
        media_pipe = SimpleNamespace(
            solutions=SimpleNamespace(hands=FakeHandsFactory())
        )
        interface = GestureInterface(
            model=FakeModel("Gajah", [0.92, 0.03, 0.03, 0.02]),
            cv2_module=cv2,
        )

        with patch(
            "src.vision.gesture_interface._load_vision_dependencies",
            return_value=(cv2, media_pipe),
        ):
            interface.start()
            interface.close()
            interface.start()

        self.assertEqual(len(cv2.captures), 2)


if __name__ == "__main__":
    unittest.main()
