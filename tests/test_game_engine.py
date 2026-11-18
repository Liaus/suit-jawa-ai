import random
from pathlib import Path
import tempfile
import unittest

from src.game import GameLogger, GameState, RoundResult, SuitJawaEngine
from src.vision.gesture_interface import GesturePrediction


def valid_prediction(label: str) -> GesturePrediction:
    return GesturePrediction(label, 0.95, True, True)


class FakeGestureInterface:
    confidence_threshold = 0.60

    def __init__(self, prediction: GesturePrediction) -> None:
        self.prediction = prediction
        self.calls = 0

    def get_prediction(self) -> GesturePrediction:
        self.calls += 1
        return self.prediction

    def close(self) -> None:
        pass


class SuitJawaRulesTests(unittest.TestCase):
    def test_all_nine_suit_combinations(self) -> None:
        expected = {
            ("Gajah", "Gajah"): "draw",
            ("Gajah", "Orang"): "player",
            ("Gajah", "Semut"): "ai",
            ("Orang", "Gajah"): "ai",
            ("Orang", "Orang"): "draw",
            ("Orang", "Semut"): "player",
            ("Semut", "Gajah"): "player",
            ("Semut", "Orang"): "ai",
            ("Semut", "Semut"): "draw",
        }

        for gestures, winner in expected.items():
            with self.subTest(player=gestures[0], ai=gestures[1]):
                self.assertEqual(
                    SuitJawaEngine.determine_winner(*gestures),
                    winner,
                )


class SuitJawaRoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def make_engine(self, **kwargs) -> SuitJawaEngine:
        options = {
            "countdown_seconds": 0,
            "counter_reveal_seconds": 0,
            "gesture_timeout": 2,
            "stable_seconds": 0.5,
            "result_seconds": 0.25,
            "rng": random.Random(1),
            "logger": GameLogger(
                Path(self._temporary_directory.name) / "game_logs.csv"
            ),
        }
        options.update(kwargs)
        return SuitJawaEngine(**options)

    def advance_to_capture(self, engine: SuitJawaEngine, now: float) -> None:
        self.assertEqual(engine.update(now=now), GameState.COUNTER_REVEAL)
        self.assertEqual(engine.update(now=now), GameState.COLLECTING_GESTURE)

    def require_result(self, engine: SuitJawaEngine) -> RoundResult:
        result = engine.last_result
        if result is None:
            self.fail("Expected a completed round result.")
        return result

    def test_stable_valid_gesture_resolves_round_and_updates_score(self) -> None:
        engine = self.make_engine()
        engine.start(now=0)

        self.advance_to_capture(engine, 0)
        self.assertEqual(
            engine.update(valid_prediction("Gajah"), now=0),
            GameState.COLLECTING_GESTURE,
        )
        self.assertEqual(
            engine.update(valid_prediction("Gajah"), now=0.5),
            GameState.ROUND_RESULT,
        )
        result = self.require_result(engine)
        self.assertEqual(result.player_gesture, "Gajah")
        self.assertEqual(result.round_number, 1)

    def test_unknown_is_ignored_until_timeout_reports_unstable_gesture(self) -> None:
        engine = self.make_engine()
        engine.start(now=0)
        self.advance_to_capture(engine, 0)

        engine.update(
            GesturePrediction("unknown", 0.9, False, True),
            now=0.2,
        )
        self.assertEqual(engine.state, GameState.COLLECTING_GESTURE)
        self.assertIsNone(engine.last_result)

        engine.update(now=2)
        self.assertEqual(engine.state, GameState.ROUND_RESULT)
        self.assertEqual(engine.message, "Gestur tidak stabil")
        self.assertEqual(self.require_result(engine).winner, "none")
        self.assertEqual(engine.player_score, 0)
        self.assertEqual(engine.ai_score, 0)

    def test_changing_gestures_restarts_stability_window(self) -> None:
        engine = self.make_engine(gesture_timeout=1.0, stable_seconds=0.5)
        engine.start(now=0)
        self.advance_to_capture(engine, 0)
        engine.update(valid_prediction("Gajah"), now=0)
        engine.update(valid_prediction("Orang"), now=0.4)
        engine.update(valid_prediction("Orang"), now=0.8)

        self.assertEqual(engine.state, GameState.COLLECTING_GESTURE)
        self.assertIsNone(engine.last_result)

        engine.update(now=1.0)
        self.assertEqual(engine.message, "Gestur tidak stabil")

    def test_missing_hand_resets_candidate_stability(self) -> None:
        engine = self.make_engine(gesture_timeout=1.0, stable_seconds=0.4)
        engine.start(now=0)
        self.advance_to_capture(engine, 0)
        engine.update(valid_prediction("Gajah"), now=0)
        engine.update(GesturePrediction("Gajah", 0.99, True, False), now=0.2)
        engine.update(valid_prediction("Gajah"), now=0.3)
        engine.update(valid_prediction("Gajah"), now=0.6)

        self.assertEqual(engine.state, GameState.COLLECTING_GESTURE)
        self.assertIsNone(engine.last_result)

        engine.update(valid_prediction("Gajah"), now=0.71)
        self.assertEqual(engine.state, GameState.ROUND_RESULT)

    def test_engine_polls_injected_gesture_interface(self) -> None:
        interface = FakeGestureInterface(valid_prediction("Semut"))
        engine = self.make_engine(
            gesture_interface=interface,
            stable_seconds=0,
        )
        engine.start(now=0)
        self.advance_to_capture(engine, 0)
        engine.update(now=0)

        self.assertEqual(interface.calls, 1)
        self.assertEqual(engine.state, GameState.ROUND_RESULT)
        self.assertEqual(self.require_result(engine).player_gesture, "Semut")

    def test_ten_rounds_progress_automatically_to_finished(self) -> None:
        engine = self.make_engine(
            total_rounds=10,
            stable_seconds=0,
            result_seconds=0,
        )
        engine.start(now=0)
        now = 0.0
        for round_number in range(1, 11):
            self.advance_to_capture(engine, now)
            engine.update(valid_prediction("Orang"), now=now)
            self.assertEqual(
                self.require_result(engine).round_number,
                round_number,
            )
            if round_number < 10:
                engine.update(now=now)
                self.assertEqual(engine.current_round, round_number + 1)
            now += 0.1

        engine.update(now=now)
        self.assertEqual(engine.state, GameState.FINISHED)
        self.assertEqual(len(engine.results), 10)

    def test_start_is_required_and_invalid_time_is_rejected(self) -> None:
        engine = self.make_engine()
        with self.assertRaisesRegex(RuntimeError, "start"):
            engine.update(now=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            engine.start(now=float("nan"))


if __name__ == "__main__":
    unittest.main()
