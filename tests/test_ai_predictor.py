import random
import unittest
from collections.abc import Sequence
from pathlib import Path
import tempfile

from src.game import (
    AICounter,
    AIPrediction,
    AIPredictor,
    GameLogger,
    GameState,
    SuitJawaEngine,
)
from src.vision.gesture_interface import GesturePrediction


class FixedPredictor:
    def __init__(self) -> None:
        self.histories: list[tuple[str, ...]] = []

    def predict(self, history: Sequence[str]) -> AIPrediction:
        self.histories.append(tuple(history))
        return AIPrediction(
            predicted_gesture="Gajah",
            probabilities={"Gajah": 0.8, "Orang": 0.1, "Semut": 0.1},
            confidence=0.8,
            strategy="frequency",
        )


class OrderedGestureSource:
    confidence_threshold = 0.6

    def __init__(
        self,
        events: list[str],
        predictions: tuple[str, ...],
    ) -> None:
        self.events = events
        self.engine: SuitJawaEngine | None = None
        self._predictions = iter(predictions)

    def get_prediction(self) -> GesturePrediction:
        self.events.append("read_player_input")
        if self.engine is None or self.engine.planned_ai_gesture is None:
            raise AssertionError("AI counter must be planned before reading input.")
        return GesturePrediction(next(self._predictions), 0.95, True, True)

    def close(self) -> None:
        pass


class AIPredictorTests(unittest.TestCase):
    def test_empty_history_returns_uniform_prior_and_deterministic_tie_choice(self) -> None:
        predictor = AIPredictor(rng=random.Random(1))

        forecast = predictor.predict([])

        self.assertEqual(forecast.strategy, "prior")
        self.assertEqual(forecast.probabilities, {
            "Gajah": 1 / 3,
            "Orang": 1 / 3,
            "Semut": 1 / 3,
        })
        self.assertIn(forecast.predicted_gesture, ("Gajah", "Orang", "Semut"))
        self.assertAlmostEqual(forecast.confidence, 1 / 3)

    def test_frequency_model_uses_most_common_player_move(self) -> None:
        predictor = AIPredictor(rng=random.Random(1))

        forecast = predictor.predict(["Gajah", "Gajah", "Orang", "Semut"])

        self.assertEqual(forecast.strategy, "frequency")
        self.assertEqual(forecast.predicted_gesture, "Gajah")
        self.assertGreater(forecast.probabilities["Gajah"], forecast.probabilities["Orang"])

    def test_markov_order_one_can_override_global_frequency(self) -> None:
        history = [
            "Semut",
            "Gajah",
            "Gajah",
            "Gajah",
            "Gajah",
            "Orang",
            "Semut",
            "Orang",
            "Semut",
            "Orang",
            "Semut",
        ]
        predictor = AIPredictor(rng=random.Random(1))

        forecast = predictor.predict(history)

        self.assertEqual(forecast.strategy, "markov+frequency")
        self.assertEqual(forecast.predicted_gesture, "Orang")

    def test_frequency_tie_breaks_only_between_equal_probability_moves(self) -> None:
        predictor = AIPredictor(rng=random.Random(5))

        forecast = predictor.predict(["Gajah", "Orang"])

        tied = {
            gesture
            for gesture, probability in forecast.probabilities.items()
            if probability == max(forecast.probabilities.values())
        }
        self.assertEqual(tied, {"Gajah", "Orang"})
        self.assertIn(forecast.predicted_gesture, tied)

    def test_unknown_gesture_is_rejected_from_player_history(self) -> None:
        predictor = AIPredictor()

        with self.assertRaisesRegex(ValueError, "Riwayat hanya boleh"):
            predictor.predict(["Gajah", "unknown"])


class AICounterTests(unittest.TestCase):
    def test_counter_beats_each_predicted_player_move(self) -> None:
        counter = AICounter()

        self.assertEqual(counter.counter_for("Gajah"), "Semut")
        self.assertEqual(counter.counter_for("Orang"), "Gajah")
        self.assertEqual(counter.counter_for("Semut"), "Orang")

    def test_unknown_cannot_be_used_as_a_counter_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "Prediksi pemain"):
            AICounter().counter_for("unknown")


class AIEngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_counter_is_prepared_and_revealed_before_player_input(self) -> None:
        events: list[str] = []
        predictor = FixedPredictor()
        source = OrderedGestureSource(events, ("Orang", "Semut"))
        engine = SuitJawaEngine(
            gesture_interface=source,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            gesture_timeout=2,
            stable_seconds=0,
            result_seconds=0,
            predictor=predictor,
            rng=random.Random(1),
            logger=GameLogger(
                Path(self._temporary_directory.name) / "game_logs.csv"
            ),
        )
        source.engine = engine

        engine.start(now=0)
        self.assertEqual(engine.update(now=0), GameState.COUNTER_REVEAL)
        events.append("counter_revealed")
        self.assertEqual(engine.planned_ai_gesture, "Semut")
        self.assertEqual(engine.message, "Counter AI: Semut (prediksi: Gajah)")
        self.assertEqual(predictor.histories, [()])

        self.assertEqual(engine.update(now=0), GameState.COLLECTING_GESTURE)
        self.assertEqual(engine.update(now=0), GameState.ROUND_RESULT)
        events.append("round_resolved")

        engine.update(now=0)
        self.assertEqual(engine.update(now=0), GameState.COUNTER_REVEAL)
        events.append("counter_revealed")
        self.assertEqual(predictor.histories, [(), ("Orang",)])
        self.assertEqual(engine.update(now=0), GameState.COLLECTING_GESTURE)
        self.assertEqual(engine.update(now=0), GameState.ROUND_RESULT)
        events.append("round_resolved")

        self.assertEqual(
            events,
            [
                "counter_revealed",
                "read_player_input",
                "round_resolved",
                "counter_revealed",
                "read_player_input",
                "round_resolved",
            ],
        )
        self.assertEqual(engine.round_history[0].player_gesture, "Orang")
        self.assertEqual(engine.round_history[0].ai_gesture, "Semut")
        self.assertEqual(engine.round_history[0].predicted_player_gesture, "Gajah")
        self.assertEqual(engine.round_history[1].player_gesture, "Semut")
        self.assertEqual(engine.round_history[1].predicted_player_gesture, "Gajah")


if __name__ == "__main__":
    unittest.main()
