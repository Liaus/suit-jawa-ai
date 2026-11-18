import csv
from pathlib import Path
import tempfile
import unittest
import uuid

from src.game import GameLogger, GameState, SuitJawaEngine
from src.game.game_logger import CSV_COLUMNS
from src.vision.gesture_interface import GesturePrediction


class GameLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self._temporary_directory.name) / "game_logs.csv"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def read_rows(self) -> list[dict[str, str]]:
        with self.log_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def log_sample_round(
        self,
        logger: GameLogger,
        round_number: int,
        *,
        player_gesture: str | None = "Gajah",
        winner: str = "player",
    ) -> None:
        logger.log_round(
            round_number=round_number,
            player_gesture=player_gesture,
            ai_prediction="Orang",
            ai_move="Gajah",
            winner=winner,
            player_score=round_number,
            ai_score=0,
        )

    def test_append_only_across_anonymous_sessions_and_has_no_pii_columns(self) -> None:
        logger = GameLogger(self.log_path)
        first_session_id = logger.start_session()
        self.log_sample_round(logger, 1)
        self.log_sample_round(logger, 2)

        second_session_id = logger.start_session()
        self.log_sample_round(logger, 1)

        rows = self.read_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(rows[0]), CSV_COLUMNS)
        self.assertEqual(
            [row["round_number"] for row in rows],
            ["1", "2", "1"],
        )
        self.assertEqual(
            {row["session_id"] for row in rows},
            {first_session_id, second_session_id},
        )
        self.assertNotEqual(first_session_id, second_session_id)
        self.assertTrue(all(uuid.UUID(row["session_id"]).version == 4 for row in rows))
        self.assertEqual(
            set(CSV_COLUMNS),
            {
                "session_id",
                "round_number",
                "player_gesture",
                "ai_prediction",
                "ai_move",
                "winner",
                "player_score",
                "ai_score",
            },
        )

    def test_rejects_duplicate_round_number_in_same_session(self) -> None:
        logger = GameLogger(self.log_path)
        self.log_sample_round(logger, 1)

        with self.assertRaisesRegex(ValueError, "sudah dicatat"):
            self.log_sample_round(logger, 1)

        self.assertEqual(len(self.read_rows()), 1)

    def test_rejects_invalid_gesture_text(self) -> None:
        logger = GameLogger(self.log_path)

        with self.assertRaisesRegex(ValueError, "player_gesture"):
            self.log_sample_round(logger, 1, player_gesture="not-a-gesture")

        self.assertFalse(self.log_path.exists())

    def test_engine_writes_one_row_for_valid_and_timeout_rounds(self) -> None:
        logger = GameLogger(self.log_path)
        engine = SuitJawaEngine(
            logger=logger,
            total_rounds=2,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            gesture_timeout=2,
            stable_seconds=0,
            result_seconds=1,
        )
        engine.start(now=0)

        self.assertEqual(engine.update(now=0), GameState.COUNTER_REVEAL)
        self.assertEqual(engine.update(now=0), GameState.COLLECTING_GESTURE)
        self.assertEqual(
            engine.update(
                GesturePrediction("Gajah", 0.95, True, True),
                now=0,
            ),
            GameState.ROUND_RESULT,
        )
        engine.update(now=0)
        self.assertEqual(engine.state, GameState.ROUND_RESULT)

        engine.update(now=1)
        self.assertEqual(engine.update(now=1), GameState.COUNTER_REVEAL)
        self.assertEqual(engine.update(now=1), GameState.COLLECTING_GESTURE)
        self.assertEqual(engine.update(now=3), GameState.ROUND_RESULT)

        rows = self.read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["round_number"] for row in rows], ["1", "2"])
        self.assertEqual(rows[0]["player_gesture"], "Gajah")
        self.assertIn(rows[0]["winner"], {"player", "ai", "draw"})
        self.assertEqual(rows[1]["player_gesture"], "")
        self.assertEqual(rows[1]["winner"], "none")
        self.assertTrue(all(row["ai_prediction"] for row in rows))
        self.assertTrue(all(row["ai_move"] for row in rows))


if __name__ == "__main__":
    unittest.main()
