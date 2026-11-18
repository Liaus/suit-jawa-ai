import csv
from pathlib import Path
import tempfile
import time
import unittest

from main import build_parser, run_game
from src.game import GameLogger
from src.vision import CameraDisconnectedError, GesturePrediction


class ScriptedGestureInterface:
    confidence_threshold = 0.60

    def __init__(
        self,
        responses: tuple[GesturePrediction | Exception, ...] = (),
    ) -> None:
        self._responses = list(responses)
        self.started = False
        self.closed = False
        self.calls = 0

    def start(self) -> None:
        self.started = True

    def get_prediction(self) -> GesturePrediction:
        self.calls += 1
        response: GesturePrediction | Exception
        if self._responses:
            response = self._responses.pop(0)
        else:
            response = GesturePrediction("unknown", 0.0, False, False)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class GameIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self._temporary_directory.name) / "game_logs.csv"
        self.logger = GameLogger(self.log_path)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def read_rows(self) -> list[dict[str, str]]:
        with self.log_path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_offline_game_runs_one_round_and_logs_local_result(self) -> None:
        interface = ScriptedGestureInterface(
            (GesturePrediction("Gajah", 0.95, True, True),)
        )
        output: list[str] = []

        exit_code = run_game(
            mode="offline",
            total_rounds=1,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            stable_seconds=0,
            result_seconds=0,
            gesture_interface=interface,
            logger=self.logger,
            output=output.append,
            sleep=lambda _: None,
        )

        rows = self.read_rows()
        self.assertEqual(exit_code, 0)
        self.assertTrue(interface.started)
        self.assertTrue(interface.closed)
        self.assertTrue(any("Mode offline" in line for line in output))
        self.assertTrue(any("Permainan selesai" in line for line in output))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["round_number"], "1")
        self.assertEqual(rows[0]["player_gesture"], "Gajah")
        self.assertIn(rows[0]["winner"], {"player", "ai", "draw"})

    def test_unstable_gesture_times_out_and_is_logged(self) -> None:
        interface = ScriptedGestureInterface()
        output: list[str] = []

        exit_code = run_game(
            total_rounds=1,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            gesture_timeout=0.05,
            stable_seconds=0,
            result_seconds=0,
            poll_interval=0.005,
            gesture_interface=interface,
            logger=self.logger,
            output=output.append,
            sleep=time.sleep,
        )

        rows = self.read_rows()
        self.assertEqual(exit_code, 0)
        self.assertGreater(interface.calls, 0)
        self.assertTrue(any("tidak stabil" in line for line in output))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["player_gesture"], "")
        self.assertEqual(rows[0]["winner"], "none")

    def test_camera_disconnect_is_reported_and_resources_are_closed(self) -> None:
        interface = ScriptedGestureInterface(
            (
                CameraDisconnectedError("frame tidak terbaca"),
            )
        )
        output: list[str] = []

        exit_code = run_game(
            total_rounds=1,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            gesture_interface=interface,
            logger=self.logger,
            output=output.append,
            sleep=lambda _: None,
        )

        self.assertEqual(exit_code, 1)
        self.assertTrue(interface.closed)
        self.assertTrue(any("Kamera terputus" in line for line in output))
        self.assertFalse(self.log_path.exists())

    def test_repeated_games_append_new_anonymous_sessions(self) -> None:
        for _ in range(2):
            interface = ScriptedGestureInterface(
                (GesturePrediction("Semut", 0.9, True, True),)
            )
            exit_code = run_game(
                total_rounds=1,
                countdown_seconds=0,
                counter_reveal_seconds=0,
                stable_seconds=0,
                result_seconds=0,
                gesture_interface=interface,
                logger=self.logger,
                output=lambda _: None,
                sleep=lambda _: None,
            )
            self.assertEqual(exit_code, 0)

        rows = self.read_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["round_number"] for row in rows], ["1", "1"])
        self.assertNotEqual(rows[0]["session_id"], rows[1]["session_id"])

    def test_full_ten_round_session_writes_ten_rows(self) -> None:
        interface = ScriptedGestureInterface(
            tuple(
                GesturePrediction("Gajah", 0.95, True, True)
                for _ in range(10)
            )
        )

        exit_code = run_game(
            total_rounds=10,
            countdown_seconds=0,
            counter_reveal_seconds=0,
            stable_seconds=0,
            result_seconds=0,
            gesture_interface=interface,
            logger=self.logger,
            output=lambda _: None,
            sleep=lambda _: None,
        )

        rows = self.read_rows()
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["round_number"] for row in rows],
            [str(round_number) for round_number in range(1, 11)],
        )
        self.assertEqual(len({row["session_id"] for row in rows}), 1)

    def test_parser_defaults_to_offline_mode(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.mode, "offline")
        self.assertEqual(args.rounds, 10)


if __name__ == "__main__":
    unittest.main()
