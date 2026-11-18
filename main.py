"""Run the offline webcam game Suit Jawa vs AI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import math
from pathlib import Path
import time
from typing import Protocol

from src.game.game_engine import GameState, MAX_ROUNDS, SuitJawaEngine
from src.game.game_logger import DEFAULT_LOG_PATH, GameLogger
from src.vision.gesture_interface import (
    DEFAULT_MODEL,
    CameraDisconnectedError,
    GestureInterface,
    GesturePrediction,
)


class GestureSource(Protocol):
    confidence_threshold: float

    def start(self) -> None: ...

    def get_prediction(self) -> GesturePrediction: ...

    def close(self) -> None: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mainkan Suit Jawa melawan AI secara lokal melalui webcam."
    )
    parser.add_argument(
        "--mode",
        choices=("offline",),
        default="offline",
        help="Mode offline memakai model lokal dan tidak membutuhkan internet.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indeks webcam yang digunakan (default: 0).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Lokasi artifact classifier lokal.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Lokasi CSV riwayat permainan.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=MAX_ROUNDS,
        help=f"Jumlah ronde (default: {MAX_ROUNDS}).",
    )
    return parser


def run_game(
    *,
    mode: str = "offline",
    camera_index: int = 0,
    model_path: Path = DEFAULT_MODEL,
    log_path: Path = DEFAULT_LOG_PATH,
    total_rounds: int = MAX_ROUNDS,
    confidence_threshold: float | None = None,
    countdown_seconds: float = 3.0,
    gesture_timeout: float = 5.0,
    stable_seconds: float = 0.35,
    counter_reveal_seconds: float = 1.0,
    result_seconds: float = 1.5,
    poll_interval: float = 0.01,
    gesture_interface: GestureSource | None = None,
    logger: GameLogger | None = None,
    output: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Run one game session; injected interfaces allow hardware-free tests."""
    if mode != "offline":
        output(f"Mode tidak didukung: {mode}")
        return 2
    if camera_index < 0:
        output("Indeks kamera tidak boleh negatif.")
        return 2
    if total_rounds < 1:
        output("Jumlah ronde harus minimal 1.")
        return 2
    if not math.isfinite(poll_interval) or poll_interval < 0:
        output("Interval polling harus finite dan tidak negatif.")
        return 2

    output("Mode offline: model dan prediksi berjalan lokal tanpa internet.")
    interface: GestureSource | None = gesture_interface
    engine: SuitJawaEngine | None = None
    try:
        if interface is None:
            interface = GestureInterface(
                model_path=model_path,
                camera_index=camera_index,
                confidence_threshold=confidence_threshold,
            )
        interface.start()
        game_logger = logger if logger is not None else GameLogger(log_path)
        engine = SuitJawaEngine(
            gesture_interface=interface,
            total_rounds=total_rounds,
            countdown_seconds=countdown_seconds,
            gesture_timeout=gesture_timeout,
            stable_seconds=stable_seconds,
            counter_reveal_seconds=counter_reveal_seconds,
            result_seconds=result_seconds,
            confidence_threshold=confidence_threshold,
            logger=game_logger,
        )
        engine.start()
        last_announced_state: GameState | None = None

        while engine.state is not GameState.FINISHED:
            state = engine.update()
            if state is not last_announced_state:
                _announce_state(engine, state, total_rounds, output)
                last_announced_state = state
            if state is not GameState.FINISHED:
                sleep(poll_interval)

        output(
            f"Permainan selesai. Skor akhir: pemain {engine.player_score} - "
            f"AI {engine.ai_score}."
        )
        return 0
    except CameraDisconnectedError as error:
        output(f"Kamera terputus: {error}")
        return 1
    except (OSError, RuntimeError, ValueError) as error:
        output(f"Permainan gagal: {error}")
        return 1
    finally:
        if engine is not None:
            engine.close()
        elif interface is not None:
            interface.close()


def _announce_state(
    engine: SuitJawaEngine,
    state: GameState,
    total_rounds: int,
    output: Callable[[str], None],
) -> None:
    if state is GameState.COUNTDOWN:
        output(f"Ronde {engine.current_round}/{total_rounds}: bersiap.")
    elif state is GameState.COUNTER_REVEAL:
        output(engine.message)
    elif state is GameState.COLLECTING_GESTURE:
        output("Suit! Tunjukkan gestur dan pertahankan agar stabil.")
    elif state is GameState.ROUND_RESULT:
        result = engine.last_result
        if result is None:
            raise RuntimeError("Hasil ronde tidak tersedia.")
        player_move = result.player_gesture or "tidak stabil"
        ai_move = result.ai_gesture or engine.planned_ai_gesture or "-"
        output(
            f"Ronde {result.round_number}: pemain {player_move}, AI {ai_move}; "
            f"{result.message}. Skor {result.player_score}-{result.ai_score}."
        )


def main(
    argv: Sequence[str] | None = None,
    *,
    gesture_interface: GestureSource | None = None,
    logger: GameLogger | None = None,
    output: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    return run_game(
        mode=args.mode,
        camera_index=args.camera_index,
        model_path=args.model,
        log_path=args.log_file,
        total_rounds=args.rounds,
        gesture_interface=gesture_interface,
        logger=logger,
        output=output,
        sleep=sleep,
    )


if __name__ == "__main__":
    raise SystemExit(main())
