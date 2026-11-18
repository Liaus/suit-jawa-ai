"""UI-independent finite-state game engine for Suit Jawa."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
import time
from typing import Literal, Protocol, Sequence

from src.game.ai_counter import AICounter
from src.game.ai_predictor import AIPrediction, AIPredictor
from src.game.game_logger import GameLogger
from src.vision.gesture_interface import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    GesturePrediction,
)


GESTURES = ("Gajah", "Orang", "Semut")
BEATS = {"Gajah": "Orang", "Orang": "Semut", "Semut": "Gajah"}
MAX_ROUNDS = 10


class GameState(str, Enum):
    READY = "ready"
    COUNTDOWN = "countdown"
    COUNTER_REVEAL = "counter_reveal"
    COLLECTING_GESTURE = "collecting_gesture"
    ROUND_RESULT = "round_result"
    FINISHED = "finished"


Winner = Literal["player", "ai", "draw", "none"]


@dataclass(frozen=True)
class RoundResult:
    round_number: int
    player_gesture: str | None
    ai_gesture: str | None
    winner: Winner
    message: str
    player_score: int
    ai_score: int
    predicted_player_gesture: str | None = None


@dataclass(frozen=True)
class PredictionStatistics:
    correct_predictions: int
    evaluated_rounds: int
    accuracy: float | None


class GestureProvider(Protocol):
    """The small part of GestureInterface consumed by the game engine."""

    confidence_threshold: float

    def get_prediction(self) -> GesturePrediction: ...

    def close(self) -> None: ...


class PredictionProvider(Protocol):
    def predict(self, history: Sequence[str]) -> AIPrediction: ...


class CounterPolicy(Protocol):
    def counter_for(self, predicted_gesture: str) -> str: ...


class SuitJawaEngine:
    """Manage ten rounds without owning any UI or drawing code.

    Defaults use a 3-second countdown, a 5-second gesture window, 0.35 seconds
    of stable input, a 1-second counter reveal, and a 1.5-second result interval.
    """

    def __init__(
        self,
        gesture_interface: GestureProvider | None = None,
        *,
        total_rounds: int = MAX_ROUNDS,
        countdown_seconds: float = 3.0,
        gesture_timeout: float = 5.0,
        stable_seconds: float = 0.35,
        counter_reveal_seconds: float = 1.0,
        result_seconds: float = 1.5,
        confidence_threshold: float | None = None,
        rng: random.Random | None = None,
        predictor: PredictionProvider | None = None,
        counter: CounterPolicy | None = None,
        logger: GameLogger | None = None,
    ) -> None:
        if total_rounds < 1:
            raise ValueError("total_rounds harus minimal 1.")
        for name, duration in (
            ("countdown_seconds", countdown_seconds),
            ("gesture_timeout", gesture_timeout),
            ("stable_seconds", stable_seconds),
            ("counter_reveal_seconds", counter_reveal_seconds),
            ("result_seconds", result_seconds),
        ):
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"{name} harus berupa angka finite dan tidak negatif.")

        source_threshold = (
            gesture_interface.confidence_threshold
            if gesture_interface is not None
            else DEFAULT_CONFIDENCE_THRESHOLD
        )
        minimum_confidence = (
            source_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        if (
            isinstance(minimum_confidence, bool)
            or not isinstance(minimum_confidence, (int, float))
            or not math.isfinite(minimum_confidence)
            or not 0 <= minimum_confidence <= 1
        ):
            raise ValueError("confidence_threshold harus berupa angka antara 0 dan 1.")

        self._gesture_interface = gesture_interface
        self.total_rounds = total_rounds
        self.countdown_seconds = countdown_seconds
        self.gesture_timeout = gesture_timeout
        self.stable_seconds = stable_seconds
        self.counter_reveal_seconds = counter_reveal_seconds
        self.result_seconds = result_seconds
        self.confidence_threshold = float(minimum_confidence)
        self._rng = rng if rng is not None else random.Random()
        self._predictor = (
            predictor if predictor is not None else AIPredictor(rng=self._rng)
        )
        self._counter = counter if counter is not None else AICounter()
        self._logger = logger if logger is not None else GameLogger()

        self.state = GameState.READY
        self.current_round = 0
        self.player_score = 0
        self.ai_score = 0
        self._results: list[RoundResult] = []
        self._deadline: float | None = None
        self._candidate_label: str | None = None
        self._candidate_since: float | None = None
        self._last_time: float | None = None
        self._forecast: AIPrediction | None = None
        self._planned_ai_gesture: str | None = None

    @property
    def results(self) -> tuple[RoundResult, ...]:
        return tuple(self._results)

    @property
    def prediction_statistics(self) -> PredictionStatistics:
        evaluated = tuple(
            result
            for result in self._results
            if result.player_gesture in GESTURES
            and result.predicted_player_gesture in GESTURES
        )
        correct = sum(
            result.predicted_player_gesture == result.player_gesture
            for result in evaluated
        )
        total = len(evaluated)
        return PredictionStatistics(
            correct_predictions=correct,
            evaluated_rounds=total,
            accuracy=correct / total if total else None,
        )

    @property
    def round_history(self) -> tuple[RoundResult, ...]:
        return self.results

    @property
    def last_result(self) -> RoundResult | None:
        return self._results[-1] if self._results else None

    @property
    def forecast(self) -> AIPrediction | None:
        return self._forecast

    @property
    def planned_ai_gesture(self) -> str | None:
        return self._planned_ai_gesture

    @property
    def session_id(self) -> str:
        return self._logger.session_id

    @property
    def message(self) -> str:
        if self.state is GameState.READY:
            return "Siap memulai"
        if self.state is GameState.COUNTDOWN:
            return f"Ronde {self.current_round}: bersiap"
        if self.state is GameState.COUNTER_REVEAL:
            if self._forecast is None or self._planned_ai_gesture is None:
                return "AI menyiapkan counter"
            return (
                f"Counter AI: {self._planned_ai_gesture} "
                f"(prediksi: {self._forecast.predicted_gesture})"
            )
        if self.state is GameState.COLLECTING_GESTURE:
            return "Suit!"
        if self.state is GameState.ROUND_RESULT and self.last_result is not None:
            return self.last_result.message
        return "Permainan selesai"

    def start(self, now: float | None = None) -> None:
        """Start or restart a game at round one."""
        if self.state in {
            GameState.COUNTDOWN,
            GameState.COUNTER_REVEAL,
            GameState.COLLECTING_GESTURE,
            GameState.ROUND_RESULT,
        }:
            raise RuntimeError("Permainan sedang berjalan.")
        current_time = self._validate_time(time.monotonic() if now is None else now)
        self._logger.start_session()
        self.current_round = 1
        self.player_score = 0
        self.ai_score = 0
        self._results.clear()
        self._candidate_label = None
        self._candidate_since = None
        self._forecast = None
        self._planned_ai_gesture = None
        self._deadline = current_time + self.countdown_seconds
        self._last_time = current_time
        self.state = GameState.COUNTDOWN

    def update(
        self,
        prediction: GesturePrediction | None = None,
        *,
        now: float | None = None,
    ) -> GameState:
        """Advance the FSM, optionally polling its configured GestureInterface."""
        if self.state is GameState.READY:
            raise RuntimeError("Panggil start() sebelum update().")

        current_time = self._read_time(now)
        if self.state is GameState.FINISHED:
            return self.state

        if self.state is GameState.COUNTDOWN:
            if current_time < self._require_deadline():
                return self.state
            self._prepare_round()
            self.state = GameState.COUNTER_REVEAL
            self._deadline = current_time + self.counter_reveal_seconds
            self._reset_candidate()
            return self.state

        if self.state is GameState.COUNTER_REVEAL:
            if current_time < self._require_deadline():
                return self.state
            self.state = GameState.COLLECTING_GESTURE
            self._deadline = current_time + self.gesture_timeout
            self._reset_candidate()
            return self.state

        if self.state is GameState.COLLECTING_GESTURE:
            if current_time >= self._require_deadline():
                self._finish_unstable_round()
                return self.state
            if prediction is None and self._gesture_interface is not None:
                prediction = self._gesture_interface.get_prediction()
            self._accept_prediction(prediction, current_time)
            return self.state

        if self.state is GameState.ROUND_RESULT:
            if current_time < self._require_deadline():
                return self.state
            if self.current_round >= self.total_rounds:
                self.state = GameState.FINISHED
                self._deadline = None
                return self.state
            self.current_round += 1
            self.state = GameState.COUNTDOWN
            self._deadline = current_time + self.countdown_seconds
            self._reset_candidate()
            self._forecast = None
            self._planned_ai_gesture = None
        return self.state

    def close(self) -> None:
        """Close the GestureInterface when one was supplied."""
        if self._gesture_interface is not None:
            self._gesture_interface.close()

    @staticmethod
    def determine_winner(player_gesture: str, ai_gesture: str) -> Winner:
        """Apply Suit Jawa rules: Gajah > Orang > Semut > Gajah."""
        if player_gesture not in GESTURES or ai_gesture not in GESTURES:
            raise ValueError("Gestur harus Gajah, Orang, atau Semut.")
        if player_gesture == ai_gesture:
            return "draw"
        if BEATS[player_gesture] == ai_gesture:
            return "player"
        return "ai"

    def _accept_prediction(
        self,
        prediction: GesturePrediction | None,
        current_time: float,
    ) -> None:
        if (
            prediction is None
            or not prediction.hand_detected
            or not prediction.is_valid
            or prediction.label not in GESTURES
            or not math.isfinite(prediction.confidence)
            or not self.confidence_threshold <= prediction.confidence <= 1
        ):
            self._reset_candidate()
            return

        if prediction.label != self._candidate_label:
            self._candidate_label = prediction.label
            self._candidate_since = current_time
            if self.stable_seconds == 0:
                self._finish_valid_round(prediction.label)
            return

        if (
            self._candidate_since is not None
            and current_time - self._candidate_since >= self.stable_seconds
        ):
            self._finish_valid_round(prediction.label)

    def _finish_valid_round(self, player_gesture: str) -> None:
        if self._planned_ai_gesture is None:
            raise RuntimeError("Counter AI harus ditentukan sebelum input ronde.")
        ai_gesture = self._planned_ai_gesture
        winner = self.determine_winner(player_gesture, ai_gesture)
        if winner == "player":
            self.player_score += 1
            message = "Pemain menang"
        elif winner == "ai":
            self.ai_score += 1
            message = "AI menang"
        else:
            message = "Seri"
        self._record_result(player_gesture, ai_gesture, winner, message)

    def _prepare_round(self) -> None:
        player_history = tuple(
            result.player_gesture
            for result in self._results
            if result.player_gesture in GESTURES
        )
        self._forecast = self._predictor.predict(player_history)
        self._planned_ai_gesture = self._counter.counter_for(
            self._forecast.predicted_gesture
        )

    def _finish_unstable_round(self) -> None:
        self._reset_candidate()
        self._record_result(None, None, "none", "Gestur tidak stabil")

    def _record_result(
        self,
        player_gesture: str | None,
        ai_gesture: str | None,
        winner: Winner,
        message: str,
    ) -> None:
        result = RoundResult(
            round_number=self.current_round,
            player_gesture=player_gesture,
            ai_gesture=ai_gesture,
            winner=winner,
            message=message,
            player_score=self.player_score,
            ai_score=self.ai_score,
            predicted_player_gesture=(
                self._forecast.predicted_gesture
                if self._forecast is not None
                else None
            ),
        )
        self._results.append(result)
        self.state = GameState.ROUND_RESULT
        self._deadline = (
            (self._last_time if self._last_time is not None else time.monotonic())
            + self.result_seconds
        )
        self._logger.log_round(
            round_number=result.round_number,
            player_gesture=result.player_gesture,
            ai_prediction=result.predicted_player_gesture,
            ai_move=self._planned_ai_gesture,
            winner=result.winner,
            player_score=result.player_score,
            ai_score=result.ai_score,
        )

    def _reset_candidate(self) -> None:
        self._candidate_label = None
        self._candidate_since = None

    def _require_deadline(self) -> float:
        if self._deadline is None:
            raise RuntimeError("State machine tidak memiliki deadline aktif.")
        return self._deadline

    def _read_time(self, now: float | None) -> float:
        current_time = self._validate_time(time.monotonic() if now is None else now)
        if self._last_time is not None and current_time < self._last_time:
            raise ValueError("Waktu update tidak boleh bergerak mundur.")
        self._last_time = current_time
        return current_time

    @staticmethod
    def _validate_time(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Waktu harus berupa angka finite.")
        return value


__all__ = [
    "SuitJawaEngine",
    "GameState",
    "RoundResult",
    "PredictionStatistics",
]
