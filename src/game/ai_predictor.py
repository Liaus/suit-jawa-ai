"""First-order Markov and frequency-based player gesture prediction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Iterable, Literal


GESTURES = ("Gajah", "Orang", "Semut")
PredictionStrategy = Literal["prior", "frequency", "markov+frequency"]


@dataclass(frozen=True)
class AIPrediction:
    predicted_gesture: str
    probabilities: dict[str, float]
    confidence: float
    strategy: PredictionStrategy


class AIPredictor:
    """Blend empirical move frequencies with order-one transition patterns."""

    def __init__(
        self,
        *,
        smoothing: float = 1.0,
        markov_weight: float = 0.7,
        rng: random.Random | None = None,
    ) -> None:
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing harus finite dan lebih besar dari 0.")
        if not math.isfinite(markov_weight) or not 0 <= markov_weight <= 1:
            raise ValueError("markov_weight harus berada di antara 0 dan 1.")
        self.smoothing = smoothing
        self.markov_weight = markov_weight
        self._rng = rng if rng is not None else random.Random()

    def predict(self, history: Iterable[str]) -> AIPrediction:
        """Predict the next move from completed, valid player gestures only."""
        moves = tuple(history)
        invalid_moves = set(moves).difference(GESTURES)
        if invalid_moves:
            raise ValueError(
                "Riwayat hanya boleh berisi Gajah, Orang, atau Semut: "
                + ", ".join(sorted(invalid_moves))
            )

        frequencies = Counter(moves)
        frequency_distribution = self._distribution(frequencies, len(moves))
        strategy: PredictionStrategy
        if not moves:
            probabilities = frequency_distribution
            strategy = "prior"
        else:
            transitions = Counter(
                next_move
                for previous_move, next_move in zip(moves, moves[1:])
                if previous_move == moves[-1]
            )
            transition_count = sum(transitions.values())
            if transition_count:
                markov_distribution = self._distribution(
                    transitions,
                    transition_count,
                )
                probabilities = {
                    gesture: (
                        self.markov_weight * markov_distribution[gesture]
                        + (1 - self.markov_weight)
                        * frequency_distribution[gesture]
                    )
                    for gesture in GESTURES
                }
                strategy = "markov+frequency"
            else:
                probabilities = frequency_distribution
                strategy = "frequency"

        highest_probability = max(probabilities.values())
        tied_gestures = [
            gesture
            for gesture in GESTURES
            if math.isclose(
                probabilities[gesture],
                highest_probability,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        predicted_gesture = self._rng.choice(tied_gestures)
        return AIPrediction(
            predicted_gesture=predicted_gesture,
            probabilities=probabilities,
            confidence=probabilities[predicted_gesture],
            strategy=strategy,
        )

    def _distribution(
        self,
        counts: Counter[str],
        observation_count: int,
    ) -> dict[str, float]:
        denominator = observation_count + self.smoothing * len(GESTURES)
        return {
            gesture: (
                counts[gesture] + self.smoothing
            ) / denominator
            for gesture in GESTURES
        }


__all__ = ["AIPredictor", "AIPrediction"]
