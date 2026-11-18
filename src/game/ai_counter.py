"""Choose an AI gesture that beats the predicted player gesture."""

from __future__ import annotations


COUNTER_MOVES = {"Gajah": "Semut", "Orang": "Gajah", "Semut": "Orang"}


class AICounter:
    """Map a predicted player move to its Suit Jawa winning counter."""

    def counter_for(self, predicted_gesture: str) -> str:
        try:
            return COUNTER_MOVES[predicted_gesture]
        except KeyError as error:
            raise ValueError(
                "Prediksi pemain harus Gajah, Orang, atau Semut."
            ) from error


__all__ = ["AICounter"]
