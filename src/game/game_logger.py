"""Append-only, PII-free CSV logging for Suit Jawa game rounds."""

from __future__ import annotations

import csv
from pathlib import Path
import uuid


CSV_COLUMNS = (
    "session_id",
    "round_number",
    "player_gesture",
    "ai_prediction",
    "ai_move",
    "winner",
    "player_score",
    "ai_score",
)
GESTURES = frozenset({"Gajah", "Orang", "Semut"})
WINNERS = frozenset({"player", "ai", "draw", "none"})
DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "game_logs.csv"


def generate_session_id() -> str:
    """Return a random UUID4 that is not derived from user or device data."""
    return str(uuid.uuid4())


class GameLogger:
    """Append validated round summaries without storing personal information."""

    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self._session_id = generate_session_id()
        self._logged_rounds: set[int] = set()

    @property
    def session_id(self) -> str:
        return self._session_id

    def start_session(self) -> str:
        """Create a fresh anonymous ID for a new game in the same process."""
        self._session_id = generate_session_id()
        self._logged_rounds.clear()
        return self._session_id

    def log_round(
        self,
        *,
        round_number: int,
        player_gesture: str | None,
        ai_prediction: str | None,
        ai_move: str | None,
        winner: str,
        player_score: int,
        ai_score: int,
    ) -> None:
        """Append one completed round, rejecting duplicate or non-game data."""
        self._validate_round_number(round_number)
        self._validate_score("player_score", player_score)
        self._validate_score("ai_score", ai_score)
        self._validate_gesture("player_gesture", player_gesture)
        self._validate_gesture("ai_prediction", ai_prediction)
        self._validate_gesture("ai_move", ai_move)
        if not isinstance(winner, str) or winner not in WINNERS:
            raise ValueError("winner harus player, ai, draw, atau none.")
        if round_number in self._logged_rounds:
            raise ValueError(
                f"Ronde {round_number} sudah dicatat untuk sesi ini."
            )

        row = (
            self._session_id,
            round_number,
            player_gesture or "",
            ai_prediction or "",
            ai_move or "",
            winner,
            player_score,
            ai_score,
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a+", newline="", encoding="utf-8") as file:
            file.seek(0)
            reader = csv.reader(file)
            existing_header = next(reader, None)
            if existing_header is None:
                writer = csv.writer(file)
                writer.writerow(CSV_COLUMNS)
            elif tuple(existing_header) != CSV_COLUMNS:
                raise ValueError(
                    f"Header CSV tidak sesuai skema game log: {self.log_path}"
                )

            file.seek(0, 2)
            writer = csv.writer(file)
            writer.writerow(row)
            file.flush()
        self._logged_rounds.add(round_number)

    @staticmethod
    def _validate_round_number(round_number: int) -> None:
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or round_number < 1
        ):
            raise ValueError("round_number harus berupa integer positif.")

    @staticmethod
    def _validate_score(name: str, score: int) -> None:
        if isinstance(score, bool) or not isinstance(score, int) or score < 0:
            raise ValueError(f"{name} harus berupa integer non-negatif.")

    @staticmethod
    def _validate_gesture(name: str, gesture: str | None) -> None:
        if gesture is not None and (
            not isinstance(gesture, str) or gesture not in GESTURES
        ):
            raise ValueError(
                f"{name} harus Gajah, Orang, Semut, atau None."
            )


__all__ = ["GameLogger", "CSV_COLUMNS", "generate_session_id"]
