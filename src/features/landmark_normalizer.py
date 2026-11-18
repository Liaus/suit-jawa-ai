"""Extract translation- and scale-normalized MediaPipe hand landmarks."""

from __future__ import annotations

import math
from typing import Sequence


LANDMARK_COUNT = 21
FEATURE_COLUMNS = tuple(
    f"{axis}{index}"
    for index in range(LANDMARK_COUNT)
    for axis in ("x", "y", "z")
)


def normalize_hand_landmarks(landmarks: object) -> list[float]:
    """Return 63 wrist-centered, scale-normalized x/y/z landmark features.

    Accepts either a sequence of landmark objects or a MediaPipe landmark
    collection exposing a ``landmark`` sequence.
    """
    points = getattr(landmarks, "landmark", landmarks)
    if not isinstance(points, Sequence) or len(points) != LANDMARK_COUNT:
        raise ValueError(f"Tepat {LANDMARK_COUNT} landmark tangan diperlukan.")

    coordinates: list[tuple[float, float, float]] = []
    for index, point in enumerate(points):
        try:
            coordinate = (float(point.x), float(point.y), float(point.z))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(
                f"Landmark ke-{index} harus memiliki koordinat x, y, dan z numerik."
            ) from error
        if not all(math.isfinite(value) for value in coordinate):
            raise ValueError(f"Koordinat landmark ke-{index} harus bernilai finite.")
        coordinates.append(coordinate)

    wrist = coordinates[0]
    centered = [
        tuple(value - origin for value, origin in zip(point, wrist))
        for point in coordinates
    ]
    scale = max(abs(value) for point in centered for value in point)
    if scale == 0:
        return [0.0] * len(FEATURE_COLUMNS)

    return [
        value / scale
        for point in centered
        for value in point
    ]
