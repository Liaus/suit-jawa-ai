"""Collect consented, pseudonymous landmark samples from a webcam."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

from src.features.landmark_normalizer import FEATURE_COLUMNS, normalize_hand_landmarks
from src.utils.csv_validator import (
    DATASET_COLUMNS,
    METADATA_COLUMNS,
    ensure_csv_header,
    ensure_dataset_csv,
    validate_csv_round_trip,
    validate_dataset_row,
    validate_metadata_row,
    validate_participant_id,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dataset_suit_jawa.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "collection_metadata.csv"
CLASS_KEYS = {"1": "Gajah", "2": "Orang", "3": "Semut", "4": "unknown"}
DISTANCE_KEYS = {"j": "near", "k": "medium", "l": "far"}
LIGHTING_KEYS = {"a": "low", "s": "normal", "d": "bright"}
SAVE_COOLDOWN_SECONDS = 0.25


def _load_vision_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            f"OpenCV belum tersedia untuk Python yang sedang dipakai "
            f"({sys.executable}). Pasang dependensi dengan "
            "`.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from error
    try:
        import mediapipe as mp
    except ImportError as error:
        raise RuntimeError(
            f"MediaPipe belum tersedia untuk Python yang sedang dipakai "
            f"({sys.executable}). Pasang dependensi dengan "
            "`.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt`."
        ) from error
    return cv2, mp


def _count_dataset_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


def _detected_hand_side(result: Any) -> str:
    if not result.multi_handedness:
        return "unknown"
    label = result.multi_handedness[0].classification[0].label
    return label if label in {"Left", "Right"} else "unknown"


def record_dataset(
    output_path: Path,
    metadata_path: Path,
    participant_id: str,
    camera_index: int = 0,
    consent_confirmed: bool = False,
) -> int:
    """Record samples for one consenting participant until the user quits."""
    if not consent_confirmed:
        raise ValueError("Persetujuan sukarelawan harus dikonfirmasi sebelum merekam.")
    validate_participant_id(participant_id)

    cv2, mp = _load_vision_dependencies()
    ensure_dataset_csv(output_path)
    ensure_csv_header(metadata_path, METADATA_COLUMNS)
    validate_csv_round_trip(
        DATASET_COLUMNS,
        {
            "participant_id": participant_id,
            "label": "Gajah",
            **{name: 0.0 for name in FEATURE_COLUMNS},
        },
    )
    validate_csv_round_trip(
        METADATA_COLUMNS,
        {
            "dataset_row": 1,
            "participant_id": participant_id,
            "label": "Gajah",
            "hand": "Left",
            "distance": "near",
            "lighting": "normal",
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Webcam dengan indeks {camera_index} tidak dapat dibuka. "
            "Periksa izin kamera, indeks perangkat, dan apakah kamera sedang "
            "digunakan aplikasi lain."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands_api = mp.solutions.hands
    selected_label: str | None = None
    selected_distance: str | None = None
    selected_lighting: str | None = None
    status = "Pilih kelas, jarak, dan pencahayaan"
    saved_count = 0
    next_dataset_row = _count_dataset_rows(output_path) + 1
    last_saved_at = 0.0
    try:
        with output_path.open("a", newline="", encoding="utf-8") as dataset_file:
            with metadata_path.open("a", newline="", encoding="utf-8") as metadata_file:
                dataset_writer = csv.DictWriter(dataset_file, fieldnames=DATASET_COLUMNS)
                metadata_writer = csv.writer(metadata_file)
                with hands_api.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                ) as hands:
                    while True:
                        success, frame = capture.read()
                        if not success:
                            raise RuntimeError("Gagal membaca frame dari webcam.")

                        frame = cv2.flip(frame, 1)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = hands.process(rgb_frame)
                        landmarks = (
                            result.multi_hand_landmarks[0]
                            if result.multi_hand_landmarks
                            else None
                        )
                        hand_side = _detected_hand_side(result) if landmarks else "unknown"

                        if landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                landmarks,
                                hands_api.HAND_CONNECTIONS,
                            )

                        hud_lines = (
                            f"Peserta: {participant_id} | Kelas: {selected_label or '-'} "
                            f"| Sampel sesi: {saved_count}",
                            f"Tangan: {hand_side} | Jarak: {selected_distance or '-'} "
                            f"| Cahaya: {selected_lighting or '-'}",
                            "Kelas 1/2/3/4=gestur/unknown | J/K/L=jarak dekat/sedang/jauh",
                            "A/S/D=cahaya redup/normal/terang | SPACE=simpan | Q=keluar",
                            status,
                        )
                        for line_number, text in enumerate(hud_lines):
                            cv2.putText(
                                frame,
                                text,
                                (10, 25 + line_number * 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.48,
                                (0, 255, 0) if line_number < 2 else (0, 255, 255),
                                1,
                            )
                        cv2.imshow("Suit Jawa vs AI - Perekam Dataset", frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            break
                        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                            selected_label = CLASS_KEYS[chr(key)]
                            status = f"Kelas {selected_label} dipilih"
                        elif key in (ord("j"), ord("k"), ord("l")):
                            selected_distance = DISTANCE_KEYS[chr(key)]
                            status = f"Jarak {selected_distance} dipilih"
                        elif key in (ord("a"), ord("s"), ord("d")):
                            selected_lighting = LIGHTING_KEYS[chr(key)]
                            status = f"Pencahayaan {selected_lighting} dipilih"
                        elif key == ord(" "):
                            now = time.monotonic()
                            if now - last_saved_at < SAVE_COOLDOWN_SECONDS:
                                continue
                            last_saved_at = now
                            if selected_label is None:
                                status = "Pilih kelas 1, 2, 3, atau 4 terlebih dahulu"
                                continue
                            if selected_distance is None or selected_lighting is None:
                                status = "Pilih jarak (J/K/L) dan cahaya (A/S/D)"
                                continue
                            if landmarks is None:
                                status = "Tangan belum terdeteksi; sampel tidak disimpan"
                                continue

                            features = normalize_hand_landmarks(landmarks)
                            dataset_row = {
                                "participant_id": participant_id,
                                "label": selected_label,
                                **dict(zip(FEATURE_COLUMNS, features)),
                            }
                            metadata_row = {
                                "dataset_row": next_dataset_row,
                                "participant_id": participant_id,
                                "label": selected_label,
                                "hand": hand_side,
                                "distance": selected_distance,
                                "lighting": selected_lighting,
                                "recorded_at": datetime.now(timezone.utc).isoformat(
                                    timespec="seconds"
                                ),
                            }
                            validate_dataset_row(dataset_row)
                            validate_metadata_row(
                                {key: str(value) for key, value in metadata_row.items()}
                            )

                            dataset_position = dataset_file.tell()
                            metadata_position = metadata_file.tell()
                            try:
                                dataset_writer.writerow(dataset_row)
                                metadata_writer.writerow(
                                    [metadata_row[column] for column in METADATA_COLUMNS]
                                )
                                dataset_file.flush()
                                metadata_file.flush()
                            except Exception:
                                dataset_file.seek(dataset_position)
                                dataset_file.truncate()
                                metadata_file.seek(metadata_position)
                                metadata_file.truncate()
                                raise

                            next_dataset_row += 1
                            saved_count += 1
                            status = f"Sampel {selected_label} dan metadata tersimpan"
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return saved_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--participant-id",
        required=True,
        help="ID anonim sukarelawan, format P001 sampai P999 (jangan gunakan nama)",
    )
    parser.add_argument(
        "--consent-confirmed",
        action="store_true",
        help="Konfirmasi bahwa sukarelawan telah menyetujui perekaman landmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Lokasi file CSV fitur (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Lokasi file metadata (default: {DEFAULT_METADATA})",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indeks webcam (default: 0)",
    )
    args = parser.parse_args()
    if not args.consent_confirmed:
        parser.error("Gunakan --consent-confirmed setelah mendapat persetujuan peserta.")
    try:
        validate_participant_id(args.participant_id)
        saved_count = record_dataset(
            output_path=args.output,
            metadata_path=args.metadata,
            participant_id=args.participant_id,
            camera_index=args.camera_index,
            consent_confirmed=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Perekaman dataset gagal: {error}")
        return 1

    print(
        f"Perekaman selesai. {saved_count} sampel ditambahkan untuk "
        f"{args.participant_id} ke {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
