"""CSV schema and round-trip helpers for collected landmark datasets."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping, Sequence

from src.features.landmark_normalizer import FEATURE_COLUMNS


VALID_LABELS = frozenset({"Gajah", "Orang", "Semut", "unknown"})
CORE_LABELS = frozenset({"Gajah", "Orang", "Semut"})
LEGACY_PARTICIPANT_ID = "LEGACY_UNVERIFIED"
PARTICIPANT_ID_PATTERN = re.compile(r"P(?!000)[0-9]{3}\Z")
DATASET_COLUMNS = ("participant_id", "label", *FEATURE_COLUMNS)
LEGACY_DATASET_COLUMNS = ("label",) + tuple(
    f"landmark_{index:02d}_{axis}"
    for index in range(21)
    for axis in ("x", "y", "z")
)
METADATA_COLUMNS = (
    "dataset_row",
    "participant_id",
    "label",
    "hand",
    "distance",
    "lighting",
    "recorded_at",
)
VALID_HANDS = frozenset({"Left", "Right", "unknown"})
VALID_DISTANCES = frozenset({"near", "medium", "far"})
VALID_LIGHTING = frozenset({"low", "normal", "bright"})


def ensure_csv_header(csv_path: Path, fieldnames: Sequence[str]) -> None:
    """Create a dataset CSV with a header, or verify its existing schema."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            existing_header = next(csv.reader(file), [])
        if existing_header != list(fieldnames):
            raise ValueError(
                f"Header CSV {csv_path} tidak sesuai dengan fitur yang diharapkan."
            )
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(fieldnames)


def validate_participant_id(participant_id: str) -> None:
    """Require a pseudonymous ID such as P001, not a participant's name."""
    if not isinstance(participant_id, str) or not PARTICIPANT_ID_PATTERN.fullmatch(
        participant_id
    ):
        raise ValueError("ID peserta harus anonim dengan format P001 sampai P999.")


def validate_dataset_row(row: Mapping[str, str]) -> None:
    """Validate a dataset row's participant, class, and 63 finite features."""
    participant_id = row.get("participant_id", "")
    if (
        participant_id != LEGACY_PARTICIPANT_ID
        and (
            not isinstance(participant_id, str)
            or not PARTICIPANT_ID_PATTERN.fullmatch(participant_id)
        )
    ):
        raise ValueError(f"ID peserta tidak valid: {participant_id!r}.")
    if row.get("label") not in VALID_LABELS:
        raise ValueError(f"Label dataset tidak valid: {row.get('label')!r}.")
    for column in FEATURE_COLUMNS:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Fitur {column} harus berupa angka.") from error
        if not math.isfinite(value):
            raise ValueError(f"Fitur {column} harus bernilai finite.")


def validate_metadata_row(row: Mapping[str, str]) -> None:
    """Validate one condition-log row."""
    try:
        dataset_row = int(row["dataset_row"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dataset_row harus berupa integer positif.") from error
    if dataset_row < 1:
        raise ValueError("dataset_row harus berupa integer positif.")
    validate_participant_id(row.get("participant_id", ""))
    if row.get("label") not in VALID_LABELS:
        raise ValueError(f"Label metadata tidak valid: {row.get('label')!r}.")
    if row.get("hand") not in VALID_HANDS:
        raise ValueError(f"Kondisi tangan tidak valid: {row.get('hand')!r}.")
    if row.get("distance") not in VALID_DISTANCES:
        raise ValueError(f"Jarak tidak valid: {row.get('distance')!r}.")
    if row.get("lighting") not in VALID_LIGHTING:
        raise ValueError(f"Pencahayaan tidak valid: {row.get('lighting')!r}.")
    if not row.get("recorded_at"):
        raise ValueError("recorded_at tidak boleh kosong.")


def ensure_dataset_csv(csv_path: Path) -> int:
    """Create the 65-column dataset or migrate the known 1.2 schema safely.

    Returns the number of legacy rows migrated. Since their participants and
    collection conditions were not recorded, they receive an explicit
    sentinel ID and are excluded from verified-volunteer audit checks.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or not csv_path.stat().st_size:
        ensure_csv_header(csv_path, DATASET_COLUMNS)
        return 0

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        header = reader.fieldnames or []
        if header == list(DATASET_COLUMNS):
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"Baris dataset {row_number} memiliki jumlah kolom yang salah."
                    )
                validate_dataset_row(row)
            return 0
        if header != list(LEGACY_DATASET_COLUMNS):
            raise ValueError(
                f"Header CSV {csv_path} tidak dikenal; file tidak diubah."
            )
        legacy_rows = list(reader)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{csv_path.stem}_",
        suffix=".tmp",
        dir=csv_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=DATASET_COLUMNS)
            writer.writeheader()
            for row_number, legacy_row in enumerate(legacy_rows, start=2):
                if None in legacy_row or any(value is None for value in legacy_row.values()):
                    raise ValueError(
                        f"Baris legacy {row_number} memiliki jumlah kolom yang salah."
                    )
                migrated_row = {
                    "participant_id": LEGACY_PARTICIPANT_ID,
                    "label": legacy_row["label"],
                    **dict(
                        zip(
                            FEATURE_COLUMNS,
                            (
                                legacy_row[column]
                                for column in LEGACY_DATASET_COLUMNS[1:]
                            ),
                        )
                    ),
                }
                validate_dataset_row(migrated_row)
                writer.writerow(migrated_row)
        os.replace(temporary_path, csv_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return len(legacy_rows)


def validate_csv_round_trip(
    fieldnames: Sequence[str],
    row: Mapping[str, str | int | float],
) -> None:
    """Write a sample to a temporary CSV and verify its header and values."""
    if len(fieldnames) != len(row) or set(fieldnames) != set(row):
        raise ValueError("Kolom sampel harus sama persis dengan header CSV.")

    expected = {name: str(row[name]) for name in fieldnames}
    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "csv_round_trip_test.csv"
        with test_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

        with test_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != list(fieldnames):
                raise ValueError("Validasi CSV gagal: header berubah setelah ditulis.")
            actual = next(reader, None)

        if actual != expected:
            raise ValueError("Validasi CSV gagal: nilai berubah setelah dibaca ulang.")
