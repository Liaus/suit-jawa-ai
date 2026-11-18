"""Audit the Suit Jawa landmark dataset and write a collection report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

from src.features.landmark_normalizer import FEATURE_COLUMNS
from src.utils.csv_validator import (
    CORE_LABELS,
    DATASET_COLUMNS,
    LEGACY_PARTICIPANT_ID,
    METADATA_COLUMNS,
    VALID_DISTANCES,
    VALID_HANDS,
    VALID_LABELS,
    VALID_LIGHTING,
    validate_metadata_row,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "dataset_suit_jawa.csv"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "collection_metadata.csv"
DEFAULT_REPORT = PROJECT_ROOT / "dataset_report.md"
PARTICIPANT_ID_PATTERN = re.compile(r"P(?!000)[0-9]{3}\Z")
REQUIRED_PARTICIPANTS = 5


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _row_has_exact_columns(row: dict[str, str], columns: tuple[str, ...]) -> bool:
    return (
        None not in row
        and set(row) == set(columns)
        and all(value is not None for value in row.values())
    )


def _participant_id(value: str | None) -> bool:
    return bool(value and PARTICIPANT_ID_PATTERN.fullmatch(value))


def _feature_values_valid(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        for column in FEATURE_COLUMNS:
            try:
                if not math.isfinite(float(row[column])):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    return bool(rows)


def audit_dataset(
    dataset_path: Path,
    metadata_path: Path,
    report_path: Path,
    balance_ratio_limit: float = 1.5,
) -> dict[str, Any]:
    """Run 11 documented collection checks and persist a Markdown report."""
    dataset_header, dataset_rows = _read_csv(dataset_path)
    metadata_header, metadata_rows = _read_csv(metadata_path)

    exact_dataset_rows = all(
        _row_has_exact_columns(row, DATASET_COLUMNS) for row in dataset_rows
    )
    valid_row_ids = all(
        row.get("participant_id") == LEGACY_PARTICIPANT_ID
        or _participant_id(row.get("participant_id"))
        for row in dataset_rows
    )
    labels = Counter(row.get("label", "") for row in dataset_rows)
    known_labels = all(label in VALID_LABELS for label in labels)
    all_labels_present = all(labels[label] > 0 for label in VALID_LABELS)
    feature_values_valid = (
        exact_dataset_rows and _feature_values_valid(dataset_rows)
    )

    verified_rows = [
        (index, row)
        for index, row in enumerate(dataset_rows, start=1)
        if _participant_id(row.get("participant_id"))
    ]
    verified_participants = {
        row["participant_id"] for _, row in verified_rows
    }
    legacy_count = sum(
        row.get("participant_id") == LEGACY_PARTICIPANT_ID for row in dataset_rows
    )

    participant_labels: dict[str, set[str]] = defaultdict(set)
    for _, row in verified_rows:
        participant_labels[row["participant_id"]].add(row.get("label", ""))
    participants_cover_core_labels = bool(verified_participants) and all(
        CORE_LABELS.issubset(participant_labels[participant_id])
        for participant_id in verified_participants
    )

    metadata_by_dataset_row: dict[int, dict[str, str]] = {}
    metadata_valid = metadata_header == list(METADATA_COLUMNS)
    for row in metadata_rows:
        if not _row_has_exact_columns(row, METADATA_COLUMNS):
            metadata_valid = False
            continue
        try:
            validate_metadata_row(row)
            row_number = int(row["dataset_row"])
        except ValueError:
            metadata_valid = False
            continue
        if row_number in metadata_by_dataset_row:
            metadata_valid = False
        else:
            metadata_by_dataset_row[row_number] = row

    verified_by_index = {index: row for index, row in verified_rows}
    metadata_coverage = (
        metadata_valid
        and bool(verified_rows)
        and set(metadata_by_dataset_row) == set(verified_by_index)
        and all(
            metadata_by_dataset_row[index]["participant_id"] == row["participant_id"]
            and metadata_by_dataset_row[index]["label"] == row["label"]
            for index, row in verified_by_index.items()
        )
    )
    valid_metadata = []
    for row_number, row in metadata_by_dataset_row.items():
        dataset_row = verified_by_index.get(row_number)
        if (
            dataset_row
            and row["participant_id"] == dataset_row["participant_id"]
            and row["label"] == dataset_row["label"]
            and row.get("hand") in VALID_HANDS
            and row.get("distance") in VALID_DISTANCES
            and row.get("lighting") in VALID_LIGHTING
        ):
            valid_metadata.append(row)
    hand_coverage = {"Left", "Right"}.issubset(
        {row["hand"] for row in valid_metadata}
    )
    distance_coverage = VALID_DISTANCES.issubset(
        {row["distance"] for row in valid_metadata}
    )
    lighting_coverage = VALID_LIGHTING.issubset(
        {row["lighting"] for row in valid_metadata}
    )

    verified_class_counts = Counter(row.get("label", "") for _, row in verified_rows)
    core_counts = [verified_class_counts[label] for label in sorted(CORE_LABELS)]
    if core_counts and min(core_counts) > 0:
        observed_balance_ratio = max(core_counts) / min(core_counts)
        class_balance = observed_balance_ratio <= balance_ratio_limit
    else:
        observed_balance_ratio = math.inf
        class_balance = False

    checks = [
        (
            "1. Skema CSV dataset tepat 65 kolom",
            dataset_header == list(DATASET_COLUMNS),
            f"{len(dataset_header)} kolom ditemukan; {len(DATASET_COLUMNS)} diwajibkan.",
        ),
        (
            "2. Dataset berisi baris dengan struktur lengkap",
            bool(dataset_rows) and exact_dataset_rows and valid_row_ids,
            f"{len(dataset_rows)} baris; "
            f"{sum(not _row_has_exact_columns(row, DATASET_COLUMNS) for row in dataset_rows)} "
            f"baris memiliki kolom kosong/berlebih; "
            f"{sum(not (row.get('participant_id') == LEGACY_PARTICIPANT_ID or _participant_id(row.get('participant_id'))) for row in dataset_rows)} "
            "ID peserta tidak valid.",
        ),
        (
            "3. Empat label valid (Gajah, Orang, Semut, unknown) terwakili",
            known_labels and all_labels_present,
            ", ".join(f"{label}: {labels[label]}" for label in sorted(VALID_LABELS)),
        ),
        (
            "4. Semua 63 fitur numerik dan finite",
            feature_values_valid,
            "Seluruh nilai fitur valid." if feature_values_valid else "Ada fitur kosong, non-numerik, atau non-finite.",
        ),
        (
            f"5. Minimal {REQUIRED_PARTICIPANTS} sukarelawan anonim terverifikasi",
            len(verified_participants) >= REQUIRED_PARTICIPANTS,
            f"{len(verified_participants)} ID unik memenuhi format P001-P999; "
            f"{legacy_count} baris legacy dikecualikan.",
        ),
        (
            "6. Setiap sukarelawan memiliki tiga gestur utama",
            participants_cover_core_labels,
            f"{sum(1 for labels_for_person in participant_labels.values() if CORE_LABELS.issubset(labels_for_person))}"
            f"/{len(verified_participants)} peserta memiliki Gajah, Orang, dan Semut.",
        ),
        (
            "7. Metadata lengkap dan cocok untuk setiap sampel terverifikasi",
            metadata_coverage,
            f"{len(metadata_by_dataset_row)} metadata untuk {len(verified_rows)} "
            "sampel terverifikasi; data legacy tidak dihitung.",
        ),
        (
            "8. Variasi tangan kiri dan kanan tercatat",
            hand_coverage,
            ", ".join(
                f"{hand}: {sum(row['hand'] == hand for row in valid_metadata)}"
                for hand in ("Left", "Right")
            ),
        ),
        (
            "9. Variasi jarak dekat, sedang, dan jauh tercatat",
            distance_coverage,
            ", ".join(
                f"{distance}: {sum(row['distance'] == distance for row in valid_metadata)}"
                for distance in sorted(VALID_DISTANCES)
            ),
        ),
        (
            "10. Variasi pencahayaan redup, normal, dan terang tercatat",
            lighting_coverage,
            ", ".join(
                f"{lighting}: {sum(row['lighting'] == lighting for row in valid_metadata)}"
                for lighting in sorted(VALID_LIGHTING)
            ),
        ),
        (
            f"11. Keseimbangan tiga kelas utama (rasio maksimum {balance_ratio_limit:.2f}:1)",
            class_balance,
            (
                f"Rasio {observed_balance_ratio:.2f}:1; "
                if math.isfinite(observed_balance_ratio)
                else "Rasio tidak dapat dihitung; "
            )
            + ", ".join(
                f"{label}: {verified_class_counts[label]}"
                for label in sorted(CORE_LABELS)
            ),
        ),
    ]

    feature_signatures = [
        tuple(row[column] for column in FEATURE_COLUMNS)
        for _, row in verified_rows
        if all(row.get(column) is not None for column in FEATURE_COLUMNS)
    ]
    duplicate_count = len(feature_signatures) - len(set(feature_signatures))
    passed_count = sum(passed for _, passed, _ in checks)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report_lines = [
        "# Laporan Audit Dataset Suit Jawa vs AI",
        "",
        f"- Dibuat: {generated_at}",
        f"- Dataset: `{dataset_path}`",
        f"- Metadata: `{metadata_path}`",
        f"- Status keseluruhan: **{'LULUS' if passed_count == len(checks) else 'BELUM LULUS'}** "
        f"({passed_count}/{len(checks)} indikator)",
        "",
        "Audit hanya menghitung ID sukarelawan anonim berbentuk `P001`-`P999`. "
        f"{legacy_count} sampel lama ditandai `{LEGACY_PARTICIPANT_ID}` karena "
        "identitas dan kondisi pengumpulannya tidak tersedia; sampel tersebut "
        "dipertahankan, tetapi tidak dihitung sebagai sukarelawan terverifikasi.",
        "",
        "Keseimbangan dihitung hanya untuk Gajah/Orang/Semut; `unknown` adalah "
        "kelas tambahan untuk gestur di luar tiga kelas permainan. Ambang "
        "keseimbangan default adalah rasio kelas terbesar terhadap terkecil "
        f"maksimal {balance_ratio_limit:.2f}:1.",
        "",
        "## 11 indikator",
        "",
        "| # | Indikator | Status | Hasil |",
        "| --- | --- | --- | --- |",
    ]
    for index, (name, passed, detail) in enumerate(checks, start=1):
        safe_detail = detail.replace("|", "\\|")
        report_lines.append(
            f"| {index} | {name} | {'LULUS' if passed else 'BELUM LULUS'} | {safe_detail} |"
        )
    report_lines.extend(
        [
            "",
            "## Distribusi label",
            "",
            "| Label | Semua baris | Peserta terverifikasi |",
            "| --- | ---: | ---: |",
        ]
    )
    for label in sorted(VALID_LABELS):
        report_lines.append(
            f"| {label} | {labels[label]} | {verified_class_counts[label]} |"
        )
    report_lines.extend(
        [
            "",
            "## Pemeriksaan tambahan",
            "",
            f"- Duplikasi vektor fitur persis pada peserta terverifikasi: {duplicate_count}.",
            f"- Baris legacy tanpa metadata yang dapat diverifikasi: {legacy_count}.",
            "",
            "## Batasan dan privasi",
            "",
            "Perekam menyimpan landmark dan metadata kondisi, bukan frame/video. "
            "Gunakan ID pseudonim, dapatkan persetujuan peserta, dan jangan "
            "menambahkan nama, email, atau pengenal langsung ke file dataset.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "checks": checks,
        "passed": passed_count,
        "total": len(checks),
        "report_path": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=1.5,
        help="Rasio kelas gesture terbesar/terkecil maksimum (default: 1.5)",
    )
    args = parser.parse_args()
    if args.balance_ratio < 1:
        parser.error("--balance-ratio harus minimal 1")
    try:
        result = audit_dataset(
            args.dataset,
            args.metadata,
            args.report,
            balance_ratio_limit=args.balance_ratio,
        )
    except (OSError, csv.Error, ValueError) as error:
        print(f"Audit gagal: {error}")
        return 1

    print(
        f"Audit selesai: {result['passed']}/{result['total']} indikator lulus. "
        f"Laporan: {result['report_path']}"
    )
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
