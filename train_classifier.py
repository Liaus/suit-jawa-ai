"""Train and evaluate a participant-independent Suit Jawa classifier."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Sequence, TypedDict, cast

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC

from src.features.landmark_normalizer import FEATURE_COLUMNS
from src.utils.csv_validator import (
    DATASET_COLUMNS,
    LEGACY_PARTICIPANT_ID,
    VALID_LABELS,
    validate_dataset_row,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "dataset_suit_jawa.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "suit_jawa_classifier.joblib"
CLASS_ORDER = ("Gajah", "Orang", "Semut", "unknown")


class TrainingResult(TypedDict):
    train_participants: list[str]
    test_participants: list[str]
    train_samples: int
    test_samples: int
    confusion_matrix: np.ndarray
    classification_report: str
    model_path: Path
    participant_count: int


def load_training_data(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load verified participant samples and exclude legacy rows."""
    features: list[list[float]] = []
    labels: list[str] = []
    participants: list[str] = []
    with dataset_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(DATASET_COLUMNS):
            raise ValueError(
                f"Skema dataset {dataset_path} harus memiliki tepat "
                f"{len(DATASET_COLUMNS)} kolom yang diharapkan."
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Baris dataset {row_number} tidak lengkap.")
            validate_dataset_row(row)
            participant_id = row["participant_id"]
            if participant_id == LEGACY_PARTICIPANT_ID:
                continue
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
            labels.append(row["label"])
            participants.append(participant_id)

    if not features:
        raise ValueError("Tidak ada sampel peserta terverifikasi untuk training.")
    missing_labels = VALID_LABELS.difference(labels)
    if missing_labels:
        raise ValueError(
            "Data peserta terverifikasi belum mencakup semua label: "
            + ", ".join(sorted(missing_labels))
        )
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels),
        np.asarray(participants),
    )


def split_by_participant(
    features: np.ndarray,
    labels: np.ndarray,
    participants: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a group split with all labels represented in train and test."""
    participant_ids = set(participants.tolist())
    if len(participant_ids) < 2:
        raise ValueError("Training memerlukan minimal dua participant_id berbeda.")

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=test_size,
        random_state=random_state,
    )
    candidates: list[tuple[tuple[float, float], np.ndarray, np.ndarray]] = []
    for train_indices, test_indices in splitter.split(features, labels, participants):
        if (
            set(labels[train_indices]) != set(CLASS_ORDER)
            or set(labels[test_indices]) != set(CLASS_ORDER)
        ):
            continue
        test_participants = set(participants[test_indices])
        train_participants = set(participants[train_indices])
        if train_participants.intersection(test_participants):
            continue
        group_fraction_error = abs(
            len(test_participants) / len(participant_ids) - test_size
        )
        sample_fraction_error = abs(len(test_indices) / len(labels) - test_size)
        candidates.append(
            (
                (group_fraction_error, sample_fraction_error),
                train_indices,
                test_indices,
            )
        )

    if not candidates:
        raise ValueError(
            "Tidak ditemukan split peserta dengan semua empat label di kedua set. "
            "Pastikan setiap label direkam oleh sedikitnya dua peserta."
        )
    _, train_indices, test_indices = min(candidates, key=lambda item: item[0])
    return train_indices, test_indices


def format_confusion_matrix(matrix: np.ndarray) -> str:
    """Format the confusion matrix with a fixed, explicit class order."""
    lines = ["Confusion Matrix (baris=aktual, kolom=prediksi)"]
    lines.append("aktual\\prediksi\t" + "\t".join(CLASS_ORDER))
    for label, row in zip(CLASS_ORDER, matrix):
        lines.append(label + "\t" + "\t".join(str(int(value)) for value in row))
    return "\n".join(lines)


def train_classifier(
    dataset_path: Path,
    model_path: Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TrainingResult:
    """Fit, evaluate, and save a random-forest model."""
    features, labels, participants = load_training_data(dataset_path)
    train_indices, test_indices = split_by_participant(
        features,
        labels,
        participants,
        test_size=test_size,
        random_state=random_state,
    )
    train_participants = sorted(set(participants[train_indices]))
    test_participants = sorted(set(participants[test_indices]))
    training_label_counts = {
        label: int(np.count_nonzero(labels[train_indices] == label))
        for label in CLASS_ORDER
    }
    calibration_folds = min(training_label_counts.values())
    if calibration_folds < 2:
        raise ValueError(
            "Set training harus memiliki minimal dua sampel per label untuk "
            "mengkalibrasi probabilitas model."
        )
    model = CalibratedClassifierCV(
        estimator=SVC(class_weight="balanced"),
        cv=min(5, calibration_folds),
        ensemble=False,
    )
    model.fit(features[train_indices], labels[train_indices])
    predictions = model.predict(features[test_indices])
    matrix = confusion_matrix(
        labels[test_indices],
        predictions,
        labels=CLASS_ORDER,
    )
    report = cast(
        str,
        classification_report(
            labels[test_indices],
            predictions,
            labels=CLASS_ORDER,
            target_names=CLASS_ORDER,
            zero_division=0,
            output_dict=False,
        ),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(FEATURE_COLUMNS),
            "labels": list(CLASS_ORDER),
        },
        model_path,
    )
    result: TrainingResult = {
        "train_participants": train_participants,
        "test_participants": test_participants,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "confusion_matrix": matrix,
        "classification_report": report,
        "model_path": model_path,
        "participant_count": len(set(participants)),
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proporsi participant_id untuk test (default: 0.25)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)
    if not 0 < args.test_size < 1:
        parser.error("--test-size harus di antara 0 dan 1.")

    try:
        result = train_classifier(
            args.dataset,
            args.model_output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (OSError, ValueError, csv.Error) as error:
        print(f"Training gagal: {error}", file=sys.stderr)
        return 1

    print(f"Participant ID pada Train Set: {', '.join(result['train_participants'])}")
    print(f"Participant ID pada Test Set: {', '.join(result['test_participants'])}")
    print(
        f"Jumlah sampel: train={result['train_samples']}, "
        f"test={result['test_samples']}"
    )
    if result["participant_count"] < 5:
        print(
            "Peringatan: dataset memiliki kurang dari 5 peserta terverifikasi; "
            "hasil evaluasi awal belum mewakili variasi pengguna yang cukup."
        )
    print(format_confusion_matrix(result["confusion_matrix"]))
    print("Classification Report:")
    print(result["classification_report"])
    print(f"Model tersimpan ke {result['model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
