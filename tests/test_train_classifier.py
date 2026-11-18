import unittest

import numpy as np

from train_classifier import (
    CLASS_ORDER,
    format_confusion_matrix,
    split_by_participant,
)


class ParticipantSplitTests(unittest.TestCase):
    def test_split_keeps_participants_disjoint_and_all_classes(self) -> None:
        labels = np.asarray(CLASS_ORDER * 3)
        participants = np.repeat(["P001", "P002", "P003"], len(CLASS_ORDER))
        features = np.zeros((len(labels), 63))

        train_indices, test_indices = split_by_participant(
            features,
            labels,
            participants,
            test_size=0.34,
            random_state=42,
        )

        self.assertFalse(
            set(participants[train_indices]).intersection(participants[test_indices])
        )
        self.assertEqual(set(labels[train_indices]), set(CLASS_ORDER))
        self.assertEqual(set(labels[test_indices]), set(CLASS_ORDER))

    def test_split_rejects_a_single_participant(self) -> None:
        labels = np.asarray(CLASS_ORDER)
        participants = np.repeat("P001", len(CLASS_ORDER))

        with self.assertRaisesRegex(ValueError, "minimal dua"):
            split_by_participant(
                np.zeros((len(labels), 63)),
                labels,
                participants,
            )

    def test_confusion_matrix_displays_unknown_row_and_column(self) -> None:
        output = format_confusion_matrix(np.eye(len(CLASS_ORDER), dtype=int))

        lines = output.splitlines()
        self.assertEqual(lines[1], "aktual\\prediksi\t" + "\t".join(CLASS_ORDER))
        self.assertTrue(lines[-1].startswith("unknown\t"))


if __name__ == "__main__":
    unittest.main()
