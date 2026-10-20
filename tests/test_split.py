"""Хронологическое разделение при редком хвосте истории и совпадающем времени."""

import unittest

import pandas as pd

from src.data import SPLIT_METHOD, chronological_boundaries, split_period, validate_split


class TemporalSplitTests(unittest.TestCase):
    def test_sparse_tail_does_not_absorb_validation_and_test(self):
        timestamps = pd.Series(
            list(pd.date_range("2026-01-01", periods=990, freq="min"))
            + list(pd.date_range("2026-01-18", periods=10, freq="min"))
        )
        train_end, validation_end = chronological_boundaries(timestamps)
        parts = split_period(timestamps, train_end, validation_end)
        self.assertEqual(parts.value_counts().to_dict(), {"train": 600, "validation": 200, "test": 200})
        self.assertLess(timestamps[parts == "train"].max(), timestamps[parts == "validation"].min())
        self.assertLess(timestamps[parts == "validation"].max(), timestamps[parts == "test"].min())

    def test_equal_times_stay_together(self):
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=10, freq="min")).repeat(
            [10, 10, 10, 10, 15, 15, 5, 10, 5, 10]
        ).reset_index(drop=True)
        boundaries = chronological_boundaries(timestamps)
        parts = split_period(timestamps, *boundaries)
        grouped = pd.DataFrame({"timestamp": timestamps, "split": parts}).groupby("timestamp").split.nunique()
        self.assertTrue(grouped.eq(1).all())
        self.assertEqual(parts.value_counts().to_dict(), {"train": 55, "test": 25, "validation": 20})
        self.assertEqual(boundaries, chronological_boundaries(timestamps.sample(frac=1, random_state=7)))

    def test_validation_rejects_degenerate_split(self):
        manifest = {
            "split_method": SPLIT_METHOD, "rows": 4930063,
            "train_end": "2022-09-11", "validation_end": "2022-09-15",
            "parts": {
                "train": {"rows": 4929200, "positive": 4611},
                "validation": {"rows": 740, "positive": 435},
                "test": {"rows": 123, "positive": 75},
            },
        }
        with self.assertRaisesRegex(ValueError, "менее 10%"):
            validate_split(manifest)

    def test_validation_accepts_chronological_split_and_rejects_one_class(self):
        manifest = {
            "split_method": SPLIT_METHOD, "rows": 1000,
            "train_end": "2026-01-02", "validation_end": "2026-01-03",
            "parts": {
                "train": {"rows": 600, "positive": 6},
                "validation": {"rows": 200, "positive": 2},
                "test": {"rows": 200, "positive": 3},
            },
        }
        validate_split(manifest)
        manifest["parts"]["test"]["positive"] = 0
        with self.assertRaisesRegex(ValueError, "нужны оба класса"):
            validate_split(manifest)


if __name__ == "__main__":
    unittest.main()
