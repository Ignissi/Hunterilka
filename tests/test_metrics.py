"""Проверки метрик: бюджетный порог, первые K, разделение test по периодам."""

import unittest

import numpy as np
import pandas as pd

from scripts.evaluate import period_metrics
from src.data import split_boundaries, split_names
from src.metrics import budget_threshold, classification_metrics, top_k_metrics


class MetricTests(unittest.TestCase):
    def test_budget_threshold_respects_ties(self):
        scores = [0.9, 0.9, 0.9, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # Бюджет 2 из 10: три одинаковые оценки 0.9 не помещаются, сигналов нет.
        threshold = budget_threshold(scores, 0.2)
        self.assertEqual(int((np.array(scores) >= threshold).sum()), 0)
        # Бюджет 3: ровно три оценки 0.9.
        threshold = budget_threshold(scores, 0.3)
        self.assertEqual(int((np.array(scores) >= threshold).sum()), 3)

    def test_top_k_counts_positives(self):
        labels = [1, 0, 1, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.1, 0.2]
        result = top_k_metrics(labels, scores, budgets=(2, 3))
        self.assertEqual(result["2"]["found"], 1)
        self.assertEqual(result["3"]["found"], 2)
        self.assertEqual(classification_metrics(labels, scores, 0.75)["alerts"], 2)

    def test_period_metrics_split_tail(self):
        timestamps = pd.Series(pd.date_range("2022-09-09", "2022-09-13", freq="h").strftime("%Y-%m-%dT%H:%M:%S"))
        labels = np.zeros(len(timestamps), dtype=int)
        labels[::5] = 1
        scores = np.where(labels == 1, 0.9, 0.1)
        test = pd.DataFrame({"timestamp": timestamps, "is_laundering": labels})
        periods = period_metrics(test, scores, 0.5)
        self.assertEqual(periods["normal"]["last_time"][:10], "2022-09-10")
        self.assertEqual(periods["tail"]["first_time"][:10], "2022-09-11")
        self.assertEqual(periods["normal"]["rows"] + periods["tail"]["rows"], len(test))

    def test_period_metrics_without_tail(self):
        timestamps = pd.Series(["2022-09-09T00:00:00", "2022-09-09T01:00:00", "2022-09-09T02:00:00"])
        test = pd.DataFrame({"timestamp": timestamps, "is_laundering": [1, 0, 0]})
        periods = period_metrics(test, np.array([0.9, 0.1, 0.2]), 0.5)
        self.assertEqual(periods["tail"]["rows"], 0)
        self.assertEqual(periods["normal"]["rows"], 3)

    def test_split_keeps_equal_times_together(self):
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=10, freq="min").strftime("%Y-%m-%dT%H:%M:%S")).repeat(
            [10, 10, 10, 10, 15, 15, 5, 10, 5, 10]).reset_index(drop=True)
        train_end, validation_end = split_boundaries(timestamps)
        parts = split_names(timestamps, train_end, validation_end)
        grouped = pd.DataFrame({"timestamp": timestamps, "split": parts}).groupby("timestamp").split.nunique()
        self.assertTrue(grouped.eq(1).all())
        self.assertEqual(parts.value_counts().to_dict(), {"train": 55, "test": 25, "validation": 20})


if __name__ == "__main__":
    unittest.main()
