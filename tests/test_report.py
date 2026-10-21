"""Отчёт отражает весь контрольный период, сохраняя дополнительную диагностику."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.report import write_report


class ReportTests(unittest.TestCase):
    def test_whole_test_in_readme_and_slices_in_report(self):
        top = {"100": {"found": 2, "reviewed": 5, "precision": .4, "recall": 1.0},
               "500": {"found": 2, "reviewed": 5, "precision": .4, "recall": 1.0}}
        model = dict(rows=5, positive=2, prevalence=.4, ap=.7, roc_auc=.8,
                     top_k=top, threshold=.6, alerts=2, confusion_matrix=[[3, 0], [0, 2]])
        part = dict(model, first_time="2026-01-01", last_time="2026-01-02")
        model["periods"] = {"normal": part, "tail": {"rows": 0}}
        structure = dict(episodes=2, ap=.5, roc_auc=.5, prevalence=.5,
                         coverage=dict(in_any_candidate=2, test_positive_operations=2,
                                       in_structure_alerts=1, additional_from_structures=0))
        report = dict(data=dict(scope="full_source", rows=20, history_days=4,
                                first_time="2026-01-01", last_time="2026-01-05"),
                      selected="B_plain", training_device="CPU", transaction_models={"B_plain": model},
                      structure=structure)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "README.md").write_text("intro\n<!-- RESULTS_START -->\nold\n<!-- RESULTS_END -->\nend")
            with patch("src.report.ROOT", root):
                write_report(report)
            readme = (root / "README.md").read_text()
            self.assertIn("всём отложенном test", readme)
            self.assertNotIn("На обычном периоде", readme)
            self.assertTrue(readme.endswith("\nend"))
            details = (root / "docs/results.md").read_text()
            self.assertIn("Дополнительная проверка", details)
            self.assertIn("2026-01-01", details)


if __name__ == "__main__":
    unittest.main()
