"""Окружение не показывает будущее, а рёбра сохраняют отдельные переводы."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.cases import CaseService, build_case
from src.patterns import make_candidate
from src.snapshot import HistorySlices
from src.storage import load_config
from src.visualization import case_graph
from src.workspace import open_saved_cases
from tests.test_core import event
from tests import test_review


class WorkspaceTests(unittest.TestCase):
    def test_context_excludes_future_and_expired_even_if_input_unsorted(self):
        evidence = event(1, "A", "B", "2026-02-05T12:00:00")
        rows = [event(2, "C", "A", "2026-02-04T12:00:00"),
                event(3, "B", "D", "2026-02-06T12:00:00"),
                event(4, "A", "E", "2026-01-01T12:00:00"), evidence,
                event(5, "F", "C", "2026-02-03T12:00:00")]
        context, _ = HistorySlices(pd.DataFrame(rows)).around([evidence], evidence["timestamp"])
        self.assertEqual({row["transaction_id"] for row in context}, {"2", "5"})

    def test_context_respects_limits(self):
        evidence = event(1, "A", "B")
        rows = [evidence] + [event(i + 2, "A", "N" + str(i)) for i in range(100)]
        context, truncated = HistorySlices(pd.DataFrame(rows)).around([evidence], evidence["timestamp"], node_limit=8, event_limit=6)
        self.assertTrue(truncated)
        self.assertLessEqual(len(context), 6)
        nodes = {row[side] for row in context + [evidence] for side in ["source_key", "target_key"]}
        self.assertLessEqual(len(nodes), 8)

    def test_parallel_transfers_keep_evidence_separate_from_context(self):
        first = event(1, "A", "B")
        second = event(2, "B", "C")
        case = make_candidate("chain", [first, second], ["A", "B", "C"], first["timestamp"])
        case["context"] = [event(3, "A", "B"), event(4, "D", "A")]
        rendered, counts = case_graph(case, load_config())
        self.assertEqual(counts["edges"], 4)
        self.assertEqual(counts["pairs"], 3)
        self.assertEqual(counts["evidence_shown"], 2)
        self.assertIn('"evidence": false', rendered)
        self.assertIn('"evidence": true', rendered)
        self.assertIn('graph-fit', rendered)
        self.assertNotRegex(rendered, r'(src|href)="https?://')
        _, compact = case_graph(case, load_config(), show_context=False)
        self.assertEqual(compact["edges"], 2)

    def test_all_supported_shapes_render(self):
        families = {
            "chain": [("A", "B"), ("B", "C")],
            "cycle": [("A", "B"), ("B", "C"), ("C", "A")],
            "reciprocal": [("A", "B"), ("B", "A")],
            "fan_in": [("A", "D"), ("B", "D"), ("C", "D")],
            "fan_out": [("D", "A"), ("D", "B"), ("D", "C")],
            "gather_scatter": [("A", "M"), ("B", "M"), ("M", "C"), ("M", "D")],
            "scatter_gather": [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        }
        for family, pairs in families.items():
            with self.subTest(family=family):
                rows = [event(i, source, target) for i, (source, target) in enumerate(pairs)]
                case = make_candidate(family, rows, [], rows[0]["timestamp"])
                _, info = case_graph(case, load_config())
                self.assertEqual(info["evidence_shown"], len(rows))

    def test_workspace_upgrade_preserves_review(self):
        row = event(1, "A", "B")
        candidate = make_candidate("transaction", [row], ["1"], row["timestamp"])
        case = build_case(candidate, {"1": .9}, {}, test_review.ReviewTests().models(), "test")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = CaseService(root / "demo/cases")
            original.save(case)
            original.flush()
            previous = CaseService(root / "runs/demo_review/test/old/cases")
            previous.save(dict(case))
            previous.review(case["case_id"], "Рассмотрен", "Проверены документы")
            with patch("src.workspace.ROOT", root):
                opened = open_saved_cases({"model_id": "test", "revision": "new"})
            restored = opened.get(case["case_id"])
            self.assertEqual(restored["status"], "Рассмотрен")
            self.assertEqual(restored["comment"], "Проверены документы")


if __name__ == "__main__":
    unittest.main()
