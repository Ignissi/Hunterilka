"""Очереди сохраняют ранжирование модели и не используют метки."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.evaluate import save_review_sample
from src.cases import CaseService, build_case
from src.metrics import top_k_metrics
from src.patterns import make_candidate
from src.review_queue import operation_order, ordered_cases
from src.storage import read_json
from tests.test_core import event


class ReviewTests(unittest.TestCase):
    def models(self):
        return {"meta": {"transaction_threshold": .8, "structure_threshold": .7, "features": ["value"],
                "model_id": "test", "config_hash": "x", "data_hash": "y", "feature_version": "1.1"}}

    def test_structures_do_not_displace_operations(self):
        rows = [
            {"case_id": "strong", "family": "transaction", "operation_score": .99, "anomaly_score": None},
            {"case_id": "weak", "family": "transaction", "operation_score": .4, "anomaly_score": None},
            {"case_id": "structure", "family": "chain", "operation_score": .1, "anomaly_score": .999},
        ]
        self.assertEqual(ordered_cases(rows, "operations").case_id.tolist(), ["strong", "weak"])
        self.assertEqual(ordered_cases(rows, "structures").case_id.tolist(), ["structure"])
        changed = [dict(row, is_laundering=i % 2) for i, row in enumerate(reversed(rows))]
        self.assertEqual(ordered_cases(changed, "operations").case_id.tolist(), ["strong", "weak"])

    def test_equal_scores_match_metric_order(self):
        scores = [.1, .8, .8, .4]
        labels = [0, 1, 0, 1]
        order = operation_order(scores, limit=2)
        measured = top_k_metrics(labels, scores, budgets=(2,))["2"]
        self.assertEqual(order.tolist(), [1, 2])
        self.assertEqual(sum(labels[i] for i in order), measured["found"])

    def test_ranked_operation_does_not_become_threshold_signal(self):
        row = event(1, "A", "B")
        candidate = make_candidate("transaction", [row], ["1"], row["timestamp"])
        self.assertIsNone(build_case(candidate, {"1": .6}, {}, self.models(), "test"))
        case = build_case(candidate, {"1": .6}, {}, self.models(), "test", review_rank=2)
        self.assertFalse(case["operation_signal"])
        self.assertFalse(case["structure_signal"])
        self.assertEqual(case["review_rank"], 2)

    def test_rank_does_not_bypass_structure_threshold(self):
        rows = [event(1, "A", "B"), event(2, "B", "C")]
        candidate = make_candidate("chain", rows, ["A", "B", "C"], rows[-1]["timestamp"])
        candidate["anomaly_score"] = .2
        self.assertIsNone(build_case(candidate, {"1": .1, "2": .2}, {}, self.models(), "test", review_rank=1))

    def test_saved_queue_preserves_ties_and_features_without_labels(self):
        events = [event(i, "A", "B") for i in [9, 2, 3]]
        test = pd.DataFrame({"transaction_id": ["9", "2", "3"], "value": [3, 4, 5], "is_laundering": [0, 1, 1]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            pd.DataFrame(events).to_parquet(root / "data/processed/events.parquet", index=False)
            with patch("scripts.evaluate.ROOT", root):
                policy = save_review_sample(test, [.9, .9, .6], self.models(), {"source_url": "test"}, [], "previous_sample")
            service = CaseService(root / "demo/cases")
            ordered = ordered_cases(list(service.index.values()), "operations")
            cases = [service.get(i) for i in ordered.case_id]
            self.assertEqual([c["transaction_ids"][0] for c in cases], ["9", "2", "3"])
            self.assertEqual(cases[-1]["features"], {"3": {"value": 5}})
            self.assertFalse(cases[-1]["operation_signal"])
            self.assertNotIn("is_laundering", str(cases))
            self.assertEqual(policy["operation_cases"], 3)
            self.assertEqual(read_json(root / "demo/demo_manifest.json")["cases"], 3)

    def test_missing_event_keeps_previous_cases(self):
        test = pd.DataFrame({"transaction_id": ["1", "2"], "value": [3, 4]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            (root / "demo/cases").mkdir(parents=True)
            previous = root / "demo/cases/previous.json"
            previous.write_text("old")
            pd.DataFrame([event(1, "A", "B")]).to_parquet(root / "data/processed/events.parquet", index=False)
            with patch("scripts.evaluate.ROOT", root), self.assertRaises(ValueError):
                save_review_sample(test, [.9, .8], self.models(), {"source_url": "test"}, [], "previous_sample")
            self.assertEqual(previous.read_text(), "old")


if __name__ == "__main__":
    unittest.main()
