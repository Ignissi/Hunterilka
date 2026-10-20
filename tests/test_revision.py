"""Проверки исправлений: охват поиска, независимость порядка и выбор порога."""

import copy
import tempfile
import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.graph import RollingGraph
from src.metrics import budget_threshold, top_k_metrics
from src.pipeline import process_group
from src.storage import load_config
from scripts.build_features import append_table
from tests.test_core import event


class RevisionTests(unittest.TestCase):
    def test_search_reaches_event_after_first_128(self):
        config = load_config()
        state = RollingGraph(config)
        process_group(state, [event(1, "A", "B")], config)
        process_group(state, [event(2, "B", "C", "2026-01-01T01:00:00")], config)
        timestamp = "2026-01-01T02:00:00"
        group = [event(i + 10, "X" + str(i), "Y" + str(i), timestamp) for i in range(200)]
        group.append(event(999, "C", "A", timestamp))
        _, _, candidates, info = process_group(state, group, config)
        self.assertTrue(any(c["family"] == "cycle" and "999" in c["transaction_ids"] for c in candidates))
        self.assertEqual(info["searched_seeds"], 201)
        self.assertEqual(info["skipped_seeds"], 0)

    def test_candidate_limit_does_not_depend_on_input_order(self):
        config = load_config()
        config["candidates_per_group"] = 2
        group = [event(i, "A", "N" + str(i)) for i in range(5)]
        group += [event(i + 20, "N" + str(i), "A") for i in range(5)]
        outputs = []
        for ordered in [group, list(reversed(group))]:
            _, _, candidates, info = process_group(RollingGraph(config), ordered, config)
            outputs.append([c["candidate_id"] for c in candidates])
            self.assertGreater(info["candidate_limit_dropped"], 0)
            self.assertLessEqual(len(candidates), 2)
        self.assertEqual(outputs[0], outputs[1])

    def test_search_does_not_use_attached_labels(self):
        config = load_config()
        group = [event(i, "A", "N" + str(i)) for i in range(5)]
        changed = [dict(row, is_laundering=i % 2) for i, row in enumerate(group)]
        left = process_group(RollingGraph(config), group, config)[2]
        right = process_group(RollingGraph(config), changed, config)[2]
        self.assertEqual([c["candidate_id"] for c in left], [c["candidate_id"] for c in right])
        self.assertEqual([c["features"] for c in left], [c["features"] for c in right])

    def test_scan_bound_scales_with_new_operations(self):
        config = load_config()
        group = [event(i, "A", "N" + str(i)) for i in range(40)]
        _, _, _, info = process_group(RollingGraph(config), group, config)
        bound = len(group) * (config["path_scan_limit"] + config["composition_scan_limit"])
        bound += info["searched_centers"] * config["center_scan_limit"]
        self.assertLessEqual(info["scanned"], bound)

    def test_threshold_ties_do_not_exceed_budget(self):
        values = pd.Series([.9, .8, .8, .8, .1])
        threshold = budget_threshold(values, .4)
        self.assertEqual(int((values >= threshold).sum()), 1)
        self.assertEqual(budget_threshold([.9], 1), .9)
        with self.assertRaises(ValueError):
            budget_threshold(values, 0)

    def test_top_k_counts_hits_and_actual_size(self):
        result = top_k_metrics([0, 1, 1], [.9, .8, .1], budgets=(1, 100))
        self.assertEqual(result["1"]["found"], 0)
        self.assertEqual(result["100"]["found"], 2)
        self.assertEqual(result["100"]["reviewed"], 3)

    def test_live_signal_saves_context_with_current_config(self):
        from src.cases import CaseService, register_signals
        config = load_config()
        state = RollingGraph(config)
        first = event(1, "A", "B")
        second = event(2, "B", "C", "2026-01-01T01:00:00")
        process_group(state, [first], config)
        process_group(state, [second], config)
        models = {"meta": {"transaction_threshold": .5, "structure_threshold": .5,
                  "model_id": "test", "config_hash": "x", "data_hash": "y", "feature_version": "1.1"}}
        with tempfile.TemporaryDirectory() as folder:
            service = CaseService(folder)
            register_signals(service, state, [second], [], {"2": .9}, {"2": {"value": 1}}, models, "test")
            case = service.get(next(iter(service.index)))
            self.assertEqual(case["transaction_ids"], ["2"])
            self.assertTrue(any(row["transaction_id"] == "1" for row in case["context"]))

    def test_later_non_signal_does_not_erase_saved_case(self):
        from scripts.evaluate import read_structures
        rows = [
            {"case_key": "episode", "transaction_ids": ["positive"], "family": "chain",
             "detected_at": "2026-01-02T01:00:00", "features": {"value": .9}},
            {"case_key": "episode", "transaction_ids": ["ordinary"], "family": "chain",
             "detected_at": "2026-01-02T02:00:00", "features": {"value": .1}},
        ]
        model = SimpleNamespace(score_samples=lambda frame: -frame.value.to_numpy())
        models = {"structure": model, "meta": {"structure_features": ["value"],
                  "transaction_threshold": .5, "structure_threshold": .5}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "artifacts").mkdir()
            (root / "artifacts/candidates.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with patch("scripts.evaluate.ROOT", root):
                latest, saved, covered, _ = read_structures(models, {"validation_end": "2026-01-02"}, {"positive"}, {})
            self.assertEqual(latest[0]["transaction_ids"], ["ordinary"])
            self.assertEqual(saved[0]["transaction_ids"], ["positive"])
            self.assertEqual(covered, {"positive"})

    def test_parquet_batches_preserve_missing_numeric_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.parquet"
            writer = append_table(None, [{"ratio": float("nan"), "name": "A"}], path)
            writer = append_table(writer, [{"ratio": 2.0, "name": "B"}], path)
            writer.close()
            result = pd.read_parquet(path)
            self.assertEqual(len(result), 2)
            self.assertTrue(pd.isna(result.ratio.iloc[0]))
            self.assertEqual(result.ratio.iloc[1], 2)


if __name__ == "__main__":
    unittest.main()
