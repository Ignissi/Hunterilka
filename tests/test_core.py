"""Малые фикстуры проверяют алгоритмы. Они не используются для обучения."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.cases import CaseService, build_case
from src.context import short_return_path, snapshot_context
from src.data import HEADER, account_key, prepare_data
from src.features import transaction_features
from src.graph import RollingGraph
from src.patterns import make_candidate, search_patterns
from src.pipeline import process_group
from src.storage import load_config
from src.visualization import case_graph


def event(number, source="A", target="B", timestamp="2026-01-01T00:00:00", currency="USD", amount="10.00"):
    return {
        "transaction_id": str(number), "source_row": number, "timestamp": timestamp,
        "source_bank": "01", "source_account": source, "target_bank": "02", "target_account": target,
        "source_key": account_key("01", source), "target_key": account_key("01", target),
        "amount_paid": amount, "amount_received": amount,
        "payment_currency": currency, "receiving_currency": currency, "payment_format": "Wire",
    }


def feed(state, events):
    for timestamp in sorted({e["timestamp"] for e in events}):
        group = [e for e in events if e["timestamp"] == timestamp]
        process_group(state, group, state.config)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.state = RollingGraph(self.config)

    def test_future_does_not_change_past(self):
        early = [event(1), event(2, timestamp="2026-01-01T01:00:00")]
        results = []
        for extra in [[], [event(3, timestamp="2026-01-02T00:00:00")]]:
            state = RollingGraph(self.config)
            values = []
            for item in early + extra:
                rows, _, _, _ = process_group(state, [item], self.config)
                values.append(json.dumps(rows, sort_keys=True))
            results.append(values[:2])
        self.assertEqual(results[0], results[1])

    def test_same_timestamp_has_no_internal_history(self):
        group = [event(1), event(2)]
        rows, _, _, _ = process_group(self.state, group, self.config)
        self.assertEqual([row["pair_count_window"] for row in rows], [0, 0])
        self.assertEqual([row["source_out_count_1h"] for row in rows], [0, 0])

    def test_permutation_same_timestamp(self):
        group = [event(1), event(2, "B", "C")]
        left, _, _, _ = process_group(self.state, group, self.config)
        right, _, _, _ = process_group(RollingGraph(self.config), list(reversed(group)), self.config)
        self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(list(reversed(right)), sort_keys=True))

    def test_label_independence(self):
        item = event(1)
        altered = dict(item, is_laundering=1)
        left = process_group(self.state, [item], self.config)[0]
        right = process_group(RollingGraph(self.config), [altered], self.config)[0]
        self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True))

    def test_repeat_pair_partial_expiry(self):
        self.config["window_days"] = 1
        feed(self.state, [event(1), event(2, timestamp="2026-01-01T12:00:00")])
        self.state.advance("2026-01-02T00:00:00")
        self.assertEqual(len(self.state.events), 1)
        self.assertEqual(self.state.graph.number_of_edges(), 1)
        self.assertEqual(next(iter(self.state.graph.edges(data=True)))[2]["count"], 1)

    def test_last_pair_exact_expiry(self):
        self.config["window_days"] = 1
        feed(self.state, [event(1)])
        self.state.advance("2026-01-02T00:00:00")
        self.assertEqual(self.state.graph.number_of_nodes(), 0)
        self.assertFalse(self.state.pair_events)

    def test_short_window_expires_independently(self):
        feed(self.state, [event(1)])
        self.state.advance("2026-01-01T01:00:00")
        key = event(1)["source_key"]
        self.assertEqual(self.state.statistics(1, "out", key, "USD")[0], 0)
        self.assertEqual(self.state.statistics(24, "out", key, "USD")[0], 1)
        self.assertEqual(len(self.state.events), 1)

    def test_currency_aggregates(self):
        feed(self.state, [event(1, currency="USD"), event(2, currency="EUR", amount="20.00")])
        key = event(1)["source_key"]
        self.assertEqual(self.state.statistics(24, "out", key, "USD")[2], 10)
        self.assertEqual(self.state.statistics(24, "out", key, "EUR")[2], 20)

    def test_duplicate_group_rejected(self):
        feed(self.state, [event(1)])
        with self.assertRaises(ValueError):
            self.state.add_group([event(1)])
        self.assertEqual(len(self.state.events), 1)

    def test_late_event_rejected(self):
        feed(self.state, [event(1, timestamp="2026-01-02T00:00:00")])
        with self.assertRaises(ValueError):
            self.state.advance("2026-01-01T00:00:00")

    def test_high_degree_search_budget(self):
        feed(self.state, [event(i, "B", "N" + str(i)) for i in range(30)])
        found, complete = short_return_path(self.state.graph, account_key("01", "Z"), account_key("01", "B"), 5)
        self.assertFalse(found)
        self.assertFalse(complete)

    def test_chronological_cycle(self):
        events = [event(1, "A", "B"), event(2, "B", "C", "2026-01-01T01:00:00"), event(3, "C", "A", "2026-01-01T02:00:00")]
        feed(self.state, events)
        candidates, _ = search_patterns(self.state, [events[-1]], self.config)
        self.assertTrue(any(c["family"] == "cycle" for c in candidates))

    def test_wrong_temporal_cycle_not_confirmed(self):
        events = [event(1, "B", "C"), event(2, "A", "B", "2026-01-01T01:00:00"), event(3, "C", "A", "2026-01-01T02:00:00")]
        feed(self.state, events)
        candidates, _ = search_patterns(self.state, [events[-1]], self.config)
        self.assertFalse(any(c["family"] == "cycle" for c in candidates))

    def test_all_supported_families(self):
        early = [event(i + 1, "S" + str(i), "M") for i in range(3)]
        late = [event(i + 4, "M", "T" + str(i), "2026-01-01T01:00:00") for i in range(3)]
        feed(self.state, early + late)
        candidates, _ = search_patterns(self.state, late, self.config)
        families = {c["family"] for c in candidates}
        self.assertTrue({"fan_out", "gather_scatter"}.issubset(families))

        state = RollingGraph(self.config)
        branch = [event(11, "A", "B"), event(12, "A", "C"), event(13, "B", "D", "2026-01-01T01:00:00"), event(14, "C", "D", "2026-01-01T01:00:00")]
        feed(state, branch)
        candidates, _ = search_patterns(state, branch[-2:], self.config)
        self.assertTrue(any(c["family"] == "scatter_gather" for c in candidates))

    def test_chain_and_fan_in(self):
        chain = [event(1, "A", "B"), event(2, "B", "C", "2026-01-01T01:00:00"), event(3, "C", "D", "2026-01-01T02:00:00")]
        feed(self.state, chain)
        candidates, _ = search_patterns(self.state, [chain[-1]], self.config)
        self.assertTrue(any(c["family"] == "chain" for c in candidates))
        state = RollingGraph(self.config)
        star = [event(i + 5, "A" + str(i), "B") for i in range(3)]
        _, _, candidates, _ = process_group(state, star, self.config)
        self.assertTrue(any(c["family"] == "fan_in" for c in candidates))

    def test_equal_time_cycle_marked(self):
        group = [event(1, "A", "B"), event(2, "B", "C"), event(3, "C", "A")]
        _, _, candidates, _ = process_group(self.state, group, self.config)
        cycles = [c for c in candidates if c["family"] == "cycle"]
        self.assertTrue(cycles)
        self.assertTrue(all(c["equal_timestamps"] for c in cycles))

    def test_search_does_not_need_model_score(self):
        group = [event(i, "A", "X" + str(i)) for i in range(4)]
        _, scores, candidates, _ = process_group(self.state, group, self.config)
        self.assertIsNone(scores)
        self.assertTrue(any(c["family"] == "fan_out" for c in candidates))

    def test_new_account(self):
        self.state.advance(event(1)["timestamp"])
        row = transaction_features(event(1), self.state)
        self.assertEqual(row["source_is_new"], 1)
        self.assertEqual(row["paid_ratio_missing"], 1)

    def test_case_snapshot_and_review(self):
        candidate = make_candidate("transaction", [event(1)], ["1"], event(1)["timestamp"])
        models = {"meta": {"transaction_threshold": .5, "structure_threshold": .5, "model_id": "test", "config_hash": "x", "data_hash": "y", "feature_version": "1"}}
        case = build_case(candidate, {"1": .9}, {"1": {"value": 0.1}}, models, "test")
        with tempfile.TemporaryDirectory() as directory:
            service = CaseService(directory)
            service.save(case)
            service.save(copy.deepcopy(case))
            service.flush()
            service.review(case["case_id"], "Рассмотрен", "Проверено")
            restored = CaseService(directory).get(case["case_id"])
            self.assertEqual(restored["version"], 1)
            self.assertEqual(restored["comment"], "Проверено")
            self.assertEqual(restored["events"][0]["amount_paid"], "10.00")
            self.config["window_days"] = 1
            feed(self.state, [event(1)])
            self.state.advance("2026-01-03T00:00:00")
            self.assertFalse(self.state.events)
            self.assertEqual(service.get(case["case_id"])["events"][0]["amount_paid"], "10.00")

    def test_new_version_preserves_comment(self):
        candidate = make_candidate("transaction", [event(1)], ["1"], event(1)["timestamp"])
        models = {"meta": {"transaction_threshold": .5, "structure_threshold": .5, "model_id": "test", "config_hash": "x", "data_hash": "y", "feature_version": "1"}}
        case = build_case(candidate, {"1": .9}, {"1": {"value": .1}}, models, "test")
        with tempfile.TemporaryDirectory() as directory:
            service = CaseService(directory)
            service.save(case)
            service.review(case["case_id"], "Требует проверки", "Не потерять")
            changed = copy.deepcopy(case)
            changed["candidate_id"] = "new-evidence"
            service.save(changed)
            self.assertEqual(service.get(case["case_id"])["version"], 2)
            self.assertEqual(service.get(case["case_id"])["comment"], "Не потерять")
            self.assertEqual(len(list((Path(directory) / "versions").glob("*.json"))), 1)

    def test_import_ids_labels_and_complete_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADER)
                writer.writerows([
                    ["2026/01/01 02:00", "001", "000A", "002", "000B", "10.00", "USD", "10.00", "USD", "Wire", "0"],
                    ["2026/01/01 00:00", "001", "000A", "002", "000B", "20.00", "USD", "20.00", "USD", "Wire", "1"],
                    ["2026/01/01 00:00", "001", "000A", "002", "000B", "20.00", "USD", "20.00", "USD", "Wire", "1"],
                ])
            output = root / "processed"
            prepare_data(source, output, limit=1)
            events = pd.read_parquet(output / "events.parquet")
            self.assertEqual(len(events), 2)
            self.assertEqual(events.source_row.tolist(), [2, 3])
            self.assertEqual(events.source_bank.tolist(), ["001", "001"])
            self.assertNotIn("is_laundering", events.columns)
            self.assertEqual(events.transaction_id.nunique(), 2)
            before = events.to_json()
            prepare_data(source, output, limit=1)
            self.assertEqual(before, pd.read_parquet(output / "events.parquet").to_json())

    def test_import_invalid_and_crypto_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(HEADER)
                writer.writerows([
                    ["2026/01/01 00:00", "1", "A", "2", "B", "10", "USD", "10", "USD", "Wire", "0"],
                    ["2026/01/01 01:00", "1", "A", "2", "B", "10", "Bitcoin", "10", "USD", "Wire", "1"],
                    ["wrong-time", "1", "A", "2", "B", "-1", "USD", "-1", "USD", "Wire", "0"],
                ])
            manifest = prepare_data(source, root / "processed", limit=0)
            self.assertEqual(manifest["excluded_crypto"], 1)
            self.assertEqual(manifest["invalid_rows"], 1)
            self.assertEqual(manifest["rows"], 1)

    def test_offline_graph_and_display_limit(self):
        events = [event(i, "A", "B" + str(i)) for i in range(80)]
        case = make_candidate("fan_out", events, ["A"], events[0]["timestamp"])
        rendered, info = case_graph(case, self.config)
        self.assertLessEqual(info["nodes"], 40)
        self.assertLess(info["shown"], info["total"])
        self.assertNotRegex(rendered, r'(src|href)="https?://')

    def test_composite_account_key(self):
        self.assertNotEqual(account_key("1", "23"), account_key("12", "3"))
        self.assertNotEqual(account_key("1", "001"), account_key("1", "1"))


if __name__ == "__main__":
    unittest.main()
