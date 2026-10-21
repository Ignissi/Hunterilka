"""Проверки ядра: время, окна, признаки, формы структур, импорт."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.context import short_return_path
from src.data import HEADER, account_key, prepare_data
from src.features import transaction_features
from src.graph import RollingGraph
from src.patterns import search_patterns
from src.pipeline import process_group
from src.storage import load_config, load_table
from tests.helpers import event


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

    def test_same_minute_has_no_internal_history(self):
        rows, _, _, _ = process_group(self.state, [event(1), event(2)], self.config)
        self.assertEqual([row["pair_count_window"] for row in rows], [0, 0])
        self.assertEqual([row["source_out_count_1h"] for row in rows], [0, 0])

    def test_order_inside_minute_does_not_matter(self):
        group = [event(1), event(2, "B", "C")]
        left, _, _, _ = process_group(self.state, group, self.config)
        right, _, _, _ = process_group(RollingGraph(self.config), list(reversed(group)), self.config)
        self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(list(reversed(right)), sort_keys=True))

    def test_label_does_not_change_features(self):
        item = event(1)
        altered = dict(item, is_laundering=1)
        left = process_group(self.state, [item], self.config)[0]
        right = process_group(RollingGraph(self.config), [altered], self.config)[0]
        self.assertEqual(json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True))

    def test_window_expiry(self):
        self.config["window_days"] = 1
        feed(self.state, [event(1), event(2, timestamp="2026-01-01T12:00:00")])
        self.state.advance("2026-01-02T00:00:00")
        self.assertEqual(len(self.state.events), 1)
        self.assertEqual(self.state.graph.number_of_edges(), 1)
        self.state.advance("2026-01-03T00:00:00")
        self.assertEqual(self.state.graph.number_of_nodes(), 0)
        self.assertFalse(self.state.pair_events)

    def test_short_window_expires_independently(self):
        feed(self.state, [event(1)])
        self.state.advance("2026-01-01T01:00:00")
        key = event(1)["source_key"]
        self.assertEqual(self.state.statistics(1, "out", key, "USD")[0], 0)
        self.assertEqual(self.state.statistics(24, "out", key, "USD")[0], 1)
        self.assertEqual(len(self.state.events), 1)

    def test_currency_sums_are_separate(self):
        feed(self.state, [event(1, currency="USD"), event(2, currency="EUR", amount="20.00")])
        key = event(1)["source_key"]
        self.assertEqual(self.state.statistics(24, "out", key, "USD")[2], 10)
        self.assertEqual(self.state.statistics(24, "out", key, "EUR")[2], 20)

    def test_duplicate_and_late_events_rejected(self):
        feed(self.state, [event(1, timestamp="2026-01-02T00:00:00")])
        with self.assertRaises(ValueError):
            self.state.add_group([event(1, timestamp="2026-01-02T00:00:00")])
        with self.assertRaises(ValueError):
            self.state.advance("2026-01-01T00:00:00")

    def test_search_budget_stops_walk(self):
        feed(self.state, [event(i, "B", "N" + str(i)) for i in range(30)])
        found, complete = short_return_path(self.state.graph, account_key("01", "Z"), account_key("01", "B"), 5)
        self.assertFalse(found)
        self.assertFalse(complete)

    def test_cycle_requires_time_order(self):
        good = [event(1, "A", "B"), event(2, "B", "C", "2026-01-01T01:00:00"), event(3, "C", "A", "2026-01-01T02:00:00")]
        feed(self.state, good)
        candidates, _ = search_patterns(self.state, [good[-1]], self.config)
        self.assertTrue(any(c["family"] == "cycle" for c in candidates))

        state = RollingGraph(self.config)
        bad = [event(1, "B", "C"), event(2, "A", "B", "2026-01-01T01:00:00"), event(3, "C", "A", "2026-01-01T02:00:00")]
        feed(state, bad)
        candidates, _ = search_patterns(state, [bad[-1]], self.config)
        self.assertFalse(any(c["family"] == "cycle" for c in candidates))

    def test_all_families_are_found(self):
        early = [event(i + 1, "S" + str(i), "M") for i in range(3)]
        late = [event(i + 4, "M", "T" + str(i), "2026-01-01T01:00:00") for i in range(3)]
        feed(self.state, early + late)
        candidates, _ = search_patterns(self.state, late, self.config)
        self.assertTrue({"fan_out", "gather_scatter"}.issubset({c["family"] for c in candidates}))

        state = RollingGraph(self.config)
        branch = [event(11, "A", "B"), event(12, "A", "C"),
                  event(13, "B", "D", "2026-01-01T01:00:00"), event(14, "C", "D", "2026-01-01T01:00:00")]
        feed(state, branch)
        candidates, _ = search_patterns(state, branch[-2:], self.config)
        self.assertTrue(any(c["family"] == "scatter_gather" for c in candidates))

        state = RollingGraph(self.config)
        chain = [event(1, "A", "B"), event(2, "B", "C", "2026-01-01T01:00:00"), event(3, "C", "D", "2026-01-01T02:00:00")]
        feed(state, chain)
        candidates, _ = search_patterns(state, [chain[-1]], self.config)
        self.assertTrue(any(c["family"] == "chain" for c in candidates))

        state = RollingGraph(self.config)
        _, _, candidates, _ = process_group(state, [event(i + 5, "A" + str(i), "B") for i in range(3)], self.config)
        self.assertTrue(any(c["family"] == "fan_in" for c in candidates))

    def test_equal_time_cycle_is_marked(self):
        group = [event(1, "A", "B"), event(2, "B", "C"), event(3, "C", "A")]
        _, _, candidates, _ = process_group(self.state, group, self.config)
        cycles = [c for c in candidates if c["family"] == "cycle"]
        self.assertTrue(cycles)
        self.assertTrue(all(c["equal_timestamps"] for c in cycles))

    def test_search_works_without_models(self):
        group = [event(i, "A", "X" + str(i)) for i in range(4)]
        _, scores, candidates, _ = process_group(self.state, group, self.config)
        self.assertIsNone(scores)
        self.assertTrue(any(c["family"] == "fan_out" for c in candidates))

    def test_new_account_features(self):
        self.state.advance(event(1)["timestamp"])
        row = transaction_features(event(1), self.state)
        self.assertEqual(row["source_is_new"], 1)
        self.assertEqual(row["paid_ratio_missing"], 1)
        self.assertNotIn("history_age", row)

    def test_composite_account_key(self):
        self.assertNotEqual(account_key("1", "23"), account_key("12", "3"))
        self.assertNotEqual(account_key("1", "001"), account_key("1", "1"))


class ImportTests(unittest.TestCase):
    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(rows)

    def test_sorting_ids_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for minute in range(10):
                label = "1" if minute in (0, 5, 8) else "0"
                rows.append([f"2026/01/01 00:{minute:02d}", "001", "000A", "002", "000B", "10.00", "USD", "10.00", "USD", "Wire", label])
            rows.append(["2025/12/31 23:00", "001", "000A", "002", "000B", "20.00", "USD", "20.00", "USD", "Wire", "0"])
            self.write_csv(root / "input.csv", rows)

            manifest = prepare_data(root / "input.csv", root / "processed")
            events = load_table(root / "processed" / "events.parquet")
            self.assertEqual(events.timestamp.iloc[0], "2025-12-31T23:00:00")
            self.assertEqual(events.transaction_id.iloc[0], 11)
            self.assertNotIn("is_laundering", events.columns)
            self.assertEqual(events.source_key.iloc[0], "001:000A")
            self.assertEqual(manifest["rows"], 11)
            self.assertEqual(manifest["positive"], 3)

    def test_invalid_and_crypto_rows_are_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for minute in range(10):
                rows.append([f"2026/01/01 00:{minute:02d}", "1", "A", "2", "B", "10", "USD", "10", "USD", "Wire", "1" if minute % 3 == 0 else "0"])
            rows.append(["2026/01/01 01:00", "1", "A", "2", "B", "10", "Bitcoin", "10", "USD", "Wire", "1"])
            rows.append(["wrong-time", "1", "A", "2", "B", "-1", "USD", "-1", "USD", "Wire", "0"])
            self.write_csv(root / "input.csv", rows)

            manifest = prepare_data(root / "input.csv", root / "processed")
            self.assertEqual(manifest["excluded_crypto"], 1)
            self.assertEqual(manifest["invalid_rows"], 1)
            self.assertEqual(manifest["rows"], 10)


if __name__ == "__main__":
    unittest.main()
