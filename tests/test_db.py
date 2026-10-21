"""Проверки слоя PostgreSQL. Пропускаются, если тестовая база недоступна."""

import unittest

from src import db
from src.cases import build_case
from src.patterns import make_candidate
from tests.helpers import event, test_connection


META = {"transaction_threshold": 0.5, "structure_threshold": 0.5, "model_id": "test-model"}


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = test_connection()
        if self.conn is None:
            self.skipTest("Тестовая база недоступна: задайте HUNTERILKA_TEST_DB")

    def tearDown(self):
        if self.conn is not None:
            self.conn.close()

    def test_transactions_roundtrip_and_window(self):
        events = [event(1), event(2, "B", "C", "2026-01-02T00:00:00", amount="20.50"), event(3, "C", "D", "2026-01-03T00:00:00")]
        db.insert_transactions(self.conn, events)
        self.assertEqual(db.count_transactions(self.conn), 3)
        self.assertEqual(db.max_transaction_time(self.conn), "2026-01-03T00:00:00")

        loaded = list(db.iterate_transactions(self.conn, "2026-01-01T00:00:00", "2026-01-03T00:00:00"))
        self.assertEqual([e["transaction_id"] for e in loaded], [1, 2])
        self.assertEqual(loaded[1]["amount_paid"], "20.50")
        self.assertEqual(loaded[1]["source_key"], "01:B")
        self.assertEqual(loaded[1]["timestamp"], "2026-01-02T00:00:00")

        self.assertEqual(db.delete_old_transactions(self.conn, "2026-01-01T00:00:00"), 1)
        self.assertEqual(db.count_transactions(self.conn), 2)

        # У счёта C две операции: входящая 2 и исходящая 3; свежие идут первыми.
        around = db.account_transactions(self.conn, "01", "C", "2026-01-01T00:00:00", "2026-01-04T00:00:00")
        self.assertEqual([e["transaction_id"] for e in around], [3, 2])
        found = db.search_accounts(self.conn, "c")
        self.assertIn(("01", "C"), [(row["bank"], row["account"]) for row in found])

    def test_case_versions_and_reviews(self):
        candidate = make_candidate("fan_out", [event(1), event(2, "A", "C")], ["01:A"], "2026-01-01T00:00:00")
        candidate["anomaly_score"] = 0.9
        case = build_case(candidate, {1: 0.2, 2: 0.3}, {1: {"x": float("nan")}, 2: {"x": 1.0}}, META)
        self.assertEqual(db.save_case(self.conn, case), "created")
        self.assertEqual(db.save_case(self.conn, case), "unchanged")

        stored = db.get_case(self.conn, case["case_id"])
        self.assertEqual(stored["version"], 1)
        self.assertIsNone(stored["features"]["1"]["x"])
        self.assertEqual(stored["events"][0]["amount_paid"], "10.00")
        self.assertTrue(stored["structure_signal"])
        self.assertFalse(stored["operation_signal"])

        db.add_review(self.conn, case["case_id"], "Требует проверки", "Не потерять", "Анжелика")
        grown = make_candidate("fan_out", [event(1), event(2, "A", "C"), event(3, "A", "D")], ["01:A"], "2026-01-01T00:30:00")
        grown["anomaly_score"] = 0.95
        self.assertEqual(grown["case_key"], candidate["case_key"])
        updated = build_case(grown, {}, {}, META)
        self.assertEqual(db.save_case(self.conn, updated), "updated")

        stored = db.get_case(self.conn, case["case_id"])
        self.assertEqual(stored["version"], 2)
        self.assertEqual(stored["status"], "Требует проверки")
        self.assertEqual(stored["first_detected_at"], "2026-01-01T00:00:00")
        self.assertEqual(len(stored["events"]), 3)
        self.assertEqual(len(db.case_versions(self.conn, case["case_id"])), 1)
        self.assertEqual(db.case_reviews(self.conn, case["case_id"])[0]["comment"], "Не потерять")

        queue = db.list_cases(self.conn, "2026-01-01T00:00:00", "2026-01-02T00:00:00", "structures")
        self.assertEqual([row["case_id"] for row in queue], [case["case_id"]])
        self.assertEqual(db.list_cases(self.conn, "2026-01-01T00:00:00", "2026-01-02T00:00:00", "operations"), [])
        self.assertEqual(db.status_counts(self.conn, "2026-01-01T00:00:00", "2026-01-02T00:00:00"), {"Требует проверки": 1})
        self.assertEqual(db.signalled_transaction_ids(self.conn), {1, 2, 3})

    def test_case_without_signal_is_not_built(self):
        candidate = make_candidate("transaction", [event(1)], [1], "2026-01-01T00:00:00")
        self.assertIsNone(build_case(candidate, {1: 0.1}, {1: {}}, META))
        self.assertIsNotNone(build_case(candidate, {1: 0.9}, {1: {}}, META))


if __name__ == "__main__":
    unittest.main()
