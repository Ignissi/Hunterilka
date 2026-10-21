"""Регрессии: сбой, повтор минуты, продолжение прогрева и восстановление сигналов."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import db
from src.contracts import feature_contract
from src.service import Service
from src.storage import load_config
from scripts.run_stream import load_history
from scripts.export_demo import export_to
from scripts.load_demo import load_from
from tests.helpers import event, fake_models, start_test_stream, test_connection


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.conn = test_connection()
        if self.conn is None:
            self.skipTest("Для интеграционного теста нужна HUNTERILKA_TEST_DB")
        self.config = load_config()
        self.models = fake_models(self.config)
        self.models["meta"]["structure_threshold"] = 10.0
        start_test_stream(self.conn, self.config, self.models)
        self.service = Service(self.config, self.models, self.conn)
        self.first = [event(1, "A", "B", "2026-01-01T12:00:00", amount="50000"),
                      event(2, "A", "C", "2026-01-01T12:00:00", amount="5")]
        self.second = [event(3, "A", "D", "2026-01-01T12:01:00", amount="5")]

    def tearDown(self):
        if self.conn:
            self.conn.close()

    def snapshot(self):
        rows = db.list_cases(self.conn, "2026-01-01", "2026-01-03", "structures")
        return [db.get_case(self.conn, row["case_id"]) for row in rows]

    def test_restart_keeps_same_cases_scores_and_vectors(self):
        self.service.process(self.first)
        self.service.process(self.second)
        expected = self.snapshot()
        self.assertTrue(expected)
        db.reset(self.conn)
        start_test_stream(self.conn, self.config, self.models)
        self.service = Service(self.config, self.models, self.conn)
        self.service.process(self.first)
        restarted = Service(self.config, self.models, self.conn)
        restarted.restore()
        self.assertEqual(set(restarted.scores), {1, 2})
        restarted.process(self.second)
        self.assertEqual(self.snapshot(), expected)

    def test_failure_after_insert_rolls_back_entire_minute(self):
        with patch.object(self.service, "clean_window", side_effect=RuntimeError("после INSERT")):
            with self.assertRaises(RuntimeError):
                self.service.process(self.first)
        self.assertEqual(db.count_transactions(self.conn), 0)
        self.assertEqual(db.count_cases(self.conn), 0)
        self.assertIsNone(db.stream_state(self.conn)["last_time"])
        self.service.process(self.first)
        self.assertEqual(db.count_transactions(self.conn), 2)
        self.assertEqual(db.count_cases(self.conn), 1)

    def test_failure_after_case_write_rolls_back_case_and_cursor(self):
        original = db.save_case

        def fail_after_write(conn, case):
            original(conn, case)
            raise RuntimeError("после записи случая")

        with patch.object(db, "save_case", side_effect=fail_after_write):
            with self.assertRaises(RuntimeError):
                self.service.process(self.first)
        self.assertEqual(db.count_transactions(self.conn), 0)
        self.assertEqual(db.count_cases(self.conn), 0)
        self.assertIsNone(db.stream_state(self.conn)["last_time"])

    def test_retry_committed_group_does_not_duplicate(self):
        self.service.process(self.first)
        self.assertTrue(self.service.process(self.first)["skipped_group"])
        self.assertEqual(db.count_transactions(self.conn), 2)
        self.assertEqual(db.count_cases(self.conn), 1)
        altered = [dict(self.first[0], amount_paid="1"), self.first[1]]
        with self.assertRaises(ValueError):
            self.service.process(altered)

    def test_new_model_or_settings_cannot_resume_old_source(self):
        self.service.process(self.first)
        with self.assertRaises(ValueError):
            db.ensure_stream(self.conn, "fixture", "another-model", feature_contract(self.config), "2026-01-01T00:00:00")
        config = dict(self.config, window_days=10)
        with self.assertRaises(ValueError):
            db.ensure_stream(self.conn, "fixture", self.models["meta"]["model_id"], feature_contract(config), "2026-01-01T00:00:00")

    def test_partial_warmup_resumes_complete_minutes(self):
        db.set_progress(self.conn, None, history_complete=False)
        history = [event(i, timestamp="2026-01-01T00:00:00") for i in range(1, 4)]
        history += [event(4, timestamp="2026-01-01T00:01:00")]
        self.service.ingest_history(history[:3])
        restarted = Service(self.config, self.models, self.conn)
        restarted.restore()
        remaining = pd.DataFrame(history[3:])
        with patch("scripts.run_stream.HISTORY_BATCH", 2):
            load_history(self.conn, restarted, remaining)
        self.assertEqual(db.count_transactions(self.conn), 4)
        self.assertEqual(set(restarted.state.events), {1, 2, 3, 4})
        self.assertTrue(db.stream_state(self.conn)["history_complete"])

    def test_large_minute_is_not_split_by_history_batch(self):
        db.set_progress(self.conn, None, history_complete=False)
        history = pd.DataFrame([event(i) for i in range(1, 6)])
        with patch("scripts.run_stream.HISTORY_BATCH", 2):
            load_history(self.conn, self.service, history)
        self.assertEqual(db.count_transactions(self.conn), 5)
        self.assertTrue(db.stream_state(self.conn)["history_complete"])

    def test_demo_preserves_versions_reviews_and_dates(self):
        self.models["meta"]["structure_threshold"] = 1.0
        group = [event(i, "A", "B" + str(i), "2026-01-01T12:00:00", amount="50000") for i in range(1, 4)]
        self.service.process(group)
        identifier = self.snapshot()[0]["case_id"]
        db.add_review(self.conn, identifier, "Требует проверки", "Сохранить основание", "Analyst")
        self.service.process([event(4, "A", "B4", "2026-01-01T12:30:00", amount="50000")])
        before = db.get_case(self.conn, identifier)
        reviews = self.conn.execute("SELECT case_id, reviewed_at, status, comment, analyst FROM reviews ORDER BY review_id").fetchall()
        with tempfile.TemporaryDirectory() as folder:
            export_to(self.conn, Path(folder))
            with self.assertRaises(ValueError):
                load_from(self.conn, Path(folder))
            load_from(self.conn, Path(folder), replace=True)
        self.assertEqual(db.get_case(self.conn, identifier), before)
        self.assertEqual(self.conn.execute("SELECT case_id, reviewed_at, status, comment, analyst FROM reviews ORDER BY review_id").fetchall(), reviews)
        self.assertEqual(len(db.case_versions(self.conn, identifier)), 1)
        self.assertEqual(db.stream_state(self.conn)["kind"], "demo")
        with self.assertRaises(ValueError):
            self.service.process([event(5, timestamp="2026-01-01T12:31:00")])

    def test_missing_demo_does_not_clear_database(self):
        self.service.process(self.first)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_from(self.conn, Path(folder), replace=True)
        self.assertEqual(db.count_cases(self.conn), 1)
        self.assertEqual(db.count_transactions(self.conn), 2)


if __name__ == "__main__":
    unittest.main()
