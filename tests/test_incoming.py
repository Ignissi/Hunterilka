"""Вход новых операций, подтверждение файлов и продолжение после остановки."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import db
from src.contracts import feature_contract
from src.incoming import FIELDS, consume_batch, load_batch, minute_time, normalize_event
from src.service import Service
from src.storage import load_config
from tests.helpers import event, fake_models, test_connection


def incoming(number=1, timestamp="2026-10-21T12:07:00"):
    return {key: value for key, value in event(number, timestamp=timestamp, amount="50000").items()
            if key in FIELDS}


def batch(folder, number=1, timestamp="2026-10-21T12:07:00"):
    path = Path(folder) / (timestamp.replace("-", "").replace(":", "") + ".json")
    path.write_text(json.dumps({"events": [incoming(number, timestamp)]}), encoding="utf-8")
    return path


class IncomingFormatTests(unittest.TestCase):
    def test_new_dates_without_labels_or_training_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            group = load_batch(batch(directory))
        self.assertEqual(group[0]["timestamp"], "2026-10-21T12:07:00")
        self.assertEqual(group[0]["source_key"], "01:A")
        self.assertNotIn("is_laundering", group[0])

    def test_timezone_normalizes_to_utc(self):
        self.assertEqual(minute_time("2026-10-21T14:07:00+02:00"), "2026-10-21T12:07:00")

    def test_rejects_labels_and_invalid_ids(self):
        for row in [dict(incoming(), is_laundering=1), dict(incoming(), transaction_id=True),
                    dict(incoming(), transaction_id=-1), dict(incoming(), transaction_id=2**63)]:
            with self.assertRaises(ValueError):
                normalize_event(row)

    def test_rejects_invalid_amounts(self):
        for value in ["NaN", "Infinity", "-1", "1e-999999999", "1e100", 3.14]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_event(dict(incoming(), amount_paid=value))

    def test_rejects_partial_minute_and_ambiguous_key(self):
        for row in [dict(incoming(), timestamp="2026-10-21T12:07:12"),
                    dict(incoming(), source_account="X:Y")]:
            with self.assertRaises(ValueError):
                normalize_event(row)

    def test_rejects_mixed_minutes_duplicate_ids_and_wrong_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            path = batch(directory)
            for rows in [[incoming(), incoming(2, "2026-10-21T12:08:00")],
                         [incoming(), incoming()]]:
                path.write_text(json.dumps({"events": rows}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_batch(path)
            path = batch(directory)
            wrong = path.rename(path.with_name("wrong.json"))
            with self.assertRaises(ValueError):
                load_batch(wrong)


class IncomingDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.conn = test_connection()
        if self.conn is None:
            self.skipTest("Нужна HUNTERILKA_TEST_DB")
        self.config = load_config()
        self.models = fake_models(self.config)
        db.ensure_stream(self.conn, "incoming:test", self.models["meta"]["model_id"],
                         feature_contract(self.config), "2026-10-21T12:07:00", kind="inbox")
        self.service = Service(self.config, self.models, self.conn)
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        if self.conn:
            self.conn.close()
        if hasattr(self, "directory"):
            self.directory.cleanup()

    def test_new_feed_scores_and_archives_without_dataset_split(self):
        path = batch(self.directory.name)
        consume_batch(path, self.conn, self.service)
        self.assertEqual(db.count_transactions(self.conn), 1)
        self.assertEqual(db.count_cases(self.conn), 1)
        self.assertFalse(path.exists())
        self.assertTrue((path.parent / "processed" / path.name).exists())
        self.assertEqual(db.stream_state(self.conn)["last_time"], "2026-10-21T12:07:00")

    def test_failure_keeps_file_and_database_cursor(self):
        path = batch(self.directory.name)
        with patch.object(self.service, "clean_window", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                consume_batch(path, self.conn, self.service)
        self.assertTrue(path.exists())
        self.assertEqual(db.count_transactions(self.conn), 0)
        self.assertIsNone(db.stream_state(self.conn)["last_time"])
        consume_batch(path, self.conn, self.service)
        self.assertEqual(db.count_transactions(self.conn), 1)

    def test_committed_batch_before_file_ack_can_be_retried(self):
        path = batch(self.directory.name)
        with patch.object(Path, "rename", side_effect=OSError("file ack interrupted")):
            with self.assertRaises(OSError):
                consume_batch(path, self.conn, self.service)
        restarted = Service(self.config, self.models, self.conn)
        restarted.restore()
        result = consume_batch(path, self.conn, restarted)
        self.assertTrue(result["skipped_group"])
        self.assertEqual(db.count_transactions(self.conn), 1)
        self.assertEqual(db.count_cases(self.conn), 1)

    def test_history_then_scoring_and_restart(self):
        history = batch(self.directory.name, 1, "2026-10-21T12:06:00")
        consume_batch(history, self.conn, self.service)
        self.assertEqual(db.count_cases(self.conn), 0)
        restarted = Service(self.config, self.models, self.conn)
        restarted.restore()
        consume_batch(batch(self.directory.name, 2), self.conn, restarted)
        self.assertEqual(db.count_transactions(self.conn), 2)
        self.assertTrue(db.stream_state(self.conn)["history_complete"])

    def test_old_schema_migration_preserves_progress(self):
        self.conn.execute("UPDATE stream_state SET kind = 'replay'")
        self.conn.execute("ALTER TABLE stream_state DROP CONSTRAINT stream_state_kind_v22_check")
        self.conn.execute("ALTER TABLE stream_state ADD CONSTRAINT stream_state_kind_check CHECK (kind IN ('replay', 'demo'))")
        db.init_schema(self.conn)
        self.assertEqual(db.stream_state(self.conn)["dataset_id"], "incoming:test")
        self.conn.execute("UPDATE stream_state SET kind = 'inbox'")


if __name__ == "__main__":
    unittest.main()
