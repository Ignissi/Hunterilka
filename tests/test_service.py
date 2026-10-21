"""Сквозная проверка сервиса: прогрев, оценка минуты, случаи и окно в базе."""

import unittest

from src import db
from src.service import Service, minute_groups
from src.storage import load_config
from tests.helpers import event, fake_models, start_test_stream, test_connection


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.conn = test_connection()
        if self.conn is None:
            self.skipTest("Тестовая база недоступна: задайте HUNTERILKA_TEST_DB")
        self.config = load_config()
        self.config["window_days"] = 1
        self.service = Service(self.config, fake_models(self.config), self.conn)
        start_test_stream(self.conn, self.config, self.service.models)

    def tearDown(self):
        if self.conn is not None:
            self.conn.close()

    def test_minute_groups(self):
        events = [event(1), event(2), event(3, timestamp="2026-01-01T00:01:00")]
        groups = [(timestamp, len(group)) for timestamp, group in minute_groups(events)]
        self.assertEqual(groups, [("2026-01-01T00:00:00", 2), ("2026-01-01T00:01:00", 1)])

    def test_stream_creates_cases_and_cleans_window(self):
        history = [event(1, "H", "A", "2026-01-01T00:00:00"), event(2, "H", "A", "2026-01-01T06:00:00")]
        db.insert_transactions(self.conn, history)
        self.service.warm_up(history)
        self.assertEqual(len(self.service.state.events), 2)

        # Три крупных перевода с одного счёта: сигнал операций и структура "распределение".
        group = [event(10 + i, "A", "T" + str(i), "2026-01-02T12:00:00", amount="50000.00") for i in range(3)]
        info = self.service.process(group)
        self.assertGreaterEqual(info["saved_cases"], 4)
        self.assertEqual(db.count_transactions(self.conn), 3)   # история старше суток удалена из базы
        self.assertEqual(len(self.service.state.events), 3)     # и из графа тоже

        operations = db.list_cases(self.conn, "2026-01-02T00:00:00", "2026-01-03T00:00:00", "operations")
        structures = db.list_cases(self.conn, "2026-01-02T00:00:00", "2026-01-03T00:00:00", "structures")
        self.assertEqual(len(operations), 3)
        self.assertTrue(any(row["family"] == "fan_out" for row in structures))
        case = db.get_case(self.conn, operations[0]["case_id"])
        self.assertTrue(case["operation_signal"])
        self.assertEqual(case["operation_score"], 1.0)
        self.assertEqual(list(case["features"]), [str(case["transaction_ids"][0])])

        # Структура растёт в том же часе: случай получает новую версию, а не дубликат.
        more = [event(20, "A", "T9", "2026-01-02T12:30:00", amount="50000.00")]
        self.service.process(more)
        fan_out = [row for row in db.list_cases(self.conn, "2026-01-02T00:00:00", "2026-01-03T00:00:00", "structures")
                   if row["family"] == "fan_out"]
        self.assertEqual(len(fan_out), 1)
        self.assertEqual(fan_out[0]["version"], 2)
        self.assertEqual(fan_out[0]["operations"], 4)

    def test_small_transfer_makes_no_case(self):
        group = [event(1, "A", "B", "2026-01-02T12:00:00", amount="5.00")]
        info = self.service.process(group)
        self.assertEqual(info["saved_cases"], 0)
        self.assertEqual(db.count_cases(self.conn), 0)
        self.assertEqual(db.count_transactions(self.conn), 1)


if __name__ == "__main__":
    unittest.main()
