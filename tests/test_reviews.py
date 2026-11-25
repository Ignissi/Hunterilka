"""Журнал решений на настоящей PostgreSQL, включая повторы из двух сеансов."""

import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.testing.v1 import AppTest

from scripts.export_demo import export_to
from scripts.load_demo import load_from
from src import db
from src.cases import build_case
from src.patterns import make_candidate
from src.storage import ROOT, load_config
from tests.helpers import TEST_URL, event, fake_models, start_test_stream, test_connection


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.conn = test_connection()
        if self.conn is None:
            self.skipTest("Для проверки журнала нужна тестовая PostgreSQL")
        self.addCleanup(self.conn.close)
        config = load_config()
        start_test_stream(self.conn, config, fake_models(config))
        self.case_id = self.create_case(1)

    def create_case(self, number):
        candidate = make_candidate("transaction", [event(number)], [number], "2026-01-01T00:00:00")
        case = build_case(candidate, {number: 1}, {}, {
            "transaction_threshold": 0.5, "structure_threshold": 1, "model_id": "test-model",
        })
        db.save_case(self.conn, case)
        return case["case_id"]

    def save(self, status="Рассмотрен", comment="Данные сверены", analyst="Анжелика"):
        return db.add_review(self.conn, self.case_id, status, comment, analyst)

    def status(self):
        return db.get_case(self.conn, self.case_id)["status"]

    def test_required_fields_and_status_leave_journal_unchanged(self):
        for args in [("Рассмотрен", "", "Анжелика"), ("Рассмотрен", " \n\t", "Анжелика"),
                     ("Рассмотрен", "Проверено", " "), ("Рассмотрен", "Проверено", "1"),
                     ("?", "Проверено", "Анжелика")]:
            with self.subTest(fields=args), self.assertRaises(ValueError):
                self.save(*args)
        self.assertEqual(db.case_reviews(self.conn, self.case_id), [])
        self.assertEqual(self.status(), "Новый")

    def test_only_consecutive_equal_decisions_are_duplicates(self):
        self.assertTrue(self.save(comment="  Данные сверены\n", analyst=" Анжелика "))
        self.assertFalse(self.save())
        self.assertTrue(self.save(comment="Уточнено"))
        self.assertTrue(self.save(analyst="А. Михеева"))
        self.assertTrue(self.save(status="Новый", analyst="А. Михеева"))
        self.assertTrue(self.save())
        rows = db.case_reviews(self.conn, self.case_id)
        self.assertEqual(len(rows), 5)
        self.assertEqual((rows[0]["comment"], rows[0]["analyst"]), ("Данные сверены", "Анжелика"))
        self.assertEqual(self.status(), "Рассмотрен")

    def test_simultaneous_saves_create_one_entry(self):
        ready = Barrier(2, timeout=10)

        def submit():
            with db.connect(TEST_URL) as writer:
                writer.execute("SET statement_timeout = '10s'")
                ready.wait()
                return db.add_review(writer, self.case_id, "Рассмотрен", "Проверено", "Сотрудник")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(submit) for _ in range(2)]
            self.assertEqual(sorted(f.result(timeout=15) for f in futures), [False, True])
        self.assertEqual(len(db.case_reviews(self.conn, self.case_id)), 1)

    def test_withdraw_selected_entry_and_restore_last_active_status(self):
        for state in db.REVIEW_STATUSES:
            self.save(status=state)
        first, middle, last = db.case_reviews(self.conn, self.case_id)
        self.assertFalse(db.withdraw_review(self.conn, self.create_case(2), last["review_id"]))
        self.assertTrue(db.withdraw_review(self.conn, self.case_id, first["review_id"]))
        self.assertEqual(self.status(), "Рассмотрен")
        self.assertTrue(db.withdraw_review(self.conn, self.case_id, last["review_id"]))
        self.assertEqual(self.status(), "Требует проверки")
        self.assertFalse(db.withdraw_review(self.conn, self.case_id, last["review_id"]))
        self.assertTrue(db.withdraw_review(self.conn, self.case_id, middle["review_id"]))
        self.assertEqual(self.status(), "Новый")
        self.assertEqual(db.case_reviews(self.conn, self.case_id), [])
        audit = db.case_reviews(self.conn, self.case_id, include_withdrawn=True)
        self.assertEqual(len(audit), 3)
        self.assertTrue(all(row["withdrawn_at"] for row in audit))
        self.assertTrue(self.save())

    def test_legacy_duplicate_and_empty_rows_can_be_withdrawn(self):
        for comment in ("", "дело 1.", "дело 1."):
            self.conn.execute("INSERT INTO reviews(case_id, status, comment, analyst) VALUES (%s, 'Новый', %s, '1')",
                              (self.case_id, comment))
        old = db.case_reviews(self.conn, self.case_id)
        db.withdraw_review(self.conn, self.case_id, old[2]["review_id"])
        db.withdraw_review(self.conn, self.case_id, old[0]["review_id"])
        self.assertEqual(db.case_reviews(self.conn, self.case_id), [old[1]])

    def test_withdrawal_and_case_status_roll_back_together(self):
        self.save()
        before = db.case_reviews(self.conn, self.case_id)
        with self.assertRaises(RuntimeError), self.conn.transaction():
            db.withdraw_review(self.conn, self.case_id, before[0]["review_id"])
            raise RuntimeError("Сбой до фиксации")
        self.assertEqual(db.case_reviews(self.conn, self.case_id), before)
        self.assertEqual(self.status(), "Рассмотрен")

    def test_demo_roundtrip_keeps_withdrawals(self):
        self.save(status="Требует проверки")
        self.save()
        db.withdraw_review(self.conn, self.case_id, db.case_reviews(self.conn, self.case_id)[-1]["review_id"])
        with tempfile.TemporaryDirectory() as folder:
            export_to(self.conn, Path(folder))
            load_from(self.conn, Path(folder), replace=True)
        self.assertEqual(self.status(), "Требует проверки")
        audit = db.case_reviews(self.conn, self.case_id, include_withdrawn=True)
        self.assertEqual([r["withdrawn_at"] is not None for r in audit], [False, True])
        self.assertEqual(len(db.case_reviews(self.conn, self.case_id)), 1)

    def test_review_edits_never_scan_all_transfers_in_the_fragment_or_request_a_full_rerun(self):
        scopes = []

        def traced(query):
            def read(*args, **kwargs):
                scopes.append((query.__name__, get_script_run_ctx().current_fragment_id))
                return query(*args, **kwargs)
            return read

        with patch.dict(os.environ, {"HUNTERILKA_DB": TEST_URL}), \
                patch.object(db, "count_transactions", traced(db.count_transactions)), \
                patch.object(db, "transactions_per_day", traced(db.transactions_per_day)), \
                patch.object(db, "status_counts", traced(db.status_counts)), \
                patch.object(st, "rerun", wraps=st.rerun) as rerun:
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.radio(key="queue_channel").set_value("Операции").run()

            def button(label):
                return next(item for item in app.button if item.label == label)

            app.text_area(key="review_comment_" + self.case_id).set_value("Проверка быстрого сохранения")
            app.text_input(key="review_analyst_" + self.case_id).set_value("Спец1")
            button("Сохранить решение").click().run()
            review_id = db.case_reviews(self.conn, self.case_id)[0]["review_id"]
            app.selectbox(key="withdraw_select_" + self.case_id).select(review_id).run()
            button("Подтвердить удаление").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(db.case_reviews(self.conn, self.case_id), [])
            rerun.assert_not_called()
        for name in ("count_transactions", "transactions_per_day", "status_counts"):
            ids = [fragment for query, fragment in scopes if query == name]
            self.assertTrue(ids, name)
            self.assertTrue(all(bool(fragment) == (name == "status_counts") for fragment in ids), name)

    def test_review_form_checks_input_resubmission_and_removal(self):
        with patch.dict(os.environ, {"HUNTERILKA_DB": TEST_URL}):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.radio(key="queue_channel").set_value("Операции").run()

            def widget(elements, label):
                return next(e for e in elements if e.label == label)

            def submit():
                widget(app.button, "Сохранить решение").click().run()
                self.assertFalse(app.exception)

            submit()
            self.assertIn("Введите комментарий", app.error[0].value)
            widget(app.text_area, "Комментарий").set_value("Проверено")
            widget(app.text_input, "Специалист").set_value("Анжелика")
            submit()
            self.assertEqual(len(db.case_reviews(self.conn, self.case_id)), 1)
            submit()
            self.assertTrue(any("уже сохранено" in msg.value for msg in app.info))
            rid = db.case_reviews(self.conn, self.case_id)[0]["review_id"]
            self.assertTrue(widget(app.button, "Подтвердить удаление").disabled)
            widget(app.selectbox, "Запись журнала").select(rid).run()
            widget(app.button, "Подтвердить удаление").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(db.case_reviews(self.conn, self.case_id), [])
            widget(app.checkbox, "Показать отменённые решения").check().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Отменено" in frame.value.columns for frame in app.dataframe))
