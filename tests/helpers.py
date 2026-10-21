"""Общие заготовки для тестов: события, тестовая база, простые модели вместо CatBoost."""

import os

import numpy as np
import psycopg

from src import db
from src.contracts import feature_contract
from src.data import account_key
from src.features import transaction_features
from src.graph import RollingGraph
from src.patterns import make_candidate, structure_features


TEST_URL = os.environ.get("HUNTERILKA_TEST_DB", "postgresql://hunterilka:hunterilka@localhost:55432/hunterilka_v2_test")


def event(number, source="A", target="B", timestamp="2026-01-01T00:00:00", currency="USD", amount="10.00"):
    return {
        "transaction_id": number, "timestamp": timestamp,
        "source_bank": "01", "source_account": source, "target_bank": "01", "target_account": target,
        "source_key": account_key("01", source), "target_key": account_key("01", target),
        "amount_paid": amount, "amount_received": amount,
        "payment_currency": currency, "receiving_currency": currency, "payment_format": "Wire",
    }


def test_connection():
    """Соединение с чистой тестовой базой или None, если база недоступна."""
    try:
        conn = db.connect(TEST_URL)
    except psycopg.OperationalError:
        if os.environ.get("HUNTERILKA_REQUIRE_DB") == "1":
            raise
        return None
    db.init_schema(conn)
    db.reset(conn)
    return conn


def start_test_stream(conn, config, models, ready=True):
    db.ensure_stream(conn, "fixture", models["meta"]["model_id"], feature_contract(config), "2026-01-01T00:00:00")
    db.set_progress(conn, None, history_complete=ready)


class AmountModel:
    """Заменяет CatBoost в тестах: крупная сумма даёт высокую оценку."""

    def predict_proba(self, table):
        positive = (table["log_amount_paid"] > 9).astype(float).to_numpy()
        return np.column_stack([1 - positive, positive])


class SizeModel:
    """Заменяет Isolation Forest: чем больше переводов в структуре, тем она необычнее."""

    def score_samples(self, table):
        return -table["log_events"].to_numpy()


def fake_models(config):
    state = RollingGraph(config)
    state.advance("2026-01-01T00:00:00")
    feature_names = list(transaction_features(event(1), state))
    candidate = make_candidate("fan_out", [event(1), event(2, "A", "C")], ["A"], "2026-01-01T00:00:00")
    structure_names = list(structure_features(candidate))
    return {
        "transaction": AmountModel(), "structure": SizeModel(),
        "meta": {
            "features": feature_names, "structure_features": structure_names,
            "transaction_threshold": 0.5, "structure_threshold": 1.0, "model_id": "test-model",
        },
    }
