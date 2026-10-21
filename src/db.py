"""Работа с PostgreSQL: поток операций, случаи, версии случаев и журнал решений.

Адрес базы берётся из переменной окружения HUNTERILKA_DB.
По умолчанию это база из docker-compose.yml.
"""

import os

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.data import account_key
from src.cases import clean_vector
from src.storage import ROOT


DEFAULT_URL = "postgresql://hunterilka:hunterilka@localhost:55432/hunterilka_v2"

TRANSACTION_COLUMNS = [
    "transaction_id", "ts", "source_bank", "source_account", "target_bank", "target_account",
    "amount_paid", "payment_currency", "amount_received", "receiving_currency", "payment_format",
]

CASE_COLUMNS = [
    "case_id", "candidate_id", "family", "detected_at", "first_detected_at", "first_time", "last_time",
    "operation_score", "anomaly_score", "operation_signal", "structure_signal",
    "operations", "accounts", "route", "version", "status", "model_id",
]


def connect(url=None):
    url = url or os.environ.get("HUNTERILKA_DB", DEFAULT_URL)
    return psycopg.connect(url, row_factory=dict_row, autocommit=True, connect_timeout=10, prepare_threshold=None)


def init_schema(conn):
    with conn.transaction():
        conn.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))


def reset(conn):
    with conn.transaction():
        conn.execute("TRUNCATE reviews, case_versions, cases, transactions, stream_state")


def time_text(value):
    # В коде время хранится строкой '2022-09-09T09:34:00', в базе — типом TIMESTAMP.
    if value is None:
        return None
    return value.strftime("%Y-%m-%dT%H:%M:%S")


# ---------- операции ----------

def event_from_row(row):
    """Строка таблицы transactions -> событие в том же виде, что и в events.parquet."""
    return {
        "transaction_id": int(row["transaction_id"]),
        "timestamp": time_text(row["ts"]),
        "source_bank": row["source_bank"], "source_account": row["source_account"],
        "target_bank": row["target_bank"], "target_account": row["target_account"],
        "amount_paid": str(row["amount_paid"]), "payment_currency": row["payment_currency"],
        "amount_received": str(row["amount_received"]), "receiving_currency": row["receiving_currency"],
        "payment_format": row["payment_format"],
        "source_key": account_key(row["source_bank"], row["source_account"]),
        "target_key": account_key(row["target_bank"], row["target_account"]),
    }


def insert_transactions(conn, events, scores=None, vectors=None, model_id=None):
    """Пакетная вставка. Все пакеты входят в общую транзакцию шага."""
    columns = TRANSACTION_COLUMNS + ["operation_score", "feature_vector", "model_id"]
    row_sql = "(" + ", ".join(["%s"] * len(columns)) + ")"
    with conn.transaction():
        for start in range(0, len(events), 1000):
            batch = events[start:start + 1000]
            values = []
            for offset, event in enumerate(batch):
                position = start + offset
                values.extend([
                    event["transaction_id"], event["timestamp"],
                    event["source_bank"], event["source_account"],
                    event["target_bank"], event["target_account"],
                    event["amount_paid"], event["payment_currency"],
                    event["amount_received"], event["receiving_currency"], event["payment_format"],
                    None if scores is None else float(scores[position]),
                    None if vectors is None else Jsonb(clean_vector(vectors[position])), model_id,
                ])
            sql = "INSERT INTO transactions (" + ", ".join(columns) + ") VALUES "
            conn.execute(sql + ", ".join([row_sql] * len(batch)), values)


def delete_old_transactions(conn, before):
    """Удаляет операции старше границы окна. Возвращает число удалённых строк."""
    with conn.transaction():
        return conn.execute("DELETE FROM transactions WHERE ts <= %s", (before,)).rowcount


def max_transaction_time(conn):
    row = conn.execute("SELECT max(ts) AS last FROM transactions").fetchone()
    return time_text(row["last"])


def count_transactions(conn):
    return conn.execute("SELECT count(*) AS n FROM transactions").fetchone()["n"]


def iterate_transactions(conn, start, end):
    """Операции в [start, end) по порядку времени. Читаются порциями, не всей таблицей."""
    sql = ("SELECT * FROM transactions WHERE ts >= %s AND ts < %s "
           "ORDER BY ts, transaction_id")
    with conn.transaction():
        with conn.cursor(name="transactions_stream") as cursor:
            cursor.itersize = 50000
            for row in cursor.execute(sql, (start, end)):
                yield event_from_row(row)


def transactions_per_day(conn, start, end):
    sql = ("SELECT ts::date AS day, count(*) AS operations FROM transactions "
           "WHERE ts >= %s AND ts < %s GROUP BY day ORDER BY day")
    return [{"day": row["day"].isoformat(), "operations": row["operations"]}
            for row in conn.execute(sql, (start, end)).fetchall()]


def search_accounts(conn, text, limit=50):
    """Счета, номер или банк которых содержит введённый текст."""
    like = "%" + text + "%"
    sql = ("SELECT DISTINCT bank, account FROM ("
           "  SELECT source_bank AS bank, source_account AS account FROM transactions"
           "  WHERE source_account ILIKE %s OR source_bank ILIKE %s"
           "  UNION"
           "  SELECT target_bank, target_account FROM transactions"
           "  WHERE target_account ILIKE %s OR target_bank ILIKE %s"
           ") AS found ORDER BY bank, account LIMIT %s")
    return conn.execute(sql, (like, like, like, like, limit)).fetchall()


def account_transactions(conn, bank, account, start, end, limit=32):
    """Последние операции одного счёта в периоде."""
    sql = ("SELECT * FROM transactions "
           "WHERE ((source_bank = %s AND source_account = %s) OR (target_bank = %s AND target_account = %s)) "
           "AND ts >= %s AND ts < %s ORDER BY ts DESC, transaction_id DESC LIMIT %s")
    rows = conn.execute(sql, (bank, account, bank, account, start, end, limit)).fetchall()
    return [event_from_row(row) for row in rows]


def transactions_between(conn, keys, start, end, limit=300):
    """Операции между счетами из списка keys (формат банк:номер) в периоде."""
    sql = ("SELECT * FROM transactions "
           "WHERE source_bank || ':' || source_account = ANY(%s) "
           "AND target_bank || ':' || target_account = ANY(%s) "
           "AND ts >= %s AND ts < %s ORDER BY ts DESC, transaction_id DESC LIMIT %s")
    rows = conn.execute(sql, (list(keys), list(keys), start, end, limit)).fetchall()
    return [event_from_row(row) for row in rows]


# ---------- случаи ----------

def case_from_row(row):
    case = dict(row)
    for key in ["detected_at", "first_detected_at", "first_time", "last_time"]:
        if key in case:
            case[key] = time_text(case[key])
    return case


def save_case(conn, case):
    with conn.transaction():
        return _save_case(conn, case)


def _save_case(conn, case):
    """Создаёт случай, обновляет его новой версией или оставляет как есть.

    Возвращает 'created', 'updated' или 'unchanged'.
    """
    previous = conn.execute(
        "SELECT candidate_id, operation_signal, structure_signal, version, detected_at, "
        "operation_score, anomaly_score, details, model_id FROM cases WHERE case_id = %s FOR UPDATE",
        (case["case_id"],),
    ).fetchone()

    if previous is None:
        conn.execute(
            "INSERT INTO cases (case_id, candidate_id, family, detected_at, first_detected_at, first_time, last_time, "
            "operation_score, anomaly_score, operation_signal, structure_signal, operations, accounts, route, "
            "model_id, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (case["case_id"], case["candidate_id"], case["family"], case["detected_at"], case["detected_at"],
             case["first_time"], case["last_time"], case["operation_score"], case["anomaly_score"],
             case["operation_signal"], case["structure_signal"], case["operations"], case["accounts"],
             case["route"], case["model_id"], Jsonb(case["details"])),
        )
        return "created"

    same_evidence = previous["candidate_id"] == case["candidate_id"]
    same_signals = (previous["operation_signal"] == case["operation_signal"]
                    and previous["structure_signal"] == case["structure_signal"])
    if same_evidence and same_signals and previous["model_id"] == case["model_id"]:
        return "unchanged"

    # Старое основание уходит в историю версий, статус и комментарии остаются у случая.
    conn.execute(
        "INSERT INTO case_versions (case_id, version, detected_at, operation_score, anomaly_score, details) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        (case["case_id"], previous["version"], previous["detected_at"], previous["operation_score"],
         previous["anomaly_score"], Jsonb(previous["details"])),
    )
    conn.execute(
        "UPDATE cases SET candidate_id = %s, detected_at = %s, first_time = %s, last_time = %s, "
        "operation_score = %s, anomaly_score = %s, operation_signal = %s, structure_signal = %s, "
        "operations = %s, accounts = %s, route = %s, model_id = %s, details = %s, version = version + 1 "
        "WHERE case_id = %s",
        (case["candidate_id"], case["detected_at"], case["first_time"], case["last_time"],
         case["operation_score"], case["anomaly_score"], case["operation_signal"], case["structure_signal"],
         case["operations"], case["accounts"], case["route"], case["model_id"], Jsonb(case["details"]),
         case["case_id"]),
    )
    return "updated"


def get_case(conn, case_id):
    row = conn.execute("SELECT * FROM cases WHERE case_id = %s", (case_id,)).fetchone()
    if row is None:
        raise ValueError("Случай не найден: " + case_id)
    case = case_from_row(row)
    details = case.pop("details")
    case.update(details)
    return case


def list_cases(conn, start, end, queue, family=None, status=None):
    """Очередь случаев за период: операции по оценке CatBoost, структуры по необычности."""
    sql = "SELECT " + ", ".join(CASE_COLUMNS) + " FROM cases WHERE detected_at >= %s AND detected_at < %s"
    params = [start, end]
    if queue == "operations":
        sql += " AND family = 'transaction'"
        order = " ORDER BY operation_score DESC NULLS LAST, case_id"
    else:
        sql += " AND family <> 'transaction'"
        order = " ORDER BY anomaly_score DESC NULLS LAST, case_id"
    if family:
        sql += " AND family = %s"
        params.append(family)
    if status:
        sql += " AND status = %s"
        params.append(status)
    rows = conn.execute(sql + order, params).fetchall()
    return [case_from_row(row) for row in rows]


def cases_per_day(conn, start, end):
    sql = ("SELECT detected_at::date AS day, "
           "count(*) FILTER (WHERE family = 'transaction') AS operation_cases, "
           "count(*) FILTER (WHERE family <> 'transaction') AS structure_cases "
           "FROM cases WHERE detected_at >= %s AND detected_at < %s GROUP BY day ORDER BY day")
    return [{"day": row["day"].isoformat(), "operation_cases": row["operation_cases"],
             "structure_cases": row["structure_cases"]}
            for row in conn.execute(sql, (start, end)).fetchall()]


def status_counts(conn, start, end):
    sql = "SELECT status, count(*) AS n FROM cases WHERE detected_at >= %s AND detected_at < %s GROUP BY status"
    return {row["status"]: row["n"] for row in conn.execute(sql, (start, end)).fetchall()}


def case_time_range(conn):
    row = conn.execute("SELECT min(detected_at) AS first, max(detected_at) AS last FROM cases").fetchone()
    return time_text(row["first"]), time_text(row["last"])


def count_cases(conn):
    return conn.execute("SELECT count(*) AS n FROM cases").fetchone()["n"]


def signalled_transaction_ids(conn):
    """Операции, которые входят в основание хотя бы одного случая."""
    sql = "SELECT DISTINCT (jsonb_array_elements(details -> 'transaction_ids'))::bigint AS transaction_id FROM cases"
    return {row["transaction_id"] for row in conn.execute(sql).fetchall()}


def case_versions(conn, case_id):
    sql = ("SELECT version, detected_at, operation_score, anomaly_score FROM case_versions "
           "WHERE case_id = %s ORDER BY version")
    return [case_from_row(row) for row in conn.execute(sql, (case_id,)).fetchall()]


# ---------- решения специалиста ----------

def add_review(conn, case_id, status, comment, analyst=""):
    """Запись в журнал и смена текущего статуса выполняются одной транзакцией."""
    with conn.transaction():
        conn.execute(
            "INSERT INTO reviews (case_id, status, comment, analyst) VALUES (%s, %s, %s, %s)",
            (case_id, status, comment, analyst),
        )
        conn.execute("UPDATE cases SET status = %s WHERE case_id = %s", (status, case_id))


# ---------- состояние источника и восстановление ----------

def acquire_writer(conn):
    """Один сервис записи на базу; рабочие места специалистов подключаются независимо."""
    row = conn.execute("SELECT pg_try_advisory_lock(7413021) AS acquired").fetchone()
    if not row["acquired"]:
        raise ValueError("Поток уже запущен другим процессом. Остановите его перед повторным запуском.")


def stream_state(conn):
    row = conn.execute("SELECT * FROM stream_state WHERE singleton = 1").fetchone()
    if row:
        row["last_time"] = time_text(row["last_time"])
        row["score_from"] = time_text(row["score_from"])
    return row


def ensure_stream(conn, dataset_id, model_id, contract, score_from, kind="replay"):
    previous = stream_state(conn)
    if previous:
        expected = (kind, dataset_id, model_id, contract, score_from)
        actual = tuple(previous[key] for key in ["kind", "dataset_id", "model_id", "feature_contract", "score_from"])
        if expected != actual:
            raise ValueError("В базе другой источник, модель, режим или начало потока. Для нового эксперимента нужна отдельная база или явный --reset.")
        return previous
    if count_transactions(conn) or count_cases(conn):
        raise ValueError("В базе есть данные без курсора v2.1. Используйте отдельную базу или явный --reset.")
    conn.execute(
        "INSERT INTO stream_state (singleton, kind, dataset_id, model_id, feature_contract, score_from) "
        "VALUES (1, %s, %s, %s, %s, %s)",
        (kind, dataset_id, model_id, Jsonb(contract), score_from),
    )
    return stream_state(conn)


def set_progress(conn, timestamp, history_complete=None):
    result = conn.execute(
        "UPDATE stream_state SET last_time = %s, history_complete = COALESCE(%s, history_complete) WHERE singleton = 1",
        (timestamp, history_complete),
    )
    if result.rowcount != 1:
        raise ValueError("Источник потока не зарегистрирован.")


def recent_scores(conn, start, end, model_id):
    sql = ("SELECT transaction_id, ts, operation_score, feature_vector FROM transactions "
           "WHERE ts >= %s AND ts <= %s AND operation_score IS NOT NULL AND model_id = %s "
           "ORDER BY ts, transaction_id")
    with conn.transaction():
        with conn.cursor(name="scores_stream") as cursor:
            cursor.itersize = 10000
            for row in cursor.execute(sql, (start, end, model_id)):
                yield row


def forget_old_scores(conn, before):
    conn.execute("UPDATE transactions SET operation_score = NULL, feature_vector = NULL, model_id = NULL "
                 "WHERE ts < %s AND operation_score IS NOT NULL", (before,))


def group_already_saved(conn, group):
    """Повтор последней полной минуты допускается только с тем же содержимым."""
    from decimal import Decimal
    rows = conn.execute("SELECT * FROM transactions WHERE ts = %s", (group[0]["timestamp"],)).fetchall()
    stored = {row["transaction_id"]: event_from_row(row) for row in rows}
    if len(stored) != len(group):
        return False
    for event in group:
        previous = stored.get(event["transaction_id"])
        if previous is None:
            return False
        for name in previous:
            if name in {"amount_paid", "amount_received"}:
                if Decimal(previous[name]) != Decimal(event[name]):
                    return False
            elif previous[name] != event[name]:
                return False
    return True


def case_reviews(conn, case_id):
    sql = "SELECT reviewed_at, status, comment, analyst FROM reviews WHERE case_id = %s ORDER BY reviewed_at, review_id"
    rows = conn.execute(sql, (case_id,)).fetchall()
    for row in rows:
        row["reviewed_at"] = time_text(row["reviewed_at"])
    return rows
