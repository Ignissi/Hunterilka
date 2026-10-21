"""Вход новых операций: законченная минута в JSON, без обучающих меток."""

import json
import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.data import account_key


TEXT_FIELDS = ("source_bank", "source_account", "target_bank", "target_account",
               "payment_currency", "receiving_currency", "payment_format")
AMOUNT_FIELDS = ("amount_paid", "amount_received")
FIELDS = set(TEXT_FIELDS + AMOUNT_FIELDS + ("transaction_id", "timestamp"))
MAX_BATCH_BYTES = 64 * 1024 * 1024


def minute_time(value):
    if not isinstance(value, str):
        raise ValueError("timestamp должен быть строкой ISO 8601.")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Некорректное время операции.") from error
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc).replace(tzinfo=None)
    if stamp.second or stamp.microsecond:
        raise ValueError("Вход сгруппирован по минутам: секунды должны быть равны нулю.")
    return stamp.strftime("%Y-%m-%dT%H:%M:%S")


def normalize_event(raw):
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise ValueError("Ожидаются поля входного контракта без меток, split и дополнительных полей.")
    event = dict(raw)
    identifier = event["transaction_id"]
    if type(identifier) is not int or not 0 <= identifier <= 2**63 - 1:
        raise ValueError("transaction_id должен быть неотрицательным целым BIGINT.")
    event["timestamp"] = minute_time(event["timestamp"])
    for name in TEXT_FIELDS:
        if not isinstance(event[name], str) or not event[name].strip():
            raise ValueError(name + ": нужна непустая строка.")
        event[name] = event[name].strip()
        if name.endswith(("_bank", "_account")) and ":" in event[name]:
            raise ValueError(name + ": двоеточие зарезервировано для составного ключа.")
    for name in AMOUNT_FIELDS:
        if not isinstance(event[name], str):
            raise ValueError(name + ": сумма передаётся десятичной строкой.")
        try:
            amount = Decimal(event[name])
        except InvalidOperation as error:
            raise ValueError(name + ": некорректная сумма.") from error
        if not amount.is_finite() or amount < 0 or not math.isfinite(float(amount)):
            raise ValueError(name + ": нужна конечная неотрицательная сумма.")
        if amount.as_tuple().exponent < -18 or amount.adjusted() > 29:
            raise ValueError(name + ": допустимо до 30 целых и 18 дробных знаков.")
        event[name] = format(amount, "f")
    event["source_key"] = account_key(event["source_bank"], event["source_account"])
    event["target_key"] = account_key(event["target_bank"], event["target_account"])
    return event


def load_batch(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_BATCH_BYTES:
        raise ValueError("Ожидается обычный JSON-файл размером не более 64 MiB.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"events"} or not isinstance(raw["events"], list) or not raw["events"]:
        raise ValueError("Файл должен содержать непустой массив events.")
    events = [normalize_event(item) for item in raw["events"]]
    timestamp = events[0]["timestamp"]
    if any(item["timestamp"] != timestamp for item in events):
        raise ValueError("Один файл должен содержать одну полную минуту.")
    if len({item["transaction_id"] for item in events}) != len(events):
        raise ValueError("Идентификаторы операций внутри минуты повторяются.")
    expected = datetime.fromisoformat(timestamp).strftime("%Y%m%dT%H%M%S.json")
    if path.name != expected:
        raise ValueError("Имя файла для этой минуты: " + expected)
    return sorted(events, key=lambda item: item["transaction_id"])


def consume_batch(path, conn, service):
    """Подтверждает файл переносом только после успешного сохранения в PostgreSQL."""
    from src import db

    path = Path(path)
    group = load_batch(path)
    timestamp = group[0]["timestamp"]
    progress = db.stream_state(conn)
    if not progress or progress["kind"] != "inbox":
        raise ValueError("Вход новых операций не зарегистрирован.")
    destination = path.parent / "processed" / path.name
    if destination.exists():
        raise ValueError("В архиве источника уже есть файл этой минуты: " + path.name)
    if timestamp < progress["score_from"]:
        if progress["last_time"] and timestamp <= progress["last_time"]:
            if timestamp != progress["last_time"] or not db.group_already_saved(conn, group):
                raise ValueError("Опоздавшая или изменённая история.")
        else:
            service.ingest_history(group)
        result = {"history": len(group), "saved_cases": 0}
    else:
        if not progress["history_complete"]:
            with conn.transaction():
                db.set_progress(conn, progress["last_time"], history_complete=True)
        result = service.process(group)
    destination.parent.mkdir(exist_ok=True)
    path.rename(destination)
    return result
