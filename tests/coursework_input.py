"""Black-box examples for the minute file and the inclusive case thresholds."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.cases import build_case
from src.incoming import load_batch
from src.patterns import make_candidate
from tests.helpers import event


STAMP = "2026-10-21T12:07:00"
FILENAME = "20261021T120700.json"
SIZE_LIMIT = 67_108_864  # Requirement in bytes, independent of the implementation.


def transfer(identifier=1, **changes):
    row = {
        "transaction_id": identifier, "timestamp": STAMP,
        "source_bank": "01", "source_account": "A", "target_bank": "02", "target_account": "B",
        "amount_paid": "15000.00", "payment_currency": "USD",
        "amount_received": "15000.00", "receiving_currency": "USD", "payment_format": "Wire",
    }
    return {**row, **changes}


def minute(*rows):
    return {"events": list(rows)}


def file_inputs():
    one = minute(transfer())
    no_format = transfer()
    del no_format["payment_format"]
    exact_amount = "9" * 30 + "." + "9" * 18
    smallest = "0." + "0" * 17 + "1"
    # Accepted cases specify all IDs in their expected order. Rejections specify
    # the external exception class and a distinguishing fragment of its message.
    def refused(part, kind="ValueError"):
        return {"exception": kind, "contains": part}

    def accepted(*ids, **extra):
        return {"ids": list(ids), **extra}

    return [
        ("ЭР", "Два перевода с ID 2 и 1", minute(transfer(2), transfer(1)), {}, accepted(1, 2)),
        ("ГЗ", "Файл 67 108 864 байта", one, {"size": SIZE_LIMIT}, accepted(1)),
        ("ГЗ", "Файл 67 108 865 байт", one, {"size": SIZE_LIMIT + 1}, refused("64 MiB")),
        ("ПО", "Символьная ссылка на правильный файл", one, {"link": True}, refused("обычный JSON-файл")),
        ("ЭР", "Оборванный JSON", b'{"events":[', {}, refused("Expecting value", "JSONDecodeError")),
        ("ГЗ", "Пустой массив events", minute(), {}, refused("непустой массив events")),
        ("ЭР", "Лишнее поле is_laundering", minute(transfer(is_laundering=0)), {}, refused("Ожидаются поля")),
        ("ЭР", "Отсутствует payment_format", minute(no_format), {}, refused("Ожидаются поля")),
        ("ГЗ", "ID 2⁶³ − 1 и 0", minute(transfer(2**63 - 1), transfer(0)), {}, accepted(0, 2**63 - 1)),
        ("ГЗ", "ID равен −1", minute(transfer(-1)), {}, refused("неотрицательным целым BIGINT")),
        ("ГЗ", "ID равен 2⁶³", minute(transfer(2**63)), {}, refused("неотрицательным целым BIGINT")),
        ("ЭР", "ID задан строкой «5»", minute(transfer("5")), {}, refused("неотрицательным целым BIGINT")),
        ("ЭР", "Время 15:07:00+03:00", minute(transfer(timestamp="2026-10-21T15:07:00+03:00")), {},
         accepted(1, timestamp=STAMP)),
        ("ГЗ", "Время 12:07:30", minute(transfer(timestamp="2026-10-21T12:07:30")), {}, refused("секунды")),
        ("ЭР", "Счёт из трёх пробелов", minute(transfer(source_account="   ")), {}, refused("непустая строка")),
        ("ЭР", "Счёт «12:34»", minute(transfer(source_account="12:34")), {}, refused("двоеточие")),
        ("ГЗ", "Обе суммы «0»", minute(transfer(amount_paid="0", amount_received="0")), {},
         accepted(1, amount_paid="0", amount_received="0")),
        ("ГЗ", "Сумма «-0.01»", minute(transfer(amount_paid="-0.01")), {}, refused("неотрицательная сумма")),
        ("ПО", "Сумма «NaN»", minute(transfer(amount_paid="NaN")), {}, refused("конечная неотрицательная")),
        ("ГЗ", "30 целых и 18 дробных цифр", minute(transfer(amount_paid=exact_amount)), {},
         accepted(1, amount_paid=exact_amount)),
        ("ГЗ", "19 дробных цифр", minute(transfer(amount_paid="0." + "0" * 18 + "1")), {}, refused("18 дробных")),
        ("ГЗ", "31 целая цифра", minute(transfer(amount_paid="1" + "0" * 30)), {}, refused("30 целых")),
        ("ЭР", "В одном файле 12:07 и 12:08", minute(transfer(), transfer(2, timestamp="2026-10-21T12:08:00")), {},
         refused("одну полную минуту")),
        ("ЭР", "Два перевода с ID 1", minute(transfer(), transfer(source_account="C")), {}, refused("повторяются")),
        ("ЭР", "Имя файла за 12:08 при данных за 12:07", one, {"name": "20261021T120800.json"},
         refused(FILENAME)),
        ("ЭР", "Сумма задана числом JSON 5", minute(transfer(amount_paid=5)), {}, refused("десятичной строкой")),
        ("ЭР", "Сумма задана строкой «abc»", minute(transfer(amount_paid="abc")), {}, refused("некорректная сумма")),
        ("ЭР", "Несуществующее 30 февраля", minute(transfer(timestamp="2026-02-30T12:07:00")), {},
         refused("Некорректное время")),
        ("ЭР", "Массив вместо объекта JSON", [transfer()], {}, refused("непустой массив events")),
        ("ПО", "ID задан логическим значением true", minute(transfer(True)), {}, refused("целым BIGINT")),
        ("ЭР", "Байт FF в строке JSON", b'{"events":["\xff"]}', {}, refused("utf-8", "UnicodeDecodeError")),
        ("ГЗ", "Сумма 10⁻¹⁸", minute(transfer(amount_paid=smallest)), {}, accepted(1, amount_paid=smallest)),
    ]


def file_checks():
    results = []
    for i, (method, title, payload, options, expected) in enumerate(file_inputs(), 1):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, options.get("name", FILENAME))
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            with path.open("wb") as stream:
                stream.write(data)
                remaining = options.get("size", len(data)) - len(data)
                block = b" " * min(remaining, 65_536)
                while remaining:
                    written = stream.write(block[:remaining])
                    remaining -= written
            if options.get("link"):
                target = path.rename(Path(tmp, "target.json"))
                path.symlink_to(target)
            try:
                rows = load_batch(path)
            except ValueError as exc:
                observed = {"exception": type(exc).__name__, "message": str(exc)}
                passed = (type(exc).__name__ == expected.get("exception")
                          and expected["contains"] in str(exc))
            else:
                observed = {"ids": [r["transaction_id"] for r in rows]}
                for field in ("timestamp", "amount_paid", "amount_received"):
                    if field in expected and rows:
                        observed[field] = rows[0][field]
                passed = observed == expected
            results.append(dict(id=f"ЧЯ-{i}", method=method, input=title, expected=expected,
                                observed=observed, passed=passed))
    return results


def threshold_checks():
    limits = {"transaction_threshold": 0.5, "structure_threshold": 1.0, "model_id": "control"}
    events = [event(1, "A", "B"), event(2, "B", "C"), event(3, "C", "D")]
    # Literal expected flags are independent of build_case and its threshold formula.
    table = [
        ([0.49], 0.99, None),
        ([0.50], 0.99, (True, False)),
        ([0.49], 1.00, (False, True)),
        ([0.50], 1.00, (True, True)),
        ([0.10, 0.70, 0.20], 0.30, (True, False)),
        ([], None, None),
        ([], 1.00, (False, True)),
        ([0.50], None, (True, False)),
    ]
    results = []
    for i, (scores, anomaly, expected) in enumerate(table, 1):
        candidate = make_candidate("chain", events, ["A", "D"], events[0]["timestamp"])
        candidate["anomaly_score"] = anomaly
        case = build_case(candidate, dict(enumerate(scores, 1)), {}, limits)
        observed = None if case is None else (case["operation_signal"], case["structure_signal"])
        results.append(dict(id=f"ПСС-{i}", scores=scores, anomaly=anomaly, expected=expected,
                            observed=observed, passed=observed == expected))
    return results
