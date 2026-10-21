"""Импорт IBM AML: проверка строк, составные ключи счетов, разделение по времени.

Метки отмывания сохраняются в отдельный файл. Рабочая часть системы их не видит.
"""

import csv

import numpy as np
import pandas as pd

from src.storage import file_hash, save_table, short_hash, write_json


HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]

COLUMNS = [
    "timestamp", "source_bank", "source_account", "target_bank", "target_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering",
]

EVENT_COLUMNS = [
    "transaction_id", "timestamp",
    "source_bank", "source_account", "target_bank", "target_account",
    "amount_paid", "payment_currency", "amount_received", "receiving_currency",
    "payment_format", "source_key", "target_key",
]

CRYPTO = {"bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency"}


def account_key(bank, account):
    # Номер счёта уникален только внутри банка, поэтому ключ узла составной.
    return str(bank) + ":" + str(account)


def read_source(csv_path):
    """Читает исходный CSV целиком, отбрасывает крипто-операции и некорректные строки."""
    with open(csv_path, encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if [name.strip() for name in header] != HEADER:
        raise ValueError("Ожидается исходный HI-Small_Trans.csv с двумя колонками Account.")

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    frame.columns = COLUMNS
    frame["source_row"] = np.arange(1, len(frame) + 1)

    crypto = (
        frame.payment_currency.str.lower().isin(CRYPTO)
        | frame.receiving_currency.str.lower().isin(CRYPTO)
        | frame.payment_format.str.lower().isin(CRYPTO)
    )
    frame = frame.loc[~crypto].copy()

    reason = pd.Series("", index=frame.index)
    dates = pd.to_datetime(frame.timestamp, format="%Y/%m/%d %H:%M", errors="coerce")
    reason.loc[dates.isna()] = "Некорректное время"
    for column in ["source_bank", "source_account", "target_bank", "target_account",
                   "payment_currency", "receiving_currency", "payment_format"]:
        reason.loc[frame[column].str.strip() == ""] = "Пустой идентификатор или категория"
    for column in ["amount_paid", "amount_received"]:
        numbers = pd.to_numeric(frame[column], errors="coerce")
        valid = frame[column].str.fullmatch(r"\d+(\.\d+)?") & numbers.notna() & (numbers >= 0)
        reason.loc[~valid] = "Некорректная сумма"
    reason.loc[~frame.is_laundering.isin(["0", "1"])] = "Некорректная метка"

    rejected = pd.DataFrame({"source_row": frame.source_row[reason != ""], "reason": reason[reason != ""]})
    frame = frame.loc[reason == ""].copy()
    frame["timestamp"] = dates.loc[frame.index].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame, int(crypto.sum()), rejected


def split_boundaries(timestamps):
    """Границы train / validation / test по числу операций: 60% / 20% / 20%.

    Границы выбираются только по времени, без меток. Все операции с одинаковым
    временем попадают в одну часть, поэтому доли могут немного отличаться.
    """
    ordered = timestamps.sort_values(kind="stable").reset_index(drop=True)
    train_end = ordered.iloc[int(len(ordered) * 0.6)]
    validation_end = ordered.iloc[int(len(ordered) * 0.8)]
    return train_end, validation_end


def split_names(timestamps, train_end, validation_end):
    """Имя части для каждой операции по её времени."""
    names = np.where(timestamps < train_end, "train", "test").astype(object)
    names[(timestamps >= train_end) & (timestamps < validation_end)] = "validation"
    return pd.Series(names, index=timestamps.index)


def prepare_data(csv_path, output, limit=0):
    """CSV -> events.parquet (без меток), labels.parquet, manifest.json."""
    frame, crypto_rows, rejected = read_source(csv_path)
    if frame.empty:
        raise ValueError("После проверки не осталось операций.")

    frame = frame.sort_values(["timestamp", "source_row"], kind="stable").reset_index(drop=True)
    if limit and len(frame) > limit:
        # Укороченный профиль для быстрой проверки: берём первые строки до конца минуты.
        boundary = frame.timestamp.iloc[limit - 1]
        frame = frame.loc[frame.timestamp <= boundary]

    frame["transaction_id"] = frame.source_row.astype("int64")
    frame["source_key"] = [account_key(b, a) for b, a in zip(frame.source_bank, frame.source_account)]
    frame["target_key"] = [account_key(b, a) for b, a in zip(frame.target_bank, frame.target_account)]

    events = frame[EVENT_COLUMNS].reset_index(drop=True)
    labels = frame[["transaction_id"]].copy()
    labels["is_laundering"] = frame.is_laundering.astype(int).to_numpy()

    train_end, validation_end = split_boundaries(events.timestamp)
    parts = split_names(events.timestamp, train_end, validation_end)
    summary = {}
    for name in ["train", "validation", "test"]:
        chosen = labels.is_laundering[parts == name]
        summary[name] = {"rows": int(len(chosen)), "positive": int(chosen.sum())}
        if len(chosen) < len(events) * 0.1:
            raise ValueError(name + ": менее 10% операций. Истории слишком мало.")
        if chosen.sum() == 0 or chosen.sum() == len(chosen):
            raise ValueError(name + ": нужны оба класса. Расширьте историю.")

    output.mkdir(parents=True, exist_ok=True)
    save_table(events, output / "events.parquet")
    save_table(labels, output / "labels.parquet")
    rejected.to_csv(output / "rejected.csv", index=False)

    manifest = {
        "source": "IBM AML / HI-Small / Kaggle ealtman2019",
        "source_url": "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        "license": "CDLA-Sharing-1.0",
        "rows": int(len(events)), "positive": int(labels.is_laundering.sum()),
        "excluded_crypto": crypto_rows, "invalid_rows": int(len(rejected)),
        "first_time": events.timestamp.iloc[0], "last_time": events.timestamp.iloc[-1],
        "history_days": (pd.Timestamp(events.timestamp.iloc[-1]) - pd.Timestamp(events.timestamp.iloc[0])).total_seconds() / 86400,
        "scope": "full_source" if limit == 0 else "short_profile",
        "train_end": train_end, "validation_end": validation_end, "parts": summary,
        "accounts": int(len(set(events.source_key) | set(events.target_key))),
        "source_sha256": file_hash(csv_path),
        "events_sha256": file_hash(output / "events.parquet"),
        "labels_sha256": file_hash(output / "labels.parquet"),
        "preparation_version": "ibm-bank-2.1",
    }
    manifest["dataset_id"] = short_hash({
        "source": manifest["source_sha256"], "preparation_version": manifest["preparation_version"],
        "events": manifest["events_sha256"], "labels": manifest["labels_sha256"],
    })
    write_json(output / "manifest.json", manifest)
    return manifest
