"""Импорт IBM AML. Метки отделяются от рабочего потока."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.storage import file_hash, write_json


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

CRYPTO = {"bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency"}
SPLIT_METHOD = "chronological_rows_60_20_20"


def chronological_boundaries(timestamps):
    # Границы задаёт число операций, а не длина календарного периода.
    # Все операции с временем границы целиком переходят в следующую часть.
    ordered = timestamps.sort_values(kind="stable").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("Нельзя разделить пустую историю.")
    train_end = pd.Timestamp(ordered.iloc[int(len(ordered) * 0.6)]).isoformat()
    validation_end = pd.Timestamp(ordered.iloc[int(len(ordered) * 0.8)]).isoformat()
    return train_end, validation_end


def validate_split(manifest):
    if manifest.get("split_method") != SPLIT_METHOD:
        raise ValueError("Нужна новая подготовка данных с хронологическим разделением 60/20/20 по операциям.")
    total = manifest["rows"]
    parts = manifest["parts"]
    if total <= 0 or sum(part["rows"] for part in parts.values()) != total:
        raise ValueError("Число операций в частях не совпадает с размером истории.")
    if manifest["train_end"] >= manifest["validation_end"]:
        raise ValueError("Не удалось получить три временные части. Расширьте историю.")
    for name in ["train", "validation", "test"]:
        part = parts[name]
        if part["rows"] < total * 0.1:
            raise ValueError(name + ": менее 10% операций. Проверьте временные группы; расчёт остановлен.")
        if not 0 < part["positive"] < part["rows"]:
            raise ValueError(name + ": нужны оба класса. Расширьте историю; метки не используются для выбора границ.")


def account_key(bank, account):
    # Составной ключ не склеивает одинаковые счета разных банков.
    return json.dumps([str(bank), str(account)], ensure_ascii=False, separators=(",", ":"))


def prefix_complete_groups(frame, limit):
    frame = frame.sort_values(["timestamp", "source_row"], kind="stable")
    if limit and len(frame) > limit:
        boundary = frame.iloc[limit - 1]["timestamp"]
        frame = frame.loc[frame.timestamp <= boundary]
    return frame


def prepare_data(csv_path, output, limit=0):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if [name.strip() for name in header] != HEADER:
        raise ValueError("Ожидается исходный HI-Small_Trans.csv с двумя колонками Account.")

    checksum = file_hash(csv_path)
    selected = pd.DataFrame()
    chunks = []
    rejected = []
    offset = 0
    filtered = 0
    filtered_positive = 0
    invalid_count = 0
    source_positive = 0
    currencies = set()
    formats = set()

    # Читаем весь CSV: первые строки файла не обязаны быть первыми по времени.
    reader = pd.read_csv(csv_path, dtype=str, chunksize=100000, keep_default_na=False)
    for chunk in reader:
        chunk.columns = COLUMNS
        chunk["source_row"] = np.arange(offset + 1, offset + len(chunk) + 1)
        offset += len(chunk)
        source_positive += int((chunk.is_laundering == "1").sum())
        currencies.update(chunk.payment_currency.unique())
        currencies.update(chunk.receiving_currency.unique())
        formats.update(chunk.payment_format.unique())

        crypto = (
            chunk.payment_currency.str.lower().isin(CRYPTO)
            | chunk.receiving_currency.str.lower().isin(CRYPTO)
            | chunk.payment_format.str.lower().isin(CRYPTO)
        )
        filtered += int(crypto.sum())
        filtered_positive += int(((chunk.is_laundering == "1") & crypto).sum())
        chunk = chunk.loc[~crypto].copy()

        dates = pd.to_datetime(chunk.timestamp, format="%Y/%m/%d %H:%M", errors="coerce")
        reason = pd.Series("", index=chunk.index)
        reason.loc[dates.isna()] = "Некорректное время"

        for column in ["source_bank", "source_account", "target_bank", "target_account",
                       "payment_currency", "receiving_currency", "payment_format"]:
            reason.loc[chunk[column].str.strip() == ""] = "Пустой идентификатор или категория"

        for column in ["amount_paid", "amount_received"]:
            numbers = pd.to_numeric(chunk[column], errors="coerce")
            valid = chunk[column].str.fullmatch(r"\d+(\.\d+)?")
            valid = valid & numbers.notna() & np.isfinite(numbers) & (numbers >= 0)
            reason.loc[~valid] = "Некорректная сумма"

        reason.loc[~chunk.is_laundering.isin(["0", "1"])] = "Некорректная метка"
        invalid = reason != ""
        invalid_count += int(invalid.sum())
        for row_number, why in zip(chunk.loc[invalid, "source_row"], reason.loc[invalid]):
            rejected.append({"source_row": int(row_number), "reason": why})

        chunk = chunk.loc[~invalid].copy()
        chunk["timestamp"] = dates.loc[~invalid].dt.strftime("%Y-%m-%dT%H:%M:%S")
        if limit:
            selected = pd.concat([selected, chunk], ignore_index=True)
            selected = prefix_complete_groups(selected, limit)
        else:
            chunks.append(chunk)
        if offset % 1000000 == 0:
            print("Прочитано строк:", offset, flush=True)

    if not limit and chunks:
        selected = prefix_complete_groups(pd.concat(chunks, ignore_index=True), 0)
        chunks.clear()

    if selected.empty:
        raise ValueError("После проверки и банковского фильтра не осталось операций.")

    selected = selected.reset_index(drop=True)
    selected["transaction_id"] = checksum[:16] + ":" + selected.source_row.astype(str)
    selected["source_key"] = [account_key(b, a) for b, a in zip(selected.source_bank, selected.source_account)]
    selected["target_key"] = [account_key(b, a) for b, a in zip(selected.target_bank, selected.target_account)]
    labels = selected[["transaction_id", "is_laundering"]].copy()
    labels["is_laundering"] = labels.is_laundering.astype(int)
    events = selected.drop(columns="is_laundering")

    first = pd.Timestamp(events.timestamp.iloc[0])
    last = pd.Timestamp(events.timestamp.iloc[-1])
    span = last - first
    train_end, validation_end = chronological_boundaries(events.timestamp)
    parts = split_period(events.timestamp, train_end, validation_end)
    summary = {}
    for part in ["train", "validation", "test"]:
        chosen = labels.loc[parts == part, "is_laundering"]
        summary[part] = {"rows": len(chosen), "positive": int(chosen.sum())}

    events.to_parquet(output / "events.parquet", index=False)
    labels.to_parquet(output / "labels.parquet", index=False)
    pd.DataFrame(rejected, columns=["source_row", "reason"]).to_csv(output / "rejected.csv", index=False)

    manifest = {
        "source": "IBM AML / HI-Small / Kaggle ealtman2019",
        "source_url": "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        "license": "CDLA-Sharing-1.0", "synthetic_source": True,
        "source_file": Path(csv_path).name, "sha256": checksum,
        "source_rows": offset, "source_positive": source_positive,
        "excluded_crypto": filtered, "excluded_crypto_positive": filtered_positive,
        "invalid_rows": invalid_count, "profile_limit": limit,
        "rows": len(events), "positive": int(labels.is_laundering.sum()),
        "first_time": first.isoformat(), "last_time": last.isoformat(),
        "history_days": span.total_seconds() / 86400,
        "scope": "full_source" if limit == 0 else "technical_prefix",
        "split_method": SPLIT_METHOD,
        "train_end": train_end, "validation_end": validation_end, "parts": summary,
        "source_currencies": sorted(currencies), "source_formats": sorted(formats),
        "excluded_values": sorted(CRYPTO), "timezone": "Не задан источником",
        "accounts": len(set(events.source_key) | set(events.target_key)),
        "pairs": len(events[["source_key", "target_key"]].drop_duplicates()),
        "events_sha256": file_hash(output / "events.parquet"),
        "preparation_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "dataset_manifest.json", manifest)
    return manifest


def split_period(timestamps, train_end, validation_end):
    dates = pd.to_datetime(timestamps)
    result = pd.Series("test", index=timestamps.index)
    result.loc[dates < pd.Timestamp(validation_end)] = "validation"
    result.loc[dates < pd.Timestamp(train_end)] = "train"
    return result
