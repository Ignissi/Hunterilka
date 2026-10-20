"""Один проход истории. Таблицы записываются порциями, без списка всех признаков в памяти."""

import json
import platform
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import validate_split
from src.graph import RollingGraph
from src.pipeline import process_group
from src.storage import ROOT, fingerprint, load_config, logic_hash, read_json, write_json


def append_table(writer, rows, destination):
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
    writer.write_table(table.cast(writer.schema))
    rows.clear()
    return writer


def main():
    config = load_config()
    manifest = read_json(ROOT / "data/processed/dataset_manifest.json")
    validate_split(manifest)
    events = pd.read_parquet(ROOT / "data/processed/events.parquet")
    if len(events) > config["max_active_events"] and manifest["history_days"] <= config["window_days"]:
        raise ValueError("Весь профиль входит в окно. Увеличьте max_active_events с учётом памяти.")
    labels = pd.read_parquet(ROOT / "data/processed/labels.parquet").set_index("transaction_id").is_laundering
    label_by_id = labels.to_dict()
    state = RollingGraph(config)
    feature_rows, candidate_rows, diagnostics = [], [], []
    feature_writer = None
    structure_writer = None
    total_rows, total_candidates = 0, 0
    started = time.perf_counter()
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    manifest_path = output / "feature_manifest.json"
    manifest_path.unlink(missing_ok=True)

    try:
        with (output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
            for number, (timestamp, table) in enumerate(events.groupby("timestamp", sort=False)):
                group = table.to_dict("records")
                rows, _, candidates, info = process_group(state, group, config)
                part = "train" if timestamp < manifest["train_end"] else ("validation" if timestamp < manifest["validation_end"] else "test")
                for event, row in zip(group, rows):
                    identifier = event["transaction_id"]
                    row.update(transaction_id=identifier, timestamp=timestamp,
                               is_laundering=label_by_id[identifier], split=part)
                    feature_rows.append(row)
                total_rows += len(rows)

                for candidate in candidates:
                    row = dict(candidate["features"])
                    row.update(candidate_id=candidate["candidate_id"], case_key=candidate["case_key"], timestamp=timestamp,
                               family=candidate["family"], split=part)
                    candidate_rows.append(row)
                    stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                total_candidates += len(candidates)
                info.update(timestamp=timestamp, split=part)
                diagnostics.append(info)

                if len(feature_rows) >= 25000:
                    feature_writer = append_table(feature_writer, feature_rows, output / "features.parquet")
                if len(candidate_rows) >= 10000:
                    structure_writer = append_table(structure_writer, candidate_rows, output / "structures.parquet")
                if number % 200 == 0:
                    elapsed = time.perf_counter() - started
                    print(f"Групп: {number + 1}; операций: {total_rows}/{len(events)}; структур: {total_candidates}; секунд: {elapsed:.0f}", flush=True)

        if feature_rows:
            feature_writer = append_table(feature_writer, feature_rows, output / "features.parquet")
        if candidate_rows:
            structure_writer = append_table(structure_writer, candidate_rows, output / "structures.parquet")
    finally:
        for writer in [feature_writer, structure_writer]:
            if writer is not None:
                writer.close()

    if not total_candidates:
        raise ValueError("Кандидатов нет. Проверьте данные и настройки поиска.")
    pd.DataFrame(diagnostics).to_csv(ROOT / "reports/search_diagnostics.csv", index=False)
    write_json(manifest_path, {
        "config": config, "config_hash": fingerprint(config),
        "data_hash": manifest["events_sha256"], "logic_hash": logic_hash(),
        "rows": total_rows, "candidates": total_candidates,
        "seconds": time.perf_counter() - started, "python": platform.python_version(),
        "platform": platform.platform(), "active_events": len(state.events),
        "nodes": state.graph.number_of_nodes(), "edges": state.graph.number_of_edges(),
        "group_ms_p50": float(pd.Series([x["group_seconds"] for x in diagnostics]).quantile(.5) * 1000),
        "group_ms_p95": float(pd.Series([x["group_seconds"] for x in diagnostics]).quantile(.95) * 1000),
        "incomplete_groups": sum(x["search_truncated"] for x in diagnostics),
        "groups": len(diagnostics), "scope": manifest["scope"],
    })
    print("Признаки готовы. Секунд:", round(time.perf_counter() - started, 1))


if __name__ == "__main__":
    main()
