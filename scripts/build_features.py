"""Шаг 2. Один проход истории через граф: признаки операций и кандидаты структур.

Результат: artifacts/features/*.parquet, artifacts/structures/*.parquet,
artifacts/candidates.jsonl (полные кандидаты для оценки второго уровня).
"""

import json
import shutil
import time

import pandas as pd

from src.data import split_names
from src.contracts import feature_contract, validate_events
from src.graph import RollingGraph
from src.pipeline import process_group
from src.storage import ROOT, file_hash, load_config, load_table, read_json, save_table, write_json


CHUNK_ROWS = 200000


def write_chunk(rows, folder, number):
    """Записывает накопленные строки отдельным файлом, чтобы не держать всё в памяти."""
    if rows:
        save_table(pd.DataFrame(rows), folder / f"part-{number:05d}.parquet")
        rows.clear()
    return number + 1


def main():
    config = load_config()
    manifest = read_json(ROOT / "data" / "manifest.json")
    validate_events(ROOT / "data" / "events.parquet", manifest)
    if file_hash(ROOT / "data" / "labels.parquet") != manifest.get("labels_sha256"):
        raise ValueError("Метки не соответствуют подготовленному набору. Повторите prepare_data.")
    events = load_table(ROOT / "data" / "events.parquet")
    labels = load_table(ROOT / "data" / "labels.parquet")
    label_by_id = dict(zip(labels.transaction_id, labels.is_laundering))
    parts = split_names(events.timestamp, manifest["train_end"], manifest["validation_end"])
    events["split"] = parts.to_numpy()

    output = ROOT / "artifacts"
    for name in ["features", "structures"]:
        shutil.rmtree(output / name, ignore_errors=True)
        (output / name).mkdir(parents=True)

    state = RollingGraph(config)
    feature_rows, structure_rows, diagnostics = [], [], []
    feature_part, structure_part = 0, 0
    total_rows, total_candidates = 0, 0
    started = time.perf_counter()

    with (output / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for number, (timestamp, table) in enumerate(events.groupby("timestamp", sort=False)):
            group = table.drop(columns="split").to_dict("records")
            part = table.split.iloc[0]
            rows, _, candidates, info = process_group(state, group, config)

            for event, row in zip(group, rows):
                row.update(transaction_id=event["transaction_id"], timestamp=timestamp,
                           is_laundering=label_by_id[event["transaction_id"]], split=part)
                feature_rows.append(row)
            total_rows += len(rows)

            for candidate in candidates:
                row = dict(candidate["features"])
                row.update(candidate_id=candidate["candidate_id"], case_key=candidate["case_key"],
                           timestamp=timestamp, family=candidate["family"], split=part)
                structure_rows.append(row)
                stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")
            total_candidates += len(candidates)
            info.update(timestamp=timestamp, split=part)
            diagnostics.append(info)

            if len(feature_rows) >= CHUNK_ROWS:
                feature_part = write_chunk(feature_rows, output / "features", feature_part)
            if len(structure_rows) >= CHUNK_ROWS:
                structure_part = write_chunk(structure_rows, output / "structures", structure_part)
            if number % 500 == 0:
                elapsed = time.perf_counter() - started
                print(f"Минут: {number + 1}; операций: {total_rows}/{len(events)}; структур: {total_candidates}; секунд: {elapsed:.0f}", flush=True)

    write_chunk(feature_rows, output / "features", feature_part)
    write_chunk(structure_rows, output / "structures", structure_part)
    if total_candidates == 0:
        raise ValueError("Кандидатов нет. Проверьте данные и настройки поиска.")

    (ROOT / "reports").mkdir(exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(ROOT / "reports" / "search_diagnostics.csv", index=False)
    write_json(output / "feature_manifest.json", {
        "rows": total_rows, "candidates": total_candidates,
        "seconds": time.perf_counter() - started,
        "nodes": state.graph.number_of_nodes(), "edges": state.graph.number_of_edges(),
        "scope": manifest["scope"],
        "dataset_id": manifest["dataset_id"], "feature_contract": feature_contract(config),
    })
    print("Признаки готовы. Секунд:", round(time.perf_counter() - started, 1))


if __name__ == "__main__":
    main()
