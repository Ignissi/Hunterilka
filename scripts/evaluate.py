"""Шаг 4. Оценка замороженных моделей на отложенном по времени test.

Результат: reports/metrics.json, docs/results.md и блок результатов в README.md.
"""

import json

import pandas as pd
from catboost import CatBoostClassifier

from src.metrics import classification_metrics
from src.models import load_models
from src.contracts import validate_feature_tables
from src.report import number, write_report
from src.storage import ROOT, file_hash, load_config, load_table, read_json, write_json


# Диагностика HI-Small: после 10 сентября резко меняется число операций.
# Это отдельный режим данных; величину изменения качества показывают расчёты.
TAIL_START = "2022-09-11T00:00:00"


def period_metrics(test, scores, threshold):
    periods = {}
    is_tail = (test.timestamp >= TAIL_START).to_numpy()
    labels = test.is_laundering.to_numpy()
    timestamps = test.timestamp.to_numpy()

    for name, mask in [("normal", ~is_tail), ("tail", is_tail)]:
        if mask.sum() == 0:
            periods[name] = {"rows": 0}
            continue
        result = classification_metrics(labels[mask], scores[mask], threshold)
        result["first_time"] = str(timestamps[mask].min())
        result["last_time"] = str(timestamps[mask].max())
        periods[name] = result
    return periods


def structure_metrics(models, validation_end, positive_ids, test_scores):
    """Второй уровень: необычность последних версий эпизодов test и охват положительных операций."""
    meta = models["meta"]
    latest = {}
    covered = set()

    with (ROOT / "artifacts" / "candidates.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["detected_at"] < validation_end:
                continue
            covered.update(set(candidate["transaction_ids"]) & positive_ids)
            latest[candidate["case_key"]] = candidate

    episodes = list(latest.values())
    table = pd.DataFrame([c["features"] for c in episodes])[meta["structure_features"]]
    anomaly = -models["structure"].score_samples(table)
    labels = [int(bool(set(c["transaction_ids"]) & positive_ids)) for c in episodes]
    result = classification_metrics(labels, anomaly, meta["structure_threshold"])

    structure_alerts = {i for c, value in zip(episodes, anomaly) if value >= meta["structure_threshold"]
                        for i in c["transaction_ids"]}
    operation_alerts = {i for i in positive_ids if test_scores[i] >= meta["transaction_threshold"]}
    result["episodes"] = len(episodes)
    result["coverage"] = {
        "test_positive_operations": len(positive_ids),
        "in_any_candidate": len(covered),
        "in_structure_alerts": len(structure_alerts & positive_ids),
        "in_operation_alerts": len(operation_alerts),
        "additional_from_structures": len((structure_alerts & positive_ids) - operation_alerts),
    }
    return result


def main():
    models = load_models()
    meta = models["meta"]
    manifest = read_json(ROOT / "data" / "manifest.json")
    validate_feature_tables(load_config(), manifest, read_json(ROOT / "artifacts" / "feature_manifest.json"))
    if meta["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Модель и test относятся к разным наборам данных.")
    features = load_table(ROOT / "artifacts" / "features")
    test = features.loc[features.split == "test"].reset_index(drop=True)
    if test.empty or test.is_laundering.nunique() < 2:
        raise ValueError("Для оценки нужны оба класса в test.")

    reports = {}
    selected_scores = None
    for experiment in meta["experiments"]:
        model = CatBoostClassifier()
        if file_hash(ROOT / "artifacts" / experiment["model_file"]) != meta["artifact_hashes"][experiment["model_file"]]:
            raise ValueError("Изменён артефакт эксперимента: " + experiment["model_file"])
        model.load_model(str(ROOT / "artifacts" / experiment["model_file"]))
        scores = model.predict_proba(test[experiment["features"]])[:, 1]
        result = classification_metrics(test.is_laundering, scores, experiment["threshold"])
        result["periods"] = period_metrics(test, scores, experiment["threshold"])
        reports[experiment["name"]] = result
        if experiment["name"] == meta["selected"]:
            selected_scores = scores

    positive_ids = set(test.loc[test.is_laundering == 1, "transaction_id"])
    test_scores = dict(zip(test.transaction_id, selected_scores))
    structure = structure_metrics(models, manifest["validation_end"], positive_ids, test_scores)

    report = {
        "selected": meta["selected"], "selection_rule": meta["selection_rule"],
        "model_id": meta["model_id"], "training_device": meta["training_device"],
        "data": manifest, "transaction_models": reports, "structure": structure,
    }
    write_json(ROOT / "reports" / "metrics.json", report)
    write_report(report)

    main_model = reports[meta["selected"]]
    print("Основная модель:", meta["selected"])
    for name, title in [("normal", "Обычный период"), ("tail", "Хвост с 11 сентября")]:
        part = main_model["periods"][name]
        if part["rows"] == 0:
            continue
        print(title + ":", part["rows"], "операций,", part["positive"], "положительных,",
              "найдено", part["top_k"]["100"]["found"], "из", part["top_k"]["100"]["reviewed"],
              "рассмотренных, AP", number(part["ap"]))
    print("Весь test: найдено", main_model["top_k"]["100"]["found"], "из", main_model["top_k"]["100"]["reviewed"], "рассмотренных, AP", number(main_model["ap"]))
    print("Второй уровень добавил положительных операций:", structure["coverage"]["additional_from_structures"])


if __name__ == "__main__":
    main()
