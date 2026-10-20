"""Быстрое обновление очередей после готового расчёта, без обучения и поиска графа."""

from time import perf_counter

import numpy as np
import pandas as pd

from scripts.evaluate import save_review_sample
from scripts.report_results import write_report
from src.cases import CaseService
from src.metrics import classification_metrics, top_k_metrics
from src.models import load_models
from src.storage import ROOT, load_config, read_json, write_json


def main():
    started = perf_counter()
    print("Проверяю сохранённые модели и отчёт.", flush=True)
    models = load_models(load_config())
    meta = models["meta"]
    report = read_json(ROOT / "reports/metrics.json")
    manifest = read_json(ROOT / "data/processed/dataset_manifest.json")
    demo = read_json(ROOT / "demo/demo_manifest.json")
    if report["model_id"] != meta["model_id"] or demo["model_id"] != meta["model_id"]:
        raise ValueError("Модель, отчёт и карточки относятся к разным запускам. Нужен scripts.evaluate.")
    if any(data["events_sha256"] != meta["data_hash"] for data in [manifest, report["data"], demo["data"]]):
        raise ValueError("Данные не совпадают с сохранённой моделью. Файлы не изменены.")
    if report["selected"] != meta["selected"]:
        raise ValueError("В отчёте указана другая основная модель. Файлы не изменены.")

    print("Повторяю только прогноз основной модели на готовых признаках test.", flush=True)
    test = pd.read_parquet(ROOT / "artifacts/features.parquet", filters=[("split", "=", "test")])
    scores = models["transaction"].predict_proba(test[meta["features"]])[:, 1]
    result = classification_metrics(test.is_laundering, scores, meta["transaction_threshold"])
    result["top_k"] = top_k_metrics(test.is_laundering, scores)
    previous = report["transaction_models"][report["selected"]]
    for key in ["rows", "positive", "confusion_matrix", "top_k", "threshold"]:
        if result[key] != previous[key]:
            raise ValueError("Повторный прогноз расходится с отчётом: " + key + ". Карточки не изменены.")
    for key in ["ap", "roc_auc"]:
        if not np.isclose(result[key], previous[key], rtol=1e-8, atol=1e-10):
            raise ValueError("Повторный прогноз расходится с отчётом: " + key + ". Карточки не изменены.")

    # Сохраняем уже подготовленные структурные примеры, не пересчитывая миллионы кандидатов.
    service = CaseService(ROOT / "demo/cases")
    structures = []
    for row in service.index.values():
        if row["family"] == "transaction":
            continue
        case = service.get(row["case_id"])
        candidate = {key: case[key] for key in ["candidate_id", "family", "detected_at", "first_time", "last_time",
                     "transaction_ids", "events", "search_truncated", "equal_timestamps", "anomaly_score"]}
        candidate["case_key"] = case["case_id"]
        candidate["features"] = case["structure_features"]
        structures.append(candidate)
    print("Прогноз совпал с отчётом. Сохраняю отдельные очереди.", flush=True)
    policy = save_review_sample(test, scores, models, manifest, structures, "previous_sample")
    report["review_policy"] = policy
    report["review_budgets_policy"] = "legacy_mixed"
    write_json(ROOT / "reports/metrics.json", report)
    write_report(report)
    for budget, value in result["top_k"].items():
        print("Найдено в первых", budget + ":", value["found"], "из", value["reviewed"])
    print("Карточек операций:", policy["operation_cases"], "; структур:", policy["structure_cases"])
    print("Готово за", round(perf_counter() - started, 1), "с. Модели и признаки не изменены.")


if __name__ == "__main__":
    main()
