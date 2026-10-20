"""Оценка замороженных моделей: пропуски поиска, ранжирование и нагрузка на просмотр."""

import json
import shutil
from collections import Counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.cases import CaseService, build_case
from src.metrics import classification_metrics, top_k_metrics
from src.models import load_models
from src.patterns import make_candidate
from src.review_queue import REVIEW_POLICY, operation_order
from src.storage import ROOT, load_config, read_json, write_json


def read_structures(models, manifest, positive_ids, test_scores):
    latest = {}
    saved = {}
    covered = set()
    families = Counter()
    batch = []
    names = models["meta"]["structure_features"]

    def score_batch():
        if not batch:
            return
        table = pd.DataFrame([c["features"] for c in batch])[names]
        scores = -models["structure"].score_samples(table)
        for candidate, score in zip(batch, scores):
            candidate["anomaly_score"] = float(score)
            latest[candidate["case_key"]] = candidate
            operation_signal = any(test_scores.get(i, -1) >= models["meta"]["transaction_threshold"]
                                   for i in candidate["transaction_ids"])
            if operation_signal or score >= models["meta"]["structure_threshold"]:
                saved[candidate["case_key"]] = candidate
        batch.clear()

    with (ROOT / "artifacts/candidates.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate["detected_at"] < manifest["validation_end"]:
                continue
            families[candidate["family"]] += 1
            covered.update(set(candidate["transaction_ids"]) & positive_ids)
            batch.append(candidate)
            if len(batch) >= 5000:
                score_batch()
    score_batch()
    return list(latest.values()), list(saved.values()), covered, families


def queue_coverage(candidates, positive_ids):
    covered = {i for candidate in candidates for i in candidate["transaction_ids"]}
    return {
        "positive_ids": covered & positive_ids,
        "reviewed": len(candidates),
        "false_cases": sum(not (set(c["transaction_ids"]) & positive_ids) for c in candidates),
        "operations_to_read": sum(len(c["transaction_ids"]) for c in candidates),
    }


def save_review_sample(test, scores, models, manifest, structures, structure_scope):
    meta = models["meta"]
    ranked_ids = test.transaction_id.iloc[operation_order(scores)].tolist()
    events = pd.read_parquet(ROOT / "data/processed/events.parquet",
                             filters=[("transaction_id", "in", ranked_ids)])
    event_by_id = {row["transaction_id"]: row for row in events.to_dict("records")}
    if set(event_by_id) != set(ranked_ids):
        raise ValueError("Не найдены исходные операции для очереди. Карточки не изменены.")
    candidates = [make_candidate("transaction", [event_by_id[i]], [i], event_by_id[i]["timestamp"])
                  for i in ranked_ids]
    needed = set(ranked_ids) | {i for c in structures for i in c["transaction_ids"]}
    vectors = test.loc[test.transaction_id.isin(needed)].set_index("transaction_id")[meta["features"]].to_dict("index")
    test_scores = dict(zip(test.transaction_id, scores))

    # Готовим новый набор до замены старого. Решения пользователя лежат в runs.
    destination = ROOT / "demo/cases"
    temporary = ROOT / "demo/cases_new"
    if temporary.exists():
        shutil.rmtree(temporary)
    service = CaseService(temporary)
    for rank, candidate in enumerate(candidates + structures, start=1):
        review_rank = rank if candidate["family"] == "transaction" else None
        case = build_case(candidate, test_scores, vectors, models, "recorded-test", review_rank=review_rank)
        if case is None:
            continue
        case["data_notice"] = "Производная выборка IBM AML HI-Small. CDLA-Sharing-1.0."
        case["source_url"] = manifest["source_url"]
        service.save(case)
    service.flush()
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    policy = {"version": REVIEW_POLICY, "operation_cases": len(candidates),
              "structure_cases": len(service.index) - len(candidates), "structure_scope": structure_scope}
    write_json(ROOT / "demo/demo_manifest.json", {
        "mode": "Сохранённый test: первые 500 операций по оценке и отдельная выборка структур",
        "model_id": meta["model_id"], "data": manifest, "cases": len(service.index),
        "review_policy": policy, "license": "CDLA-Sharing-1.0",
    })
    return policy


def main():
    config = load_config()
    models = load_models(config)
    meta = models["meta"]
    manifest = read_json(ROOT / "data/processed/dataset_manifest.json")
    test = pd.read_parquet(ROOT / "artifacts/features.parquet", filters=[("split", "=", "test")])
    if test.empty or test.is_laundering.nunique() < 2:
        raise ValueError("Для итоговой проверки нужны оба класса в test.")
    positive_ids = set(test.loc[test.is_laundering == 1, "transaction_id"])
    reports = {}
    selected_scores = None
    for experiment in meta["experiments"]:
        model = CatBoostClassifier()
        model.load_model(str(ROOT / "artifacts" / experiment["model_file"]))
        scores = model.predict_proba(test[experiment["features"]])[:, 1]
        result = classification_metrics(test.is_laundering, scores, experiment["threshold"])
        result["top_k"] = top_k_metrics(test.is_laundering, scores)
        reports[experiment["name"]] = result
        if experiment["name"] == meta["selected"]:
            selected_scores = scores
    if selected_scores is None:
        raise ValueError("Основная модель отсутствует в списке экспериментов.")
    test_scores = dict(zip(test.transaction_id, selected_scores))
    structures, saved_structures, raw_covered, families = read_structures(models, manifest, positive_ids, test_scores)
    labels = [int(bool(set(c["transaction_ids"]) & positive_ids)) for c in structures]
    anomaly = [c["anomaly_score"] for c in structures]
    structure_metrics = classification_metrics(labels, anomaly, meta["structure_threshold"])
    structure_metrics.update(raw_candidates=sum(families.values()), deduplicated_episodes=len(structures),
                             by_family=dict(families), proxy="Есть положительная операция текущего test; это не метка всей схемы")
    last_covered = queue_coverage(structures, positive_ids)["positive_ids"]
    structural_alerts = [c for c in saved_structures if c["anomaly_score"] >= meta["structure_threshold"]]
    signal_covered = queue_coverage(structural_alerts, positive_ids)["positive_ids"]
    operation_hits = {i for i in positive_ids if test_scores[i] >= meta["transaction_threshold"]}
    coverage = {
        "test_positive_operations": len(positive_ids),
        "in_any_candidate": len(raw_covered),
        "in_latest_episodes": len(last_covered),
        "in_structure_alerts": len(signal_covered),
        "in_saved_cases": len(queue_coverage(saved_structures, positive_ids)["positive_ids"]),
        "in_operation_alerts": len(operation_hits),
        "additional_from_structures": len(signal_covered - operation_hits),
        "candidate_recall": len(raw_covered) / len(positive_ids),
    }

    # Прежний смешанный порядок оставлен для воспроизводимого сравнения.
    queue = []
    for candidate in saved_structures:
        maximum = max((test_scores[i] for i in candidate["transaction_ids"] if i in test_scores), default=-1)
        operation_signal = maximum >= meta["transaction_threshold"]
        structure_signal = candidate["anomaly_score"] >= meta["structure_threshold"]
        if operation_signal or structure_signal:
            priority = 1 if operation_signal and structure_signal else (2 if structure_signal else 3)
            queue.append((priority, -candidate["anomaly_score"], -maximum, candidate["candidate_id"], candidate))

    top_ids = set(test.transaction_id.iloc[np.argsort(-selected_scores, kind="stable")[:500]])
    alert_ids = {identifier for identifier, score in test_scores.items() if score >= meta["transaction_threshold"]}
    events = pd.read_parquet(ROOT / "data/processed/events.parquet",
                             filters=[("transaction_id", "in", sorted(top_ids | alert_ids))])
    operation_queue = []
    for event in events.to_dict("records"):
        identifier = event["transaction_id"]
        score = test_scores[identifier]
        candidate = make_candidate("transaction", [event], [identifier], event["timestamp"])
        operation_queue.append((score, identifier, candidate))
        if score >= meta["transaction_threshold"]:
            queue.append((3, 0.0, -score, identifier, candidate))
    operation_queue.sort(key=lambda row: (-row[0], row[1]))
    queue.sort(key=lambda row: row[:4])
    budgets = {}
    for requested in [100, 500]:
        k = min(requested, len(queue), len(operation_queue))
        single = queue_coverage([row[2] for row in operation_queue[:k]], positive_ids)
        combined = queue_coverage([row[4] for row in queue[:k]], positive_ids)
        budgets[str(requested)] = {
            "k": k, "operation_positive_operations": len(single["positive_ids"]),
            "combined_positive_operations": len(combined["positive_ids"]),
            "additional_positive_operations": len(combined["positive_ids"] - single["positive_ids"]),
            "lost_positive_operations": len(single["positive_ids"] - combined["positive_ids"]),
            "operation_false_cases": single["false_cases"], "combined_false_cases": combined["false_cases"],
            "combined_operations_to_read": combined["operations_to_read"],
            "scope": "Последние сохранённые сигналы к концу test; одинаковое число карточек, их размеры различаются",
        }

    selected_structures = sorted(structural_alerts, key=lambda c: (-c["anomaly_score"], c["case_key"]))[:100]
    policy = save_review_sample(test, selected_scores, models, manifest, selected_structures, "top_anomaly_signals")
    report = {
        "selected": meta["selected"], "selection_rule": meta["selection_rule"],
        "transaction_models": reports, "structure": structure_metrics,
        "search_coverage": coverage, "equal_budget": budgets["100"], "review_budgets": budgets,
        "data": manifest, "model_id": meta["model_id"], "training_device": meta["training_device"],
        "review_policy": policy, "review_budgets_policy": "legacy_mixed",
    }
    write_json(ROOT / "reports/metrics.json", report)
    for budget in ["100", "500"]:
        result = reports[meta["selected"]]["top_k"][budget]
        print("Операции: найдено", result["found"], "из первых", result["reviewed"])
    print("Положительных операций в test:", len(positive_ids))
    print("Найдено поиском до второй модели:", len(raw_covered))
    print("Найдено первым уровнем:", len(operation_hits), "; вторым:", len(signal_covered))
    print("Дополнительно от второго уровня:", len(signal_covered - operation_hits))
    from scripts.report_results import write_report
    write_report(report)


if __name__ == "__main__":
    main()
