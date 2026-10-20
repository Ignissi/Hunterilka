"""Подготовка обзора истории, каталога схем и серого окружения сохранённых случаев."""

import json
import re
from time import perf_counter

import pandas as pd

from src.cases import CaseService, build_case
from src.metrics import top_k_metrics
from src.models import load_models
from src.snapshot import HistorySlices, WORKSPACE_VERSION
from src.storage import ROOT, file_hash, load_config, read_json, write_json


def collect_examples(wanted):
    found = []
    identifier = re.compile(r'"candidate_id"\s*:\s*"([^"]+)"')
    if not wanted:
        return found
    with (ROOT / "artifacts/candidates.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            match = identifier.search(line[:160])
            key = match.group(1) if match else json.loads(line)["candidate_id"]
            if key in wanted:
                found.append(json.loads(line))
                wanted.remove(key)
                if not wanted:
                    break
            if number % 1000000 == 0:
                print("Просмотрено записей каталога:", number, "; осталось примеров:", len(wanted), flush=True)
    if wanted:
        raise ValueError("Некоторые структуры отсутствуют в candidates.jsonl. Файлы карточек не изменены.")
    return found


def main():
    started = perf_counter()
    config = load_config()
    models = load_models(config)
    meta = models["meta"]
    demo = read_json(ROOT / "demo/demo_manifest.json")
    report = read_json(ROOT / "reports/metrics.json")
    if demo["model_id"] != meta["model_id"] or report["model_id"] != meta["model_id"]:
        raise ValueError("Карточки и отчёт относятся к другой модели. Выполните scripts.evaluate.")
    if not demo.get("review_policy"):
        raise ValueError("Сначала обновите очередь: python -m scripts.refresh_review.")
    if demo["data"]["events_sha256"] != meta["data_hash"]:
        raise ValueError("Данные не совпадают с моделью.")

    print("1/5. Считаю историю и распределение найденных структур. Обучение не запускается.", flush=True)
    if file_hash(ROOT / "data/processed/events.parquet") != meta["data_hash"]:
        raise ValueError("Подготовленная история изменена после расчёта. Карточки не изменены.")
    events = pd.read_parquet(ROOT / "data/processed/events.parquet")
    if len(events) != demo["data"]["rows"]:
        raise ValueError("Размер истории не совпадает с паспортом.")
    days = events.timestamp.str[:10]
    daily = events.groupby(days).size().rename("operations").to_frame()
    daily.index.name = "day"
    formats = events.groupby([days, "payment_format"]).size().rename("operations").reset_index()
    formats = formats.rename(columns={"timestamp": "day"})
    structures = pd.read_parquet(ROOT / "artifacts/structures.parquet", columns=[
        "timestamp", "family", "case_key", "candidate_id", "split", "log_events", "search_truncated"])
    structures["day"] = structures.timestamp.str[:10]
    schemes = structures.groupby(["day", "family"]).agg(candidates=("candidate_id", "size"),
                                                        episodes=("case_key", "nunique")).reset_index()
    # Примеры форм выбираются по размеру и полноте поиска, без меток и подбора по качеству.
    compact = structures.loc[(structures.split == "test") & (structures.log_events <= 3.3)]
    compact = compact.sort_values(["search_truncated", "log_events", "timestamp"], ascending=[True, False, False])
    examples = compact.drop_duplicates("case_key").groupby("family", sort=False).head(3)
    missing = set(structures.loc[structures.split == "test", "family"]) - set(examples.family)
    if missing:
        extra = structures.loc[(structures.split == "test") & structures.family.isin(missing)]
        examples = pd.concat([examples, extra.drop_duplicates("case_key").groupby("family").head(1)])
    identifiers = set(examples.candidate_id)
    del structures, compact, examples

    print("2/5. Извлекаю реальные примеры всех найденных форм из сохранённого каталога.", flush=True)
    candidates = collect_examples(identifiers)
    print("3/5. Проверяю готовую модель и считаю сигналы по дням.", flush=True)
    test = pd.read_parquet(ROOT / "artifacts/features.parquet", filters=[("split", "=", "test")])
    predictions = models["transaction"].predict_proba(test[meta["features"]])[:, 1]
    if top_k_metrics(test.is_laundering, predictions) != report["transaction_models"][meta["selected"]]["top_k"]:
        raise ValueError("Прогноз не совпадает с отчётом. Карточки не изменены.")
    test_scores = dict(zip(test.transaction_id, predictions))
    alerts = test.loc[predictions >= meta["transaction_threshold"]]
    daily["operation_signals"] = alerts.groupby(alerts.timestamp.str[:10]).size().reindex(daily.index, fill_value=0)
    if int(daily.operation_signals.sum()) != report["transaction_models"][meta["selected"]]["alerts"]:
        raise ValueError("Число сигналов расходится с отчётом. Карточки не изменены.")
    if candidates:
        values = pd.DataFrame([c["features"] for c in candidates])[meta["structure_features"]]
        for candidate, score in zip(candidates, -models["structure"].score_samples(values)):
            candidate["anomaly_score"] = float(score)
    needed = {i for c in candidates for i in c["transaction_ids"]}
    vectors = test.loc[test.transaction_id.isin(needed)].set_index("transaction_id")[meta["features"]].to_dict("index")
    del test

    print("4/5. Сохраняю окружение случаев на момент обнаружения, без будущих переводов.", flush=True)
    history = HistorySlices(events)
    service = CaseService(ROOT / "demo/cases")
    cases = [service.get(i) for i in service.index]
    existing = {c["case_id"] for c in cases}
    for candidate in candidates:
        if candidate["case_key"] in existing:
            continue
        case = build_case(candidate, test_scores, vectors, models, "recorded-test", is_example=True)
        case["data_notice"] = "Производная выборка IBM AML HI-Small. CDLA-Sharing-1.0."
        case["source_url"] = demo["data"]["source_url"]
        cases.append(case)
    sample = {}
    for number, case in enumerate(cases, start=1):
        context, truncated = history.around(case["events"], case["detected_at"], config["window_days"])
        case.update(context=context, context_truncated=truncated, context_version=WORKSPACE_VERSION)
        service.save(case)
        for row in case["events"] + context:
            sample[row["transaction_id"]] = row
        if number % 100 == 0:
            print("Готово окружений:", number, "/", len(cases), flush=True)
    service.flush()

    print("5/5. Сохраняю обзор сети и паспорт рабочего места.", flush=True)
    # Для каждого дня сохраняется участок вокруг наиболее активного счёта.
    seeds = []
    for day, positions in events.groupby(days, sort=True).indices.items():
        portion = events.iloc[positions]
        activity = portion.source_key.value_counts().add(portion.target_key.value_counts(), fill_value=0)
        seed = activity.idxmax()
        anchor = portion.loc[(portion.source_key == seed) | (portion.target_key == seed)].tail(1).to_dict("records")
        context, _ = history.around(anchor, anchor[0]["timestamp"], days=1)
        seeds.append({"day": day, "account": seed})
        for row in anchor + context:
            sample[row["transaction_id"]] = row
    incoming = events.target_key.value_counts()
    outgoing = events.source_key.value_counts()
    nodes = pd.DataFrame({"incoming": incoming, "outgoing": outgoing}).fillna(0).astype(int)
    nodes.index.name = "account"
    nodes = nodes.reset_index()
    nodes["bank"] = nodes.account.map(lambda value: json.loads(value)[0])
    nodes["number"] = nodes.account.map(lambda value: json.loads(value)[1])
    files = {
        "periods.parquet": daily.reset_index(), "schemes.parquet": schemes, "formats.parquet": formats,
        "network_sample.parquet": pd.DataFrame(sample.values()), "network_nodes.parquet": nodes,
    }
    for name, table in files.items():
        table.to_parquet(ROOT / "demo" / name, index=False)
    workspace = {"version": WORKSPACE_VERSION, "model_id": meta["model_id"], "data_hash": meta["data_hash"],
                 "cases": len(service.index), "seeds": seeds,
                 "files": {name: file_hash(ROOT / "demo" / name) for name in files},
                 "example_policy": "До трёх компактных кандидатов каждого семейства test; без меток, отдельная пометка примера"}
    write_json(ROOT / "demo/workspace_manifest.json", workspace)
    demo["cases"] = len(service.index)
    demo["workspace_version"] = WORKSPACE_VERSION
    demo["review_policy"]["structure_cases"] = sum(c["family"] != "transaction" for c in cases)
    demo["review_policy"]["structure_scope"] = "signals_and_form_examples"
    report["review_policy"] = demo["review_policy"]
    write_json(ROOT / "demo/demo_manifest.json", demo)
    write_json(ROOT / "reports/metrics.json", report)
    from scripts.report_results import write_report
    write_report(report)
    print("Рабочее место готово. Карточек:", len(service.index), "; секунд:", round(perf_counter() - started, 1))
    print("Модели, признаки и итоговые метрики не менялись.")


if __name__ == "__main__":
    main()
