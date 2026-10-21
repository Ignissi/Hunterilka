"""Шаг 3. Четыре варианта CatBoost (A/B, с весами и без) и Isolation Forest для структур.

A: признаки операции и истории счетов. B: A плюс графовые признаки.
Основная модель выбирается на validation, test до оценки не используется.
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.features import CATEGORICAL, GRAPH_FEATURES
from src.contracts import feature_contract, validate_feature_tables
from src.metrics import budget_threshold, top_k_metrics
from src.storage import ROOT, file_hash, load_config, load_table, read_json, short_hash, write_json


SERVICE_COLUMNS = {"transaction_id", "timestamp", "is_laundering", "split"}


def fit_catboost(train, validation, names, weighted, config, device):
    model = CatBoostClassifier(
        iterations=config["iterations"], depth=config["depth"], learning_rate=config["learning_rate"],
        loss_function="Logloss", random_seed=config["seed"],
        auto_class_weights="Balanced" if weighted else None,
        task_type=device, allow_writing_files=False, verbose=False,
    )
    model.fit(train[names], train.is_laundering, cat_features=CATEGORICAL,
              eval_set=(validation[names], validation.is_laundering), early_stopping_rounds=60)
    return model


def main():
    parser = argparse.ArgumentParser(description="Обучение CatBoost и Isolation Forest")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    args = parser.parse_args()
    config = load_config()
    folder = ROOT / "artifacts"
    manifest = read_json(ROOT / "data" / "manifest.json")
    validate_feature_tables(config, manifest, read_json(folder / "feature_manifest.json"))

    features = load_table(folder / "features")
    train = features.loc[features.split == "train"]
    validation = features.loc[features.split == "validation"]
    for title, part in [("train", train), ("validation", validation)]:
        if part.is_laundering.nunique() < 2:
            raise ValueError(title + ": нужны оба класса.")

    full_names = [name for name in features.columns if name not in SERVICE_COLUMNS]
    basic_names = [name for name in full_names if name not in GRAPH_FEATURES]
    experiments = []

    for variant, names in [("A", basic_names), ("B", full_names)]:
        for weighted in [False, True]:
            started = time.perf_counter()
            model = fit_catboost(train, validation, names, weighted, config, args.device)
            scores = model.predict_proba(validation[names])[:, 1]
            name = variant + ("_balanced" if weighted else "_plain")
            model.save_model(str(folder / (name + ".cbm")))
            experiments.append({
                "name": name, "variant": variant, "weighted": weighted,
                "model_file": name + ".cbm", "features": names,
                "validation_ap": float(average_precision_score(validation.is_laundering, scores)),
                "threshold": budget_threshold(scores, min(config["review_k"], len(scores)) / len(scores)),
                "validation_top_k": top_k_metrics(validation.is_laundering, scores),
                "trees": model.tree_count_, "seconds": time.perf_counter() - started,
            })
            print(name, "validation AP:", round(experiments[-1]["validation_ap"], 6), flush=True)

    # Выбор по validation: больше находок в первых 100, при равенстве выше AP.
    selected = max(experiments, key=lambda row: (row["validation_top_k"]["100"]["found"], row["validation_ap"]))

    structures = load_table(folder / "structures")
    structure_names = [name for name in structures.columns
                       if name not in {"candidate_id", "case_key", "timestamp", "family", "split"}]
    structure_train = structures.loc[structures.split == "train", structure_names]
    structure_validation = structures.loc[structures.split == "validation"].drop_duplicates("case_key", keep="last")[structure_names]
    if len(structure_train) < 20 or len(structure_validation) < 5:
        raise ValueError("Недостаточно структур для второго уровня.")

    isolation = IsolationForest(n_estimators=200, max_samples=min(256, len(structure_train)),
                                contamination="auto", random_state=config["seed"], n_jobs=-1)
    isolation.fit(structure_train)
    validation_anomaly = -isolation.score_samples(structure_validation)
    joblib.dump(isolation, folder / "isolation_forest.joblib")

    meta = {
        "version": "2.1", "training_device": args.device,
        "dataset_id": manifest["dataset_id"], "feature_contract": feature_contract(config),
        "seed": config["seed"], "training_config": config,
        "selection_rule": "validation: found@100, затем AP",
        "selected": selected["name"], "model_file": selected["model_file"],
        "features": selected["features"], "structure_features": structure_names,
        "transaction_threshold": selected["threshold"],
        "structure_threshold": budget_threshold(validation_anomaly, min(config["review_k"], len(validation_anomaly)) / len(validation_anomaly)),
        "structure_train_rows": int(len(structure_train)),
        "structure_medians": structure_train.median().to_dict(),
        "structure_q95": structure_train.quantile(0.95).to_dict(),
        "experiments": experiments,
    }
    files = [row["model_file"] for row in experiments] + ["isolation_forest.joblib"]
    meta["artifact_hashes"] = {name: file_hash(folder / name) for name in files}
    meta["model_id"] = short_hash({
        key: meta[key] for key in ["dataset_id", "feature_contract", "artifact_hashes",
                                 "selected", "transaction_threshold", "structure_threshold", "seed"]
    })
    write_json(folder / "model_meta.json", meta)

    table = pd.DataFrame([{k: v for k, v in row.items() if k not in {"features", "validation_top_k"}} for row in experiments])
    (ROOT / "reports").mkdir(exist_ok=True)
    table.to_csv(ROOT / "reports" / "experiments.csv", index=False)
    print("Основная модель:", meta["selected"], "; порог операций:", round(meta["transaction_threshold"], 6))


if __name__ == "__main__":
    main()
