"""Четыре прозрачных эксперимента CatBoost и независимый Isolation Forest."""

import time
import argparse

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from src.features import CATEGORICAL, GRAPH_FEATURES
from src.metrics import budget_threshold, top_k_metrics
from src.storage import ROOT, file_hash, fingerprint, load_config, logic_hash, read_json, write_json


def fit_catboost(train, validation, names, weighted, config, device="CPU", devices="0"):
    hardware = {"task_type": device}
    if device == "GPU":
        hardware["devices"] = devices
    model = CatBoostClassifier(
        iterations=config["iterations"], depth=config["depth"],
        learning_rate=config["learning_rate"], loss_function="Logloss",
        random_seed=config["seed"], thread_count=config["thread_count"],
        auto_class_weights="Balanced" if weighted else None,
        allow_writing_files=False, verbose=False, **hardware,
    )
    model.fit(
        train[names], train.is_laundering,
        cat_features=CATEGORICAL,
        eval_set=(validation[names], validation.is_laundering),
        early_stopping_rounds=60,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Обучение CatBoost и Isolation Forest")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="CPU")
    parser.add_argument("--devices", default="0")
    args = parser.parse_args()
    config = load_config()
    folder = ROOT / "artifacts"
    manifest = read_json(folder / "feature_manifest.json")
    if manifest["config_hash"] != fingerprint(config):
        raise ValueError("Сначала пересчитайте признаки для новых настроек.")
    if manifest["logic_hash"] != logic_hash():
        raise ValueError("Расчётный код изменился. Сначала пересчитайте признаки.")

    features = pd.read_parquet(folder / "features.parquet")
    structures = pd.read_parquet(folder / "structures.parquet")
    train = features.loc[features.split == "train"]
    validation = features.loc[features.split == "validation"]
    for title, part in [("train", train), ("validation", validation)]:
        if part.is_laundering.nunique() < 2:
            raise ValueError(title + ": нужны оба класса. Расширьте хронологический профиль.")

    excluded = {"transaction_id", "timestamp", "is_laundering", "split"}
    full_names = [name for name in features.columns if name not in excluded]
    basic_names = [name for name in full_names if name not in GRAPH_FEATURES]
    experiments = []

    # Во всех четырёх опытах совпадают строки и бюджет настройки.
    for variant, names in [("A", basic_names), ("B", full_names)]:
        for weighted in [False, True]:
            started = time.perf_counter()
            model = fit_catboost(train, validation, names, weighted, config, args.device, args.devices)
            scores = model.predict_proba(validation[names])[:, 1]
            name = variant + ("_balanced" if weighted else "_plain")
            filename = name + ".cbm"
            model.save_model(str(folder / filename))
            row = {
                "name": name, "variant": variant, "weighted": weighted,
                "model_file": filename, "features": names,
                "validation_ap": float(average_precision_score(validation.is_laundering, scores)),
                "threshold": budget_threshold(scores, min(config["review_k"], len(scores)) / len(scores)),
                "validation_top_k": top_k_metrics(validation.is_laundering, scores),
                "trees": model.tree_count_, "seconds": time.perf_counter() - started,
            }
            experiments.append(row)
            print(name, "validation AP:", round(row["validation_ap"], 6), flush=True)

    # При одинаковых 100 проверках выбираем больше находок; AP разрешает равенство.
    selected = max(experiments, key=lambda row: (
        row["validation_top_k"]["100"]["found"], row["validation_ap"],
    ))
    structure_names = [name for name in structures.columns if name not in {"candidate_id", "case_key", "timestamp", "family", "split"}]
    structure_train = structures.loc[structures.split == "train", structure_names]
    structure_validation = structures.loc[structures.split == "validation"].drop_duplicates("case_key", keep="last")[structure_names]
    if len(structure_train) < 20 or len(structure_validation) < 5:
        raise ValueError("Недостаточно структур для второго уровня. Расширьте профиль.")

    started = time.perf_counter()
    isolation = IsolationForest(
        n_estimators=200, max_samples=min(256, len(structure_train)),
        contamination="auto", random_state=config["seed"], n_jobs=config["thread_count"],
    )
    isolation.fit(structure_train)
    train_anomaly = -isolation.score_samples(structure_train)
    validation_anomaly = -isolation.score_samples(structure_validation)
    joblib.dump(isolation, folder / "isolation_forest.joblib")
    np.save(folder / "structure_reference.npy", np.sort(train_anomaly))

    meta = {
        "version": "1.1", "training_device": args.device, "devices": args.devices,
        "selection_rule": "validation: found@100, затем AP",
        "validation_review_k": 100, "scope": manifest.get("scope", "unknown"), "config_hash": fingerprint(config),
        "data_hash": manifest["data_hash"], "feature_version": config["feature_version"], "logic_hash": logic_hash(),
        "selected": selected["name"], "model_file": selected["model_file"],
        "features": selected["features"], "structure_features": structure_names,
        "transaction_threshold": selected["threshold"],
        "structure_threshold": budget_threshold(validation_anomaly, min(config["review_k"], len(validation_anomaly)) / len(validation_anomaly)),
        "structure_train_rows": len(structure_train),
        "structure_seconds": time.perf_counter() - started,
        "structure_reference": "structure_reference.npy",
        "structure_medians": structure_train.median().to_dict(),
        "structure_q95": structure_train.quantile(0.95).to_dict(),
        "experiments": experiments,
        "artifact_hashes": {
            row["model_file"]: file_hash(folder / row["model_file"]) for row in experiments
        },
    }
    for name in ["isolation_forest.joblib", "structure_reference.npy"]:
        meta["artifact_hashes"][name] = file_hash(folder / name)
    meta["model_id"] = fingerprint(meta)[:24]
    write_json(folder / "model_meta.json", meta)
    pd.DataFrame([{k: v for k, v in row.items() if k not in {"features", "validation_top_k"}} for row in experiments]).to_csv(ROOT / "reports/experiments.csv", index=False)
    print("Основная модель:", meta["selected"], "Структур train:", len(structure_train))


if __name__ == "__main__":
    main()
