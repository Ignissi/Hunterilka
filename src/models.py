"""Загрузка обученных моделей и объяснение оценки операции через SHAP."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.features import CATEGORICAL
from src.contracts import feature_contract
from src.storage import ROOT, file_hash, load_config, read_json


def load_models(folder=None, config=None):
    folder = ROOT / "artifacts" if folder is None else Path(folder)
    meta = read_json(folder / "model_meta.json")
    config = load_config() if config is None else config
    if meta.get("feature_contract") != feature_contract(config):
        raise ValueError("Модель несовместима с версией или настройками признаков. Повторите build_features и train.")
    for name in [meta["model_file"], "isolation_forest.joblib"]:
        if file_hash(folder / name) != meta.get("artifact_hashes", {}).get(name):
            raise ValueError("Артефакт изменён или взят из другого запуска: " + name)

    transaction = CatBoostClassifier()
    transaction.load_model(str(folder / meta["model_file"]))
    if list(transaction.feature_names_) != meta["features"]:
        raise ValueError("Список признаков модели не совпадает с model_meta.json.")

    structure = joblib.load(folder / "isolation_forest.joblib")
    if list(structure.feature_names_in_) != meta["structure_features"]:
        raise ValueError("Признаки Isolation Forest не совпадают с model_meta.json.")
    return {"transaction": transaction, "structure": structure, "meta": meta}


def explain_transaction(model, vector, feature_names):
    """SHAP-вклады признаков для сохранённого вектора одной операции."""
    frame = pd.DataFrame([vector])[feature_names]
    for name in feature_names:
        if name not in CATEGORICAL:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    pool = Pool(frame, cat_features=CATEGORICAL)
    shap = model.get_feature_importance(pool, type="ShapValues")[0]
    raw = float(model.predict(pool, prediction_type="RawFormulaVal")[0])

    table = pd.DataFrame({"Признак": feature_names, "Вклад": shap[:-1]})
    table = table.iloc[np.argsort(-np.abs(table["Вклад"].to_numpy()), kind="stable")]
    return {
        "table": table.reset_index(drop=True),
        "base": float(shap[-1]),
        "raw": raw,
        "score": float(1 / (1 + np.exp(-np.clip(raw, -700, 700)))),
    }
