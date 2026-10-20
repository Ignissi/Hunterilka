"""Загрузка собственных артефактов с проверкой конфигурации и состава признаков."""

import joblib
from catboost import CatBoostClassifier

from src.storage import ROOT, file_hash, fingerprint, logic_hash, read_json


def load_models(config, folder=None):
    folder = ROOT / "artifacts" if folder is None else folder
    meta = read_json(folder / "model_meta.json")
    if fingerprint(config) != meta["config_hash"]:
        raise ValueError("Настройки изменились. Повторите расчёт признаков и обучение.")
    if logic_hash() != meta["logic_hash"]:
        raise ValueError("Расчётный код изменился. Повторите расчёт признаков и обучение.")

    for name, expected in meta["artifact_hashes"].items():
        if file_hash(folder / name) != expected:
            raise ValueError("Изменён артефакт модели: " + name)

    transaction = CatBoostClassifier()
    transaction.load_model(str(folder / meta["model_file"]))
    structure = joblib.load(folder / "isolation_forest.joblib")
    if list(transaction.feature_names_) != meta["features"]:
        raise ValueError("Порядок признаков не совпадает с паспортом модели.")
    return {"transaction": transaction, "structure": structure, "meta": meta}
