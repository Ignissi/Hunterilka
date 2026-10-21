"""Короткий контракт данных и признаков между подготовкой, обучением и работой."""

from src.storage import file_hash


FEATURE_VERSION = "hunterilka-features-2.1"
FEATURE_SETTINGS = [
    "window_days", "motif_window_hours", "fast_scan_budget", "path_scan_limit",
    "center_scan_limit", "composition_scan_limit", "paths_per_seed", "cycle_max_length",
    "relay_max_edges", "star_min_neighbors", "star_max_events", "candidates_per_group",
]


def feature_contract(config):
    if config["window_days"] < 1:
        raise ValueError("Окно графа должно быть не меньше суток: есть признаки за 24 часа.")
    if config["motif_window_hours"] > 24 * config["window_days"]:
        raise ValueError("Окно поиска структур не должно превышать окно графа.")
    return {"version": FEATURE_VERSION, "settings": {key: config[key] for key in FEATURE_SETTINGS}}


def validate_feature_tables(config, manifest, feature_manifest):
    if not manifest.get("dataset_id"):
        raise ValueError("Данные подготовлены старой версией. Запустите run_pipeline с этапа prepare.")
    if feature_manifest.get("dataset_id") != manifest["dataset_id"]:
        raise ValueError("Признаки относятся к другим данным. Повторите build_features.")
    if feature_manifest.get("feature_contract") != feature_contract(config):
        raise ValueError("Изменились версия или настройки признаков. Повторите build_features и train.")


def validate_events(path, manifest):
    if not manifest.get("dataset_id") or file_hash(path) != manifest.get("events_sha256"):
        raise ValueError("Файл операций не соответствует manifest.json. Повторите prepare_data.")
