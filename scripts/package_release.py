"""Публикуется только явный список файлов согласованного полного запуска."""

import shutil
import zipfile
from pathlib import Path

import pandas as pd

from src.models import load_models
from src.data import validate_split
from src.review_queue import REVIEW_POLICY
from src.snapshot import WORKSPACE_FILES, WORKSPACE_VERSION
from src.storage import file_hash
from src.storage import ROOT, load_config, read_json


ROOT_FILES = ["README.md", "RUN_GUIDE.md", "NOTICE.md", "requirements.txt", "config.json", "app.py", ".gitignore"]
DOCS = ["architecture.md", "data_card.md", "traceability.md", "results.md", "assets/architecture.svg"]
REPORTS = ["metrics.json", "experiments.csv", "search_diagnostics.csv"]


def main():
    models = load_models(load_config())
    report = read_json(ROOT / "reports/metrics.json")
    validate_split(report["data"])
    demo = read_json(ROOT / "demo/demo_manifest.json")
    if report["data"].get("scope") != "full_source":
        raise ValueError("Публикация требует полного расчёта. Укороченная техническая проверка не подходит.")
    if report["model_id"] != models["meta"]["model_id"] or demo["model_id"] != report["model_id"]:
        raise ValueError("Модели, отчёт и карточки относятся к разным запускам. Повторите evaluate.")
    if report["data"]["events_sha256"] != models["meta"]["data_hash"]:
        raise ValueError("Отчёт относится к другому набору данных.")
    if report.get("review_policy", {}).get("version") != REVIEW_POLICY or demo.get("review_policy") != report["review_policy"]:
        raise ValueError("Обновите очереди и отчёт: python -m scripts.refresh_review.")
    workspace_path = ROOT / "demo/workspace_manifest.json"
    if demo.get("workspace_version") != WORKSPACE_VERSION or not workspace_path.exists():
        raise ValueError("Подготовьте окружение и обзор сети: python -m scripts.build_workspace.")
    workspace = read_json(workspace_path)
    if set(workspace["files"]) != set(WORKSPACE_FILES):
        raise ValueError("Состав файлов обзора отличается от разрешённого списка.")
    if workspace["model_id"] != demo["model_id"] or workspace["data_hash"] != models["meta"]["data_hash"]:
        raise ValueError("Обзор сети относится к другому запуску.")
    for name, checksum in workspace["files"].items():
        if file_hash(ROOT / "demo" / name) != checksum:
            raise ValueError("Файл обзора изменён: " + name)

    names = list(ROOT_FILES)
    names += ["docs/" + name for name in DOCS]
    names += [".github/workflows/check.yml", ".streamlit/config.toml"]
    for folder in ["src", "scripts", "tests"]:
        names += [str(path.relative_to(ROOT)) for path in sorted((ROOT / folder).glob("*.py"))]
    names += ["artifacts/" + name for name in models["meta"]["artifact_hashes"]]
    names += ["artifacts/model_meta.json", "artifacts/feature_manifest.json"]
    names += ["reports/" + name for name in REPORTS]
    names += ["demo/demo_manifest.json", "demo/DATA_NOTICE.md"]
    names += ["demo/workspace_manifest.json"] + ["demo/" + name for name in workspace["files"]]
    index_path = ROOT / "demo/cases/index.parquet"
    if demo["cases"]:
        index = pd.read_parquet(index_path)
        if len(index) != demo["cases"]:
            raise ValueError("Число карточек не совпадает с паспортом.")
        names += ["demo/cases/index.parquet"]
        names += ["demo/cases/" + identifier + ".json" for identifier in index.case_id]
    for name in names:
        if not (ROOT / name).is_file():
            raise ValueError("Не найден файл публикации: " + name)

    target = ROOT / "publication/hunterilka"
    if target.exists():
        raise ValueError("Папка publication/hunterilka уже существует. Переименуйте её перед новой сборкой.")
    target.mkdir(parents=True)
    for name in names:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    for folder in ["data/raw", "data/processed", "runs", "demo/cases"]:
        (target / folder).mkdir(parents=True, exist_ok=True)
        (target / folder / ".gitkeep").touch()
    archive = target.parent / "Hunterilka_GitHub_v1.1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                package.write(path, "hunterilka/" + str(path.relative_to(target)))
    print("Папка для GitHub:", target)
    print("Архив:", archive)


if __name__ == "__main__":
    main()
