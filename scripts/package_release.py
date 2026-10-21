"""Собирает комплект программы и локально обученных моделей без рабочих данных."""

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.models import load_models
from src.storage import ROOT, file_hash, read_json


def package(output):
    load_models()  # Проверяет контракт признаков, выбранную модель и Isolation Forest.
    manifest = read_json(ROOT / "release_files.json")
    meta = read_json(ROOT / "artifacts" / "model_meta.json")
    names = set(manifest["files"]) | {"release_files.json", "artifacts/model_meta.json"}
    for name, expected_hash in meta["artifact_hashes"].items():
        if Path(name).name != name:
            raise ValueError("В метаданных модели недопустимый путь.")
        path = ROOT / "artifacts" / name
        if file_hash(path) != expected_hash:
            raise ValueError("Файл модели изменён: " + name)
        names.add("artifacts/" + name)
    for name in ["reports/metrics.json", "reports/experiments.csv"]:
        if (ROOT / name).is_file():
            names.add(name)
    output = Path(output).expanduser().resolve()
    if output.suffix != ".zip" or output in {(ROOT / name).resolve() for name in names}:
        raise ValueError("Укажите отдельный выходной ZIP-файл.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name in sorted(names):
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Не найден файл поставки: " + name)
            archive.write(path, "hunterilka-v2-ready/" + name)
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Комплект программы с обученными моделями")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "hunterilka-v2.2-local.zip")
    args = parser.parse_args()
    print("Комплект создан:", package(args.output))


if __name__ == "__main__":
    main()
