"""Обновляет установленную копию с резервной копией кода, сохраняя данные и модели."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1]
RETIRED = ["RUN_GUIDE.md", "docs/code_map.md", "docs/system_guide.md", "docs/validation.md"]


def apply(target):
    target = Path(target).expanduser().resolve()
    if target == SOURCE or not (target / "src/service.py").is_file():
        raise ValueError("Нужен путь к другой, уже установленной копии Hunterilka.")
    manifest = json.loads((SOURCE / "release_files.json").read_text(encoding="utf-8"))
    names = manifest["files"] + ["release_files.json"]
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts or not (SOURCE / name).is_file():
            raise ValueError("Некорректный файл обновления: " + name)
        if (target / name).resolve().is_relative_to(target) is False:
            raise ValueError("Путь выходит за пределы установленной копии: " + name)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = target / "private" / ("update-backup-" + stamp)
    backup.mkdir(parents=True)
    saved = []
    created = []
    try:
        for name in names:
            # Не заменяем локальный эксперимент и настройки его признаков.
            if (name.startswith(("data/", "artifacts/", "reports/")) or name == "config.json") and (target / name).exists():
                continue
            old, new = target / name, SOURCE / name
            if old.is_file():
                out = backup / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, out)
                saved.append(name)
            else:
                created.append(name)
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(new, old)
            if name == ".gitignore" and (backup / name).is_file():
                previous = (backup / name).read_text(encoding="utf-8")
                old.write_text(previous + "\n" + new.read_text(encoding="utf-8"), encoding="utf-8")
        for name in RETIRED:
            old = target / name
            if old.is_file():
                out = backup / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(old, out)
                saved.append(name)
        if (target / "reports/metrics.json").is_file():
            command = "from src.storage import ROOT,read_json; from src.report import write_report; write_report(read_json(ROOT/'reports/metrics.json'))"
            subprocess.run([sys.executable, "-c", command], cwd=target, check=True)
    except BaseException:
        for name in created:
            (target / name).unlink(missing_ok=True)
        for name in saved:
            shutil.copy2(backup / name, target / name)
        raise
    print("Обновлено:", target)
    print("Прежние исходники и учебные материалы:", backup)
    print("Модели, данные, .env, настройки признаков и рабочая БД сохранены. Повторное обучение не требуется.")
    print("Локальный порт PostgreSQL в конфигурации обновлён на 55432.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обновление кода без повторного обучения")
    parser.add_argument("--target", required=True, type=Path)
    apply(parser.parse_args().target)
