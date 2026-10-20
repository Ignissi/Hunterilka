"""Небольшие операции с файлами и паспортами запуска."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def fingerprint(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config():
    return read_json(ROOT / "config.json")


def logic_hash():
    # Изменение расчётного кода требует пересборки признаков и моделей.
    names = ["data.py", "graph.py", "context.py", "features.py", "patterns.py", "pipeline.py"]
    sources = {}
    for name in names:
        text = (ROOT / "src" / name).read_text(encoding="utf-8")
        sources[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return fingerprint(sources)
