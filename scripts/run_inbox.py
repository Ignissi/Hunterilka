"""Новые переводы из файлового источника; процесс ожидает последующие минуты."""

import argparse
import math
import threading
from pathlib import Path

from src import db
from src.contracts import feature_contract
from src.incoming import consume_batch, minute_time
from src.models import load_models
from src.service import Service
from src.storage import load_config


def main():
    parser = argparse.ArgumentParser(description="Обработка новых операций из входного каталога")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Постоянное имя источника")
    parser.add_argument("--score-from", help="Начало оценки в UTC; предыдущие минуты используются как история")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="Обработать имеющиеся файлы и завершить процесс")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0 or not args.source.strip():
        parser.error("Интервал должен быть положительным, имя источника — непустым.")
    config = load_config()
    models = load_models(config=config)
    args.input_dir.mkdir(parents=True, exist_ok=True)
    with db.connect() as conn:
        db.acquire_writer(conn)
        db.init_schema(conn)
        previous = db.stream_state(conn)
        start = args.score_from or (previous["score_from"] if previous else None)
        if start is None:
            parser.error("Для нового источника задайте --score-from.")
        start = minute_time(start)
        db.ensure_stream(conn, "incoming:" + args.source.strip(), models["meta"]["model_id"],
                         feature_contract(config), start, kind="inbox")
        service = Service(config, models, conn)
        service.restore()
        wait = threading.Event()
        print("Источник:", args.source, "· начало оценки:", start, flush=True)
        while True:
            for path in sorted(args.input_dir.glob("*.json")):
                result = consume_batch(path, conn, service)
                print(path.name, "·", result, flush=True)
            if args.once:
                break
            wait.wait(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Обработка остановлена; сохранённый прогресс будет восстановлен при следующем запуске.")
