"""Симулятор источника: полные группы времени, устойчивый прогрев и продолжение."""

import argparse
import threading
import time
from datetime import timedelta

from src import db
from src.contracts import feature_contract, validate_events
from src.graph import parse_time
from src.models import load_models
from src.service import Service
from src.storage import ROOT, load_config, load_table, read_json


HISTORY_BATCH = 100000


def table_groups(events):
    """В словари превращается одна минута, а не вся история сразу."""
    for timestamp, table in events.groupby("timestamp", sort=False):
        yield timestamp, table.to_dict("records")


def load_history(conn, service, events):
    batch = []
    loaded = 0
    for _, group in table_groups(events):
        batch.extend(group)
        if len(batch) >= HISTORY_BATCH:
            service.ingest_history(batch)
            loaded += len(batch)
            print("Прогрев:", loaded, "/", len(events), flush=True)
            batch = []
    if batch:
        service.ingest_history(batch)
        loaded += len(batch)
    with conn.transaction():
        db.set_progress(conn, db.stream_state(conn)["last_time"], history_complete=True)
    print("Прогрев завершён; добавлено операций:", loaded, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Поток операций через сервис анализа")
    parser.add_argument("--start", default=None, help="Начало оценки; по умолчанию начало test или прежний старт")
    parser.add_argument("--speed", type=float, default=0, help="Пауза между минутами в секундах; 0: максимально быстро")
    parser.add_argument("--max-groups", type=int, default=0, help="Остановиться после N минут; 0: до конца")
    parser.add_argument("--reset", action="store_true", help="Удалить рабочие операции, случаи и решения из выбранной БД")
    args = parser.parse_args()
    if args.speed < 0 or args.max_groups < 0:
        parser.error("speed и max-groups должны быть неотрицательными")

    config = load_config()
    models = load_models(config=config)
    manifest = read_json(ROOT / "data" / "manifest.json")
    validate_events(ROOT / "data" / "events.parquet", manifest)
    if models["meta"]["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("Симулятор и модель относятся к разным экспериментам.")
    events = load_table(ROOT / "data" / "events.parquet")

    with db.connect() as conn:
        db.acquire_writer(conn)
        db.init_schema(conn)
        if args.reset:
            db.reset(conn)
        previous = db.stream_state(conn)
        start = args.start or (previous["score_from"] if previous else manifest["validation_end"])
        start = parse_time(start).strftime("%Y-%m-%dT%H:%M:%S")
        if start < manifest["validation_end"] or start > manifest["last_time"]:
            raise ValueError("Начало оценки должно находиться в test-периоде.")
        db.ensure_stream(conn, manifest["dataset_id"], models["meta"]["model_id"], feature_contract(config), start)
        service = Service(config, models, conn)
        service.restore()
        progress = db.stream_state(conn)
        if progress["last_time"]:
            print("Восстановлено:", len(service.state.events), "операций и", len(service.scores), "оценок; курсор", progress["last_time"], flush=True)

        if not progress["history_complete"]:
            lower = (parse_time(start) - timedelta(days=config["window_days"])).isoformat()
            history = events.loc[(events.timestamp >= lower) & (events.timestamp < start)]
            if progress["last_time"]:
                history = history.loc[history.timestamp > progress["last_time"]]
            load_history(conn, service, history)

        last_time = db.stream_state(conn)["last_time"]
        live = events.loc[events.timestamp >= start]
        if last_time:
            live = live.loc[live.timestamp > last_time]
        print("Минут впереди:", live.timestamp.nunique(), flush=True)
        started = time.perf_counter()
        pause = threading.Event()
        for number, (timestamp, group) in enumerate(table_groups(live), start=1):
            info = service.process(group)
            if number % 100 == 0 or info["saved_cases"]:
                print(f"{timestamp} · операций {len(group)} · случаев {info['saved_cases']} · "
                      f"граф {len(service.state.events)} · {time.perf_counter() - started:.0f} с", flush=True)
            if args.max_groups and number >= args.max_groups:
                break
            if args.speed:
                pause.wait(args.speed)
    print("Готово. Создано или обновлено случаев:", service.saved_cases)


if __name__ == "__main__":
    main()
