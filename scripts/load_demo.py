"""Загрузка демонстрации с сохранением версий, статусов и исходных дат решений."""
import argparse
import gzip
import json

from psycopg.types.json import Jsonb

from src import db
from src.storage import ROOT, read_json


def read_lines(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_from(conn, folder, replace=False):
    # Сначала читаем все файлы. Отсутствующий архив не должен очищать рабочую базу.
    manifest = read_json(folder / "demo_manifest.json")
    if manifest.get("format") != "2.1":
        raise ValueError("Нужна демонстрационная выгрузка формата 2.1.")
    cases = read_lines(folder / "cases.jsonl.gz")
    events = read_lines(folder / "transactions.jsonl.gz")
    versions = read_lines(folder / "versions.jsonl.gz")
    reviews = read_json(folder / "reviews.json")
    for name, rows in [("cases", cases), ("transactions", events), ("versions", versions), ("reviews", reviews)]:
        if len(rows) != manifest[name]:
            raise ValueError("Неполная выгрузка: " + name)
    progress = manifest.get("source_state")
    if progress is None:
        raise ValueError("Выгрузка не содержит описание исходного эксперимента.")
    db.acquire_writer(conn)
    db.init_schema(conn)
    if not replace and (db.count_transactions(conn) or db.count_cases(conn) or db.stream_state(conn)):
        raise ValueError("База непустая. Для явной замены используйте --replace или другую базу.")
    with conn.transaction():
        db.reset(conn)
        db.ensure_stream(conn, progress["dataset_id"], progress["model_id"], progress["feature_contract"], progress["score_from"], kind="demo")
        db.set_progress(conn, progress["last_time"], history_complete=True)
        db.insert_transactions(conn, events)
        columns = db.CASE_COLUMNS + ["details"]
        sql = "INSERT INTO cases (" + ", ".join(columns) + ") VALUES (" + ", ".join(["%s"] * len(columns)) + ")"
        for case in cases:
            conn.execute(sql, [Jsonb(case[name]) if name == "details" else case[name] for name in columns])
        for row in versions:
            conn.execute("INSERT INTO case_versions (case_id, version, detected_at, operation_score, anomaly_score, details) "
                         "VALUES (%s, %s, %s, %s, %s, %s)",
                         (row["case_id"], row["version"], row["detected_at"], row["operation_score"], row["anomaly_score"], Jsonb(row["details"])))
        for row in reviews:
            conn.execute("INSERT INTO reviews (case_id, reviewed_at, status, comment, analyst) VALUES (%s, %s, %s, %s, %s)",
                         (row["case_id"], row["reviewed_at"], row["status"], row["comment"], row["analyst"]))
    print("Загружено случаев:", len(cases), "; версий:", len(versions), "; решений:", len(reviews))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replace", action="store_true", help="Заменить содержимое выбранной БД демонстрацией")
    args = parser.parse_args()
    with db.connect() as conn:
        load_from(conn, ROOT / "demo", replace=args.replace)


if __name__ == "__main__":
    main()
