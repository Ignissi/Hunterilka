"""Выгружает случаи, версии и журнал решений. Это демонстрация, не резервная копия потока."""
import gzip
import json

from src import db
from src.storage import ROOT, write_json


def export_to(conn, folder):
    folder.mkdir(parents=True, exist_ok=True)
    # Один согласованный снимок, даже если сервис продолжает работу.
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        cases = [db.case_from_row(row) for row in conn.execute("SELECT * FROM cases ORDER BY detected_at, case_id").fetchall()]
        versions = [db.case_from_row(row) for row in conn.execute("SELECT * FROM case_versions ORDER BY case_id, version").fetchall()]
        reviews = conn.execute("SELECT case_id, reviewed_at, status, comment, analyst FROM reviews ORDER BY review_id").fetchall()
        progress = db.stream_state(conn)
    transactions = {}
    for case in cases:
        for event in case["details"]["events"] + case["details"]["context"]:
            transactions[event["transaction_id"]] = event
    for name, rows in [("cases", cases), ("transactions", list(transactions.values())), ("versions", versions)]:
        with gzip.open(folder / (name + ".jsonl.gz"), "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in reviews:
        row["reviewed_at"] = row["reviewed_at"].isoformat()
    write_json(folder / "reviews.json", reviews)
    write_json(folder / "demo_manifest.json", {
        "format": "2.1", "cases": len(cases), "transactions": len(transactions), "versions": len(versions),
        "reviews": len(reviews), "source_state": progress,
        "data_notice": "Демонстрационная выборка. Не содержит полного источника для продолжения потока.",
    })
    print("Выгружено случаев:", len(cases), "; версий:", len(versions), "; решений:", len(reviews))


def main():
    with db.connect() as conn:
        export_to(conn, ROOT / "demo")


if __name__ == "__main__":
    main()
