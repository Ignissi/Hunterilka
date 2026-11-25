"""Проверяет новый и повторный запуск с настоящим Docker в изолированном Compose-проекте."""
import json
import lzma
from pathlib import Path
import subprocess
import tempfile
import uuid

from scripts.start_docker import docker_compose, prepare_database
from src import db


def main():
    base = docker_compose()
    config = json.loads(subprocess.check_output([*base, "config", "--format", "json"]))
    project = "hunterilka-test-" + uuid.uuid4().hex[:12]
    service = config["services"]["postgres"]
    service.pop("container_name", None)
    service["ports"] = ["127.0.0.1::5432"]
    config["name"] = project
    config["volumes"]["hunterilka_pg"] = {}
    config.pop("networks", None)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "compose.json"
        manifest.write_text(json.dumps(config), encoding="utf-8")
        compose = [base[0], "compose", "-p", project, "-f", str(manifest)]
        snapshot = root / "demo.sql.xz"
        insert = ("INSERT INTO transactions (transaction_id,ts,source_bank,source_account,target_bank,target_account,"
                  "amount_paid,payment_currency,amount_received,receiving_currency,payment_format) VALUES "
                  "(1,'2026-01-01','A','1','B','2',37,'EUR',37,'EUR','Wire');\n").encode()
        try:
            url = prepare_database(compose, snapshot)
            snapshot.write_bytes(lzma.compress(insert + b"SELECT 1/0;\n"))
            try:
                prepare_database(compose, snapshot)
            except RuntimeError:
                pass
            with db.connect(url) as conn:
                assert db.count_transactions(conn) == 0, "Incomplete import was not rolled back"
            snapshot.write_bytes(lzma.compress(insert))
            prepare_database(compose, snapshot)
            with db.connect(url) as conn:
                conn.execute("UPDATE transactions SET payment_format='ACH' WHERE transaction_id=1")
            subprocess.run([*compose, "stop", "postgres"], check=True)
            snapshot.write_bytes(b"already initialized: never open this file")
            prepare_database(compose, snapshot)
            with db.connect(url) as conn:
                rows = conn.execute("SELECT payment_format FROM transactions").fetchall()
                assert rows == [{"payment_format": "ACH"}], rows
            print("Docker volume survived restart; existing data retained; failed import rolled back.")
        finally:
            subprocess.run([*compose, "down", "--volumes"], check=True)


if __name__ == "__main__":
    main()
