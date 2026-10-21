"""Проверка окружения и двух баз до длительного расчёта."""
import importlib
import os
import sys

from src import db


def main():
    print("Python:", sys.version.split()[0])
    for name in ["pandas", "numpy", "catboost", "sklearn", "networkx", "pyarrow", "psycopg", "streamlit"]:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "OK"))
    urls = [("Рабочая база", None), ("Тестовая база", os.environ.get(
        "HUNTERILKA_TEST_DB", "postgresql://hunterilka:hunterilka@localhost:55432/hunterilka_v2_test"))]
    for title, url in urls:
        with db.connect(url) as conn:
            db.init_schema(conn)
            print(title + ":", conn.info.dbname, "на", str(conn.info.host) + ":" + str(conn.info.port))
            print(conn.execute("SELECT version() AS version").fetchone()["version"])
    print("Окружение готово. Следующий шаг: тесты, затем run_pipeline.")


if __name__ == "__main__":
    main()
