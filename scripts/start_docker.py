"""Запуск рабочего места с PostgreSQL 16 из docker-compose.yml."""
import argparse
import lzma
import os
from pathlib import Path
import shutil
import site
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
portable_libs = Path(sys.executable).parent / "Lib" / "site-packages"
if portable_libs.is_dir():
    site.addsitedir(str(portable_libs))

import psycopg
from src import db

# Даже незаконченный источник или старый журнал означают, что база уже используется.
HAS_DATA = """SELECT EXISTS (SELECT 1 FROM public.stream_state)
                  OR EXISTS (SELECT 1 FROM public.transactions)
                  OR EXISTS (SELECT 1 FROM public.cases)
                  OR EXISTS (SELECT 1 FROM public.case_versions)
                  OR EXISTS (SELECT 1 FROM public.reviews) AS present"""


def docker_compose():
    install = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Docker" / "Docker"
    docker = shutil.which("docker")
    if not docker and (install / "resources/bin/docker.exe").is_file():
        docker = str(install / "resources/bin/docker.exe")
    if not docker:
        raise RuntimeError("Установите Docker Desktop с Linux-контейнерами и повторите запуск.")

    def engine():
        try:
            result = subprocess.run([docker, "info", "--format", "{{.OSType}}"], capture_output=True, timeout=5)
        except subprocess.TimeoutExpired:
            return ""
        return result.stdout.decode().strip() if result.returncode == 0 else ""

    kind = engine()
    desktop = install / "Docker Desktop.exe"
    if not kind and os.name == "nt" and desktop.is_file():
        print("Открываю Docker Desktop и жду готовности.", flush=True)
        subprocess.Popen([str(desktop)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 120
        while not kind and time.monotonic() < deadline:
            time.sleep(2)
            kind = engine()
    if not kind:
        raise RuntimeError("Docker не ответил. Откройте Docker Desktop и дождитесь готовности движка.")
    if kind != "linux":
        raise RuntimeError("Для PostgreSQL нужен режим Linux containers в Docker Desktop.")
    return [docker, "compose", "--project-name", "hunterilka-v2", "--file", str(ROOT / "docker-compose.yml")]


def prepare_database(compose, snapshot):
    print("Запускаю PostgreSQL в Docker. Первый запуск может загрузить образ postgres:16.", flush=True)
    subprocess.run([*compose, "up", "--detach", "--no-recreate", "--wait", "--wait-timeout", "120", "postgres"],
                   check=True)
    addresses = subprocess.check_output([*compose, "port", "postgres", "5432"], text=True).splitlines()
    if not addresses:
        raise RuntimeError("Docker не сообщил локальный порт PostgreSQL.")
    port = int(addresses[0].rsplit(":", 1)[1])
    url = f"postgresql://hunterilka:hunterilka@127.0.0.1:{port}/hunterilka_v2"
    with db.connect(url) as conn:
        # Два одновременных первых запуска используют одну инициализацию и один импорт.
        conn.execute("SELECT pg_advisory_lock(7413022)")
        db.init_schema(conn)
        if conn.execute(HAS_DATA).fetchone()["present"]:
            print("Открываю существующую базу Docker.", flush=True)
        elif snapshot.is_file():
            db.acquire_writer(conn)
            print("База Docker пуста. Загружаю демонстрацию один раз.", flush=True)
            restore_demo(compose, snapshot)
            print("Подготавливаю статистику PostgreSQL для быстрых запросов.", flush=True)
            conn.execute("VACUUM (ANALYZE) public.transactions")
        else:
            print("База Docker готова к приёму переводов.", flush=True)
    return url


def restore_demo(compose, snapshot):
    command = [*compose, "exec", "-T", "postgres", "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
               "-U", "hunterilka", "-d", "hunterilka_v2", "-f", "-"]
    check_empty = ("LOCK TABLE public.stream_state, public.transactions, public.cases, "
                   "public.case_versions, public.reviews IN SHARE ROW EXCLUSIVE MODE;\n"
                   f"DO $$ BEGIN IF ({HAS_DATA}) THEN RAISE EXCEPTION 'Database already contains data'; "
                   "END IF; END $$;\n")
    # COMMIT передаётся только после полной распаковки, чтобы повреждённая копия откатывалась.
    with lzma.open(snapshot, "rb") as source, subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        try:
            proc.stdin.write(b"BEGIN;\n" + check_empty.encode("ascii"))
            shutil.copyfileobj(source, proc.stdin, length=1024 * 1024)
            proc.stdin.write(b"\nCOMMIT;\n")
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("PostgreSQL прервал загрузку демонстрации. Изменения отменены.")
        finally:
            if not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass


def interface_port():
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 8501))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Проверить базу Docker без открытия интерфейса")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["HUNTERILKA_DB"] = prepare_database(docker_compose(), ROOT / "demo" / "docker-data.sql.xz")
    if args.check:
        return
    from streamlit.web import cli

    port = interface_port()
    print(f"Хантерилка · Docker. Откроется http://127.0.0.1:{port}. Оставьте это окно открытым.", flush=True)
    sys.argv = ["streamlit", "run", str(ROOT / "app.py"), "--server.address=127.0.0.1",
                f"--server.port={port}", "--server.headless=false", "--server.fileWatcherType=none",
                "--browser.gatherUsageStats=false"]
    cli.main()


if __name__ == "__main__":
    try:
        main()
    except (OSError, EOFError, lzma.LZMAError, ValueError, RuntimeError, subprocess.SubprocessError, psycopg.Error) as error:
        print(f"Запуск остановлен. {error}", file=sys.stderr)
        sys.exit(1)
