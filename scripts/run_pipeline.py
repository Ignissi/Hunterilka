"""Последовательный запуск с журналом и остановкой при первой ошибке."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.storage import ROOT


def main():
    parser = argparse.ArgumentParser(description="Полный расчёт Hunterилка")
    parser.add_argument("--input", default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--limit", type=int, default=0, help="0: весь файл; положительное число: только техническая проверка")
    parser.add_argument("--from-stage", choices=["prepare", "features", "train", "evaluate", "workspace"], default="prepare")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit("Не найден CSV: " + str(source))
    (ROOT / "reports").mkdir(exist_ok=True)
    stages = [
        ("environment", ["scripts.check_environment", "--device", args.device]),
        ("prepare", ["scripts.prepare_data", "--input", str(source), "--limit", str(args.limit)]),
        ("features", ["scripts.build_features"]),
        ("train", ["scripts.train", "--device", args.device]),
        ("evaluate", ["scripts.evaluate"]),
        ("workspace", ["scripts.build_workspace"]),
    ]
    start = [name for name, _ in stages].index(args.from_stage)
    for index, (name, arguments) in enumerate(stages):
        if index != 0 and index < start:
            continue
        print("\nЭтап:", name, flush=True)
        with (ROOT / "reports" / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", *arguments], cwd=ROOT,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise SystemExit("Расчёт остановлен. Для повторного запуска используйте --from-stage " + name)
        if code:
            raise SystemExit("Этап остановлен: " + name + ". Подробности в reports/" + name + ".log")
    print("\nГотово. Откройте docs/results.md и запустите: python -m streamlit run app.py")


if __name__ == "__main__":
    main()
