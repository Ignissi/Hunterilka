"""Полный цикл обучения одной командой: подготовка -> признаки -> обучение -> оценка."""

import argparse
import subprocess
import sys
from pathlib import Path

from src.storage import ROOT


def main():
    parser = argparse.ArgumentParser(description="Полный расчёт Hunterилка")
    parser.add_argument("--input", default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--limit", type=int, default=0, help="0: весь файл; N: первые N строк для проверки")
    parser.add_argument("--from-stage", choices=["prepare", "features", "train", "evaluate"], default="prepare")
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    if args.from_stage == "prepare" and not source.is_file():
        raise SystemExit("Не найден CSV: " + str(source))

    stages = [
        ("prepare", ["scripts.prepare_data", "--input", str(source), "--limit", str(args.limit)]),
        ("features", ["scripts.build_features"]),
        ("train", ["scripts.train", "--device", args.device]),
        ("evaluate", ["scripts.evaluate"]),
    ]
    names = [name for name, _ in stages]
    for name, arguments in stages[names.index(args.from_stage):]:
        print("\n=== Этап:", name, "===", flush=True)
        subprocess.run([sys.executable, "-m", *arguments], cwd=ROOT, check=True)

    print("\nГотово. Дальше: python -m scripts.run_stream --max-groups 2000, затем python -m streamlit run app.py")


if __name__ == "__main__":
    main()
