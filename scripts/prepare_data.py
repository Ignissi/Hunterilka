"""Шаг 1. Исходный IBM CSV -> data/events.parquet, data/labels.parquet, data/manifest.json."""

import argparse

from src.data import prepare_data
from src.storage import ROOT


def main():
    parser = argparse.ArgumentParser(description="Подготовка IBM HI-Small")
    parser.add_argument("--input", required=True, help="Путь к HI-Small_Trans.csv")
    parser.add_argument("--limit", type=int, default=0, help="0: весь файл; N: первые N строк для быстрой проверки")
    args = parser.parse_args()

    manifest = prepare_data(args.input, ROOT / "data", args.limit)
    print("Операций:", manifest["rows"], "; положительных:", manifest["positive"])
    print("Дней истории:", round(manifest["history_days"], 2))
    print("Части:", manifest["parts"])
    print("Границы validation и test:", manifest["train_end"], manifest["validation_end"])


if __name__ == "__main__":
    main()
