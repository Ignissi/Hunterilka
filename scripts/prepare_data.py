"""Подготовка опубликованного набора без искусственных обучающих строк."""

import argparse

from src.data import prepare_data, validate_split


def main():
    parser = argparse.ArgumentParser(description="Подготовка IBM HI-Small")
    parser.add_argument("--input", required=True, help="Путь к HI-Small_Trans.csv")
    parser.add_argument("--limit", type=int, default=0, help="0 означает весь набор")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("Лимит не может быть отрицательным.")
    manifest = prepare_data(args.input, args.output, args.limit)
    print("Подготовка завершена:", manifest["rows"], "операций")
    print("Дней истории:", round(manifest["history_days"], 2))
    print("Временные части:", manifest["parts"])
    print("Границы validation и test:", manifest["train_end"], manifest["validation_end"])
    validate_split(manifest)
    print("Хронологическое разделение проверено.")


if __name__ == "__main__":
    main()
