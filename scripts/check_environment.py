"""Короткая проверка среды до чтения большого CSV."""

import argparse
import platform
import shutil
import subprocess
import sys

import pandas as pd
from catboost import CatBoostClassifier, CatBoostError
from catboost.utils import get_gpu_device_count


def main():
    parser = argparse.ArgumentParser(description="Проверка CPU или GPU до полного расчёта")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    args = parser.parse_args()
    print("Система:", platform.system(), "; Python:", platform.python_version())
    if sys.version_info[:2] != (3, 12):
        print("Проверенная версия проекта: Python 3.12.")
    parameters = {"task_type": args.device}
    if args.device == "GPU":
        if shutil.which("nvidia-smi"):
            subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=True)
        if get_gpu_device_count() < 1:
            raise SystemExit("CatBoost не видит GPU. Проверьте драйвер NVIDIA. Полный расчёт ещё не запускался.")
        parameters["devices"] = "0"
    sample = pd.DataFrame({"amount": [1, 2, 3, 4, 20, 30, 40, 50], "format": ["A", "B"] * 4})
    model = CatBoostClassifier(iterations=2, depth=2, verbose=False, allow_writing_files=False, **parameters)
    try:
        model.fit(sample, [0, 0, 0, 0, 1, 1, 1, 1], cat_features=["format"])
    except CatBoostError as error:
        raise SystemExit("Проверка " + args.device + " не пройдена. Полный расчёт не начинался.\n" + str(error))
    print("Проверка обучения пройдена:", args.device)


if __name__ == "__main__":
    main()
