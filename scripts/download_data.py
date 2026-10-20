"""Получение опубликованного CSV из официального набора Kaggle."""

import shutil
import urllib.error
import urllib.request
import zipfile

from src.storage import ROOT


def main():
    destination = ROOT / "data/raw/HI-Small_Trans.csv"
    if destination.exists():
        print("Файл уже существует:", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = "https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv"
    temporary = destination.with_suffix(".download")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Hunterilka/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)

        if zipfile.is_zipfile(temporary):
            with zipfile.ZipFile(temporary) as archive:
                matching = [name for name in archive.namelist() if name.endswith("HI-Small_Trans.csv")]
                if len(matching) != 1:
                    raise ValueError("В архиве не найден единственный HI-Small_Trans.csv.")
                with archive.open(matching[0]) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
            temporary.unlink()
        else:
            with temporary.open("rb") as source:
                if not source.readline().startswith(b"Timestamp,From Bank,Account"):
                    raise ValueError("Источник вернул не CSV. Скачайте HI-Small_Trans.csv со страницы Kaggle.")
            temporary.replace(destination)
        print("Получен файл:", destination, "байт:", destination.stat().st_size)
    except (urllib.error.URLError, OSError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit("Загрузка не завершена: " + str(error))


if __name__ == "__main__":
    main()
