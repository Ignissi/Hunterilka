# Установка и локальный запуск

## Готовый комплект с моделями

Требуются Linux, Python 3.12 и Docker Engine с Compose plugin. PostgreSQL 16 запускается в Docker; приложение работает в Python. Обученные модели входят в комплект `hunterilka-v2.2-local.zip`.

Из корня распакованного проекта:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
sudo docker compose up -d --wait --wait-timeout 90 postgres
python -m scripts.check_runtime
HUNTERILKA_REQUIRE_DB=1 python -m unittest discover -s tests -v
```

Внешний локальный порт БД — `55432`, внутренний порт контейнера — `5432`. Рабочая база — `hunterilka_v2`, тестовая — `hunterilka_v2_test`. Для собственного сервера задайте `HUNTERILKA_DB` и `HUNTERILKA_TEST_DB`; реальные пароли храните вне Git. Указанные в Compose реквизиты предназначены для локального запуска.

Если Docker доступен текущему пользователю, `sudo` можно опустить. Вариант автоматической настройки: `bash scripts/setup_local.sh`.

## Новый источник переводов

Для первого источника используйте пустую рабочую БД. Если `hunterilka_v2` уже содержит демонстрацию, создайте отдельную базу:

```bash
sudo docker compose exec postgres createdb -U hunterilka hunterilka_live
export HUNTERILKA_DB='postgresql://hunterilka:hunterilka@127.0.0.1:55432/hunterilka_live'
```

Создание БД выполняется один раз. Укажите время начала оценки и постоянное имя источника. Время ниже — пример для файлов из `examples/incoming/`:

```bash
mkdir -p incoming
python -m scripts.run_inbox --input-dir incoming --source local-feed --score-from 2026-10-21T12:07:00
```

Процесс ожидает новые файлы. Источник публикует законченные минуты по [контракту](input_contract.md). Минуты раньше `--score-from` загружаются как история; начиная с указанного времени создаются оценки и случаи. Если история не передана, расчёт начинается с пустого графа и использует поступившие данные по мере их накопления.

В другом терминале активируйте то же окружение, задайте тот же `HUNTERILKA_DB` и запустите интерфейс:

```bash
source .venv/bin/activate
export HUNTERILKA_DB='postgresql://hunterilka:hunterilka@127.0.0.1:55432/hunterilka_live'
python -m streamlit run app.py
```

Откройте `http://localhost:8501`. Для продолжения после остановки повторите команду `run_inbox` с тем же именем источника. `--score-from` при продолжении можно опустить. `--once` обрабатывает имеющиеся файлы и завершает процесс. Новый источник или другая модель требуют отдельной БД; существующие решения автоматически не удаляются.

## Исходники без моделей

Исходный архив содержит программу и отчёты. Для первого обучения укажите собственный путь к IBM CSV:

```bash
python -m scripts.run_pipeline --input /полный/путь/HI-Small_Trans.csv --device GPU
python -m scripts.compare_versions
```

Подготовка и признаки выполняются на CPU; `--device GPU` ускоряет CatBoost. Доступен `--device CPU`. Для уже рассчитанной установленной версии повторное обучение при обновлении программы не требуется.

## Демонстрация на исторических данных

В исходной рабочей базе с моделями и подготовленными данными:

```bash
python -m scripts.run_stream --max-groups 2000
python -m scripts.run_stream --max-groups 10
python -m streamlit run app.py
```

Второй запуск продолжает сохранённый поток. `--reset` предназначен для удаления рабочих данных выбранной БД и не используется при обычном продолжении. Демонстрация воспроизводит историю; новый источник подключается через `run_inbox`.

## Комплект для получателя

После обучения, из установленного проекта:

```bash
python -m scripts.package_release
```

Результат — `dist/hunterilka-v2.2-local.zip`. В него входят исходники, эксплуатационная документация, все обученные модели из текущего набора и метаданные моделей. CSV, таблицы признаков, содержимое БД, пароли, виртуальное окружение и учебные заметки не включаются. PostgreSQL создаёт рабочие таблицы при запуске приложения или сервиса.
