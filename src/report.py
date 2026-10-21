"""Читаемый отчёт из фактически рассчитанных чисел: docs/results.md и блок в README."""

from src.storage import ROOT


PERIOD_TITLES = {"normal": "Обычный период", "tail": "Хвост с 11 сентября"}


def number(value, digits=4):
    return "н/д" if value is None else f"{value:.{digits}f}"


def period_label(part):
    return part["first_time"][:10] + " – " + part["last_time"][:10]


def period_section(report):
    periods = report["transaction_models"][report["selected"]].get("periods")
    if not periods:
        return ""
    text = '''## Дополнительная проверка по периодам test

Дополнительная диагностика одного эксперимента: состав IBM HI-Small меняется по времени.
Основной результат относится ко всему test; оба среза сохранены ниже.
Диагностическая дата не используется для приёма новых операций.

| Период | Даты | Операций | Положительных | AP | ROC AUC | Найдено в первых 100 | Найдено в первых 500 |
|---|---|---:|---:|---:|---:|---:|---:|
'''
    for name in ["normal", "tail"]:
        part = periods[name]
        if part["rows"] == 0:
            continue
        text += (f"| {PERIOD_TITLES[name]} | {period_label(part)} | {part['rows']:,} | {part['positive']:,} "
                 f"| {number(part['ap'])} | {number(part['roc_auc'])} "
                 f"| {part['top_k']['100']['found']} | {part['top_k']['500']['found']} |\n")

    text += '''
Все модели на обычном периоде:

| Модель | AP | ROC AUC | Найдено в первых 100 | Найдено в первых 500 |
|---|---:|---:|---:|---:|
'''
    for name, result in report["transaction_models"].items():
        part = result["periods"]["normal"]
        if part["rows"] == 0:
            continue
        text += (f"| {name} | {number(part['ap'])} | {number(part['roc_auc'])} "
                 f"| {part['top_k']['100']['found']} | {part['top_k']['500']['found']} |\n")
    return text + "\n"


def readme_results(report):
    """Компактный блок README; значения всегда берутся из отчёта текущего расчёта."""
    data = report["data"]
    selected = report["selected"]
    model = report["transaction_models"][selected]
    top = model["top_k"]["500"]
    _, fp = model["confusion_matrix"][0]
    _, tp = model["confusion_matrix"][1]
    additional = report["structure"]["coverage"]["additional_from_structures"]
    integer = lambda value: f"{value:,}".replace(",", " ")
    decimal = lambda value, digits=4: number(value, digits).replace(".", ",")
    scope = "Полный исходный период" if data["scope"] == "full_source" else "Укороченный профиль"
    third_title = "Качество ранжирования"
    third_value = "AP " + decimal(model["ap"])
    third_note = "ROC AUC " + decimal(model["roc_auc"])
    baseline = report["transaction_models"].get("A_plain")
    if selected == "B_plain" and baseline and baseline["top_k"]["500"]["reviewed"] == top["reviewed"]:
        base_found = baseline["top_k"]["500"]["found"]
        difference = top["found"] - base_found
        third_title = "Вклад графовых признаков"
        third_value = f"{difference:+d} при том же объёме проверки"
        third_note = f"B_plain: {top['found']} · A_plain: {base_found}"
    return f"""Локальный эксперимент на **синтетическом IBM HI-Small**. Модель **{selected}** проверена на **всём отложенном test: {integer(model['rows'])} операций**.

| Масштаб расчёта | Первые {top['reviewed']} по оценке модели | {third_title} |
|:---:|:---:|:---:|
| **{integer(data['rows'])} операций** | **{top['found']} с положительной меткой** | **{third_value}** |
| {scope} · {decimal(data['history_days'], 2)} дня | Precision@{top['reviewed']} = **{decimal(top['precision'] * 100, 2)}%** | {third_note} |

В test {integer(model['positive'])} положительных операций; первые {top['reviewed']} охватывают **{decimal(top['recall'] * 100, 2)}%** из них. **AP {decimal(model['ap'])} · ROC AUC {decimal(model['roc_auc'])}.** Это качество ранжирования на данном наборе, а не доля всех нарушений, выявляемых в банковском потоке.

При отдельном пороге, выбранном на validation: **положительных операций найдено {tp}, ложных сигналов — {fp}**. Второй уровень добавил **{additional} положительных операций** сверх пороговых сигналов первого. Ранжирование первых {top['reviewed']} и отбор по порогу — разные режимы оценки.

Обучение CatBoost выполнено локально на **{report['training_device']}**. Качество различается между временными периодами; [полный отчёт](docs/results.md) содержит сравнение моделей, оценку структур и дополнительные срезы."""


def write_report(report):
    data = report["data"]
    model = report["transaction_models"][report["selected"]]
    structure = report["structure"]
    coverage = structure["coverage"]
    first100, first500 = model["top_k"]["100"], model["top_k"]["500"]
    _, fp = model["confusion_matrix"][0]
    fn, tp = model["confusion_matrix"][1]
    scope = "Полный исходный период" if data["scope"] == "full_source" else "Укороченный профиль, не итоговый эксперимент"

    text = f'''# Результаты локального ML-эксперимента

**{scope}.** Обработано {data['rows']:,} операций за {data['history_days']:.2f} дня.
Период: {data['first_time']} → {data['last_time']}. Обучение CatBoost: {report['training_device']}.
Основная модель **{report['selected']}** выбрана на validation по находкам в первых 100; при равенстве по AP.
Ниже результаты на следующем по времени test: {model['rows']:,} операций, из них {model['positive']:,} с положительной меткой IBM.

## Основной результат: какие операции проверять первыми

| Просмотрено операций по убыванию оценки | С положительной меткой | С отрицательной меткой | Доля положительных |
|---|---:|---:|---:|
| Первые {first100['reviewed']} | {first100['found']} | {first100['reviewed'] - first100['found']} | {number(first100['precision'] * 100, 2)}% |
| Первые {first500['reviewed']} | {first500['found']} | {first500['reviewed'] - first500['found']} | {number(first500['precision'] * 100, 2)}% |

AP: **{number(model['ap'])}**. ROC AUC: **{number(model['roc_auc'])}**.
Положительные составляют {model['prevalence'] * 100:.3f}% test, поэтому случайный выбор {first500['reviewed']} операций дал бы около {first500['reviewed'] * model['prevalence']:.2f} положительных.
Первые {first500['reviewed']} охватывают {first500['recall'] * 100:.2f}% всех положительных операций test.

Рабочий порог {model['threshold']:.6f} выбран на validation и не менялся после просмотра test.
Он дал **{model['alerts']} сигналов: {tp} положительных и {fp} ложных**. Ниже порога осталось {fn} положительных операций.

{period_section(report)}## Что дали графовые признаки

A использует поля операции и историю счетов. B добавляет связи и графовые признаки.
plain обучается без весов классов, balanced с весами. Все четыре результата приведены на всём test.

| Модель | AP test | ROC AUC test | Найдено в первых 100 | Найдено в первых 500 | Сигналов при своём пороге |
|---|---:|---:|---:|---:|---:|
'''
    for name, result in report["transaction_models"].items():
        text += f"| {name} | {number(result['ap'])} | {number(result['roc_auc'])} | {result['top_k']['100']['found']} | {result['top_k']['500']['found']} | {result['alerts']} |\n"

    text += f'''
## Второй уровень: структуры

Поиск включил **{coverage['in_any_candidate']} из {coverage['test_positive_operations']}** положительных операций хотя бы в одну структуру.
Это охват кандидатов до отбора второй моделью, а не число обнаружений системы.
Среди структур, прошедших порог Isolation Forest, {coverage['in_structure_alerts']} положительных операций;
из них **{coverage['additional_from_structures']}** отсутствуют среди пороговых сигналов CatBoost.

Isolation Forest на {structure['episodes']:,} эпизодах test: AP {number(structure['ap'])}, ROC AUC {number(structure['roc_auc'])}.
Ориентир случайного AP: {number(structure['prevalence'], 5)}.
Эпизод считается положительным, если содержит хотя бы одну положительную операцию test; это приближение, а не разметка схем.
Структурный отбор оценивается отдельно: его AP относится к эпизодам, а AP CatBoost — к операциям. У этих объектов отдельные очереди.

## Границы результата

Данные IBM синтетические. Результат измерен на одном временном разделении и не доказывает перенос на реальные банковские данные.
Метки используются только обучением и оценкой; рабочий поиск и очередь от них не зависят.
Поиск ограничен бюджетами и не гарантирует нахождение всех схем.

Источники чисел: `reports/metrics.json`, `reports/experiments.csv`, `reports/search_diagnostics.csv`.
'''
    (ROOT / "docs" / "results.md").write_text(text, encoding="utf-8")

    brief = readme_results(report)

    path = ROOT / "README.md"
    readme = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if start in readme and end in readme:
        before, rest = readme.split(start, 1)
        _, after = rest.split(end, 1)
        path.write_text(before + start + "\n\n" + brief + "\n\n" + end + after, encoding="utf-8")
