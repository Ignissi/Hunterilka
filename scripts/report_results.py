"""Читаемый отчёт из фактически рассчитанных чисел."""

from src.storage import ROOT, read_json


def number(value, digits=4):
    return "н/д" if value is None else f"{value:.{digits}f}"


def write_report(report):
    data = report["data"]
    coverage = report["search_coverage"]
    model = report["transaction_models"][report["selected"]]
    structure = report["structure"]
    first100, first500 = model["top_k"]["100"], model["top_k"]["500"]
    _, fp = model["confusion_matrix"][0]
    fn, tp = model["confusion_matrix"][1]
    scope = "Полный исходный период" if data["scope"] == "full_source" else "Технический укороченный профиль, не итоговый эксперимент"
    text = f'''# Результаты Hunterилка

**{scope}.** Обработано {data['rows']:,} операций за {data['history_days']:.2f} дня.
Период: {data['first_time']} → {data['last_time']}. Обучение CatBoost: {report['training_device']}.
Основная модель **{report['selected']}** выбрана на validation по находкам в первых 100; при равенстве по AP.
Ниже результаты на следующем по времени test: {model['rows']:,} операций, из них {model['positive']:,} с положительной меткой IBM.

## Основной результат: какие операции проверять первыми

| Просмотрено операций по убыванию оценки | С положительной меткой | С отрицательной меткой | Доля положительных среди просмотренных |
|---|---:|---:|---:|
| Первые {first100['reviewed']} | {first100['found']} | {first100['reviewed'] - first100['found']} | {number(first100['precision'] * 100, 2)}% |
| Первые {first500['reviewed']} | {first500['found']} | {first500['reviewed'] - first500['found']} | {number(first500['precision'] * 100, 2)}% |

AP: **{number(model['ap'])}**. ROC AUC: **{number(model['roc_auc'])}**.
При случайном выборе {first500['reviewed']} операций ожидалось бы около **{first500['reviewed'] * model['prevalence']:.2f}** положительных.
Положительные составляют {model['prevalence'] * 100:.3f}% test. Поэтому ориентир случайного AP около {model['prevalence']:.5f}; для ROC AUC он равен 0.5.

Первые {first500['reviewed']} охватывают **{first500['recall'] * 100:.2f}% всех положительных операций test**.
Высокая точность начала очереди не означает, что система обнаруживает все положительные операции.
Порядок определяется оценкой модели, без использования меток. При равных оценках сохраняется порядок test.

## Почему в логе другое число находок

Рабочий порог {model['threshold']:.6f} выбран на validation и не менялся после просмотра test.
Он дал **{model['alerts']} сигналов: {tp} положительных и {fp} ложных**. Ниже порога осталось {fn} положительных операций.
Просмотр первых 500 и применение порога являются разными способами ограничить работу специалиста.
В первые 500 могут входить оценки ниже порога; такие карточки обозначены как выбранные по месту в очереди, а не как сработавшие сигналы.

## Что дали графовые признаки

A использует поля операции и историю счетов. B добавляет связи и графовые признаки.
plain обучается без весов классов, balanced с весами по train. Все четыре результата приведены без отбора удачных строк.

| Модель | AP test | ROC AUC test | Найдено в первых 100 | Найдено в первых 500 | Сигналов при своём validation-пороге |
|---|---:|---:|---:|---:|---:|
'''
    for name, result in report["transaction_models"].items():
        text += f"| {name} | {number(result['ap'])} | {number(result['roc_auc'])} | {result['top_k']['100']['found']} | {result['top_k']['500']['found']} | {result['alerts']} |\n"
    text += f'''
## Второй уровень: отдельная задача и ограничение качества

Поиск включил **{coverage['in_any_candidate']} из {coverage['test_positive_operations']}** положительных операций хотя бы в одну структуру.
Это охват кандидатов до отбора второй моделью, а не число обнаружений системы.
После структурного порога в сохранённых сигналах осталось {coverage['in_structure_alerts']} положительных операций;
из них **{coverage['additional_from_structures']}** отсутствуют среди пороговых сигналов CatBoost.
Этот дополнительный охват требует отдельного просмотра структур.

Isolation Forest на последних версиях {structure.get('deduplicated_episodes', 0):,} эпизодов:
AP {number(structure.get('ap'))}, ROC AUC {number(structure.get('roc_auc'))}.
Ориентир случайного AP: {number(structure.get('prevalence'), 5)}.
Доля положительных эпизодов среди его пороговых сигналов: {number(structure.get('precision') * 100 if structure.get('precision') is not None else None, 2)}%.
Необычность помогает исследовать связи, но качество этого отбора существенно слабее основной модели.
Оценки относятся к эпизодам, содержащим положительную операцию test. Это не разметка истинных схем отмывания.

Последняя версия эпизода и последний сохранённый сигнал могут различаться: новая группа без сигнала не удаляет прежнее основание.
Поэтому метрики последних версий и охват сохранённых сигналов показаны раздельно.

## Проверка прежней общей очереди

Прежний порядок ставил структурные сигналы перед многими операциями CatBoost. При одинаковом числе карточек он ухудшал результат.
Измерение сохранено как проверка неудачного решения, а не как характеристика новой очереди операций.

| Карточек | Только операции: положительных операций | Прежняя общая очередь: положительных операций | Новых относительно операций | Потеряно относительно операций | Операций внутри общих карточек |
|---|---:|---:|---:|---:|---:|
'''
    for budget in report.get("review_budgets", {}).values():
        text += f"| {budget['k']} | {budget['operation_positive_operations']} | {budget['combined_positive_operations']} | {budget['additional_positive_operations']} | {budget['lost_positive_operations']} | {budget['combined_operations_to_read']} |\n"
    text += '''
В интерфейсе предусмотрены две отдельные очереди: операции по оценке CatBoost и структуры по необычности.
Оценки разных объектов не складываются. Просмотр структур расходует дополнительный бюджет, а не заменяет первые операции.
Это исправление организации просмотра принято после данного test. Оно сохраняет уже измеренное ранжирование CatBoost;
нового обучения, подбора порогов или заявления о росте качества модели на независимых данных нет.
'''
    policy = report.get("review_policy")
    if policy:
        selection = "первые структурные сигналы по необычности" if policy["structure_scope"] == "top_anomaly_signals" else "ранее сохранённые примеры структур; это не полный рейтинг всех структур"
        text += f"\nСохранено {policy['operation_cases']} карточек операций и {policy['structure_cases']} карточек структур. Структурная выборка: {selection}.\n"
    else:
        text += "\nДля обновления сохранённых карточек после готового расчёта: `python -m scripts.refresh_review`.\n"
    text += '''
## Границы результата

Данные IBM синтетические. Результат измерен на одном временном разделении и не доказывает перенос на реальные банковские данные.
Метки используются обучением и оценкой; рабочий поиск и выбор карточек от них не зависят.
Поиск ограничен бюджетами и не гарантирует нахождение всех схем. Реальная задержка потокового обнаружения отдельно не измерена.

Источники чисел: `reports/metrics.json`, `reports/experiments.csv`, `reports/search_diagnostics.csv`.
'''
    (ROOT / "docs/results.md").write_text(text, encoding="utf-8")
    brief = f'''На отложенном по времени test **{report['selected']} находит {first100['found']} положительных операций в первых {first100['reviewed']} и {first500['found']} в первых {first500['reviewed']}** по оценке модели. AP **{number(model['ap'])}**, ROC AUC **{number(model['roc_auc'])}**. При случайном выборе {first500['reviewed']} операций ожидалось бы около {first500['reviewed'] * model['prevalence']:.2f} положительных.

При зафиксированном на validation пороге: **{tp} находок и {fp} ложных сигналов**. Отбор первых 500 и пороговые сигналы показаны отдельно. Второй уровень добавил {coverage['additional_from_structures']} положительных операций к пороговым сигналам первого; качество структурного отбора слабее, поэтому у него отдельная очередь.

Расчёт: **{data['rows']:,} операций, {data['history_days']:.2f} дня**. {scope}; синтетический IBM HI-Small. [Четыре модели, ограничения и проверка очередей](docs/results.md).'''
    path = ROOT / "README.md"
    readme = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if start in readme and end in readme:
        before, rest = readme.split(start, 1)
        _, after = rest.split(end, 1)
        path.write_text(before + start + "\n\n" + brief + "\n\n" + end + after, encoding="utf-8")


if __name__ == "__main__":
    write_report(read_json(ROOT / "reports/metrics.json"))
