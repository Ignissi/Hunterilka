"""Сравнение с зафиксированной v1: только один источник и то же временное разделение."""
from src.report import number
from src.storage import ROOT, read_json


def main():
    old = read_json(ROOT / "reports" / "v1_reference.json")
    new = read_json(ROOT / "reports" / "metrics.json")
    data = new["data"]
    same = (data.get("source_sha256") == old["source_sha256"] and data["rows"] == old["rows"]
            and data["train_end"] == old["train_end"] and data["validation_end"] == old["validation_end"])
    if not same:
        raise ValueError("Для сравнения с v1 нужен полный тот же HI-Small CSV и тот же временной сплит. Техническая выборка не сравнивается с итоговыми метриками.")
    text = "# Сравнение v1 и v2\n\nОдин исходный CSV и временное разделение. V1 — сохранённый результат коммита `" + old["commit"][:7] + "`; v2 — новый расчёт.\n\n"
    text += "| Версия | Модель | AP всего test | ROC AUC | Найдено @100 | Найдено @500 |\n|---|---|---:|---:|---:|---:|\n"
    for title, report in [("v1", old), ("v2", new)]:
        for name, result in report["transaction_models"].items():
            text += f"| {title} | {name} | {number(result['ap'])} | {number(result['roc_auc'])} | {result['top_k']['100']['found']} | {result['top_k']['500']['found']} |\n"
    text += "\nВыбранные на validation модели: v1 — **" + old["selected"] + "**, v2 — **" + new["selected"] + "**.\n\n"
    text += "В v2 изменены признаки, поэтому это сравнение версий, а не чистый эксперимент о влиянии PostgreSQL. Влияние графовых признаков оценивается сравнением A/B внутри одного запуска. Метрики периодов v2 приведены в results.md; для v1 дополнительные срезы в этом сравнении не рассчитывались.\n"
    (ROOT / "docs" / "version_comparison.md").write_text(text, encoding="utf-8")
    print("Сравнение готово: docs/version_comparison.md")


if __name__ == "__main__":
    main()
