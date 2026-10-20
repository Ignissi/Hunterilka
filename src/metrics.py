"""Метрики вычисляются после прогнозов, отдельно от рабочего графа."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def budget_threshold(scores, fraction):
    values = np.sort(np.asarray(scores))[::-1]
    if len(values) == 0:
        raise ValueError("Для выбора порога нужна validation-часть.")
    if not 0 < fraction <= 1:
        raise ValueError("Доля рассмотрения должна быть больше 0 и не больше 1.")
    count = min(len(values), max(1, int(len(values) * fraction)))
    if count == len(values):
        return float(values[-1])
    # Исключаем группу равных оценок на границе, чтобы не превышать бюджет.
    return float(np.nextafter(values[count], np.inf))


def classification_metrics(labels, scores, threshold, k=100):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(labels) == 0:
        return {"status": "Не определено: пустая часть"}
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None
    used_k = min(k, len(labels))
    order = np.argsort(-scores, kind="stable")[:used_k]
    two_classes = len(np.unique(labels)) == 2
    return {
        "rows": len(labels), "positive": int(labels.sum()), "prevalence": float(labels.mean()),
        "ap": float(average_precision_score(labels, scores)) if two_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "undefined_reason": None if two_classes else "В части нет обоих классов",
        "precision": precision, "recall": recall, "f1": f1,
        "precision_at_k": float(labels[order].mean()), "k": used_k,
        "threshold": float(threshold), "alerts": int(predicted.sum()),
        "false_alerts_per_1000": float(fp * 1000 / len(labels)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def top_k_metrics(labels, scores, budgets=(100, 500)):
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-np.asarray(scores), kind="stable")
    result = {}
    for budget in budgets:
        count = min(budget, len(labels))
        found = int(labels[order[:count]].sum())
        result[str(budget)] = {"reviewed": count, "found": found,
                               "precision": found / count if count else None,
                               "recall": found / int(labels.sum()) if labels.sum() else None}
    return result
