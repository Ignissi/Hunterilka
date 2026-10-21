"""Метрики качества. Считаются после прогнозов, отдельно от рабочего графа."""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def budget_threshold(scores, fraction):
    """Порог, при котором сигналом становится примерно заданная доля операций."""
    values = np.sort(np.asarray(scores))[::-1]
    if len(values) == 0:
        raise ValueError("Для выбора порога нужна validation-часть.")
    if not 0 < fraction <= 1:
        raise ValueError("Доля рассмотрения должна быть больше 0 и не больше 1.")
    count = min(len(values), max(1, int(len(values) * fraction)))
    if count == len(values):
        return float(values[-1])
    # Группа одинаковых оценок на границе не включается, чтобы не превысить бюджет.
    return float(np.nextafter(values[count], np.inf))


def classification_metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    two_classes = len(np.unique(labels)) == 2

    return {
        "rows": int(len(labels)), "positive": int(labels.sum()), "prevalence": float(labels.mean()),
        "ap": float(average_precision_score(labels, scores)) if two_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if two_classes else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        "threshold": float(threshold), "alerts": int(predicted.sum()),
        "false_alerts_per_1000": float(fp * 1000 / len(labels)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "top_k": top_k_metrics(labels, scores),
    }


def top_k_metrics(labels, scores, budgets=(100, 500)):
    """Сколько положительных операций попало в первые K по оценке модели."""
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-np.asarray(scores), kind="stable")
    result = {}
    for budget in budgets:
        count = min(budget, len(labels))
        found = int(labels[order[:count]].sum())
        result[str(budget)] = {
            "reviewed": count, "found": found,
            "precision": found / count if count else None,
            "recall": found / int(labels.sum()) if labels.sum() else None,
        }
    return result
