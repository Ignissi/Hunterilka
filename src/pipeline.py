"""Один шаг обработки минуты. Этот порядок общий для обучения и для рабочего потока."""

import pandas as pd

from src.features import transaction_features
from src.patterns import search_patterns


def process_group(state, group, config, models=None):
    """Признаки -> оценка операций -> добавление в граф -> поиск структур -> оценка структур.

    Пока группа не добавлена, все её операции видят одно и то же прошлое.
    Если models не передан, возвращаются только признаки и кандидаты (режим обучения).
    """
    state.advance(group[0]["timestamp"])

    features = [transaction_features(event, state) for event in group]
    scores = None
    if models is not None:
        table = pd.DataFrame(features)[models["meta"]["features"]]
        scores = models["transaction"].predict_proba(table)[:, 1]

    state.add_group(group)
    candidates, diagnostics = search_patterns(state, group, config)

    if models is not None and candidates:
        table = pd.DataFrame([c["features"] for c in candidates])[models["meta"]["structure_features"]]
        anomaly = -models["structure"].score_samples(table)
        for candidate, value in zip(candidates, anomaly):
            candidate["anomaly_score"] = float(value)

    return features, scores, candidates, diagnostics
