"""Единственный порядок вычислений для подготовки и рабочего воспроизведения."""

import time

import pandas as pd

from src.features import transaction_features
from src.patterns import search_patterns


def process_group(state, group, config, models=None, score_enabled=True):
    started = time.perf_counter()
    state.advance(group[0]["timestamp"])

    # До фиксации группы все строки видят одно и то же прошлое состояние.
    features = [transaction_features(event, state) for event in group]
    scores = None
    if models is not None and score_enabled:
        scores = models["transaction"].predict_proba(
            pd.DataFrame(features)[models["meta"]["features"]]
        )[:, 1]

    state.add_group(group)
    search_started = time.perf_counter()
    candidates, diagnostics = search_patterns(state, group, config)
    diagnostics["search_seconds"] = time.perf_counter() - search_started

    if models is not None and score_enabled and candidates:
        table = pd.DataFrame([c["features"] for c in candidates])
        table = table[models["meta"]["structure_features"]]
        anomaly = -models["structure"].score_samples(table)
        for candidate, score in zip(candidates, anomaly):
            candidate["anomaly_score"] = float(score)

    diagnostics["group_seconds"] = time.perf_counter() - started
    diagnostics["group_size"] = len(group)
    return features, scores, candidates, diagnostics
