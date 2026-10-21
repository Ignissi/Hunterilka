"""Случай — единица работы специалиста: сигнал плюс замороженный снимок подграфа."""

import math


def clean_vector(vector):
    # В JSON нельзя записать NaN, поэтому пропуски признаков хранятся как null.
    result = {}
    for name, value in vector.items():
        if isinstance(value, float) and math.isnan(value):
            result[name] = None
        else:
            result[name] = value
    return result


def build_case(candidate, scores, vectors, meta):
    """Собирает случай из кандидата. Возвращает None, если ни один порог не пройден."""
    identifiers = candidate["transaction_ids"]
    available = {i: float(scores[i]) for i in identifiers if i in scores}
    maximum = max(available.values()) if available else None
    operation_signal = maximum is not None and maximum >= meta["transaction_threshold"]

    anomaly = candidate.get("anomaly_score")
    structure_signal = anomaly is not None and anomaly >= meta["structure_threshold"]

    if not operation_signal and not structure_signal:
        return None

    events = candidate["events"]
    accounts = {e[side] for e in events for side in ["source_key", "target_key"]}
    return {
        "case_id": candidate["case_key"],
        "candidate_id": candidate["candidate_id"],
        "family": candidate["family"],
        "detected_at": candidate["detected_at"],
        "first_time": candidate["first_time"],
        "last_time": candidate["last_time"],
        "operation_score": maximum,
        "anomaly_score": anomaly,
        "operation_signal": bool(operation_signal),
        "structure_signal": bool(structure_signal),
        "operations": len(identifiers),
        "accounts": len(accounts),
        "route": events[0]["source_account"][-8:] + " → " + events[-1]["target_account"][-8:],
        "model_id": meta["model_id"],
        "status": "Новый",
        "version": 1,
        "details": {
            "transaction_ids": identifiers,
            "anchors": candidate["anchors"],
            "events": events,
            "context": [],
            "context_truncated": False,
            "features": {str(i): clean_vector(vectors[i]) for i in identifiers if i in vectors},
            "operation_scores": {str(i): value for i, value in available.items()},
            "structure_features": candidate.get("features", {}),
            "search_truncated": candidate["search_truncated"],
            "equal_timestamps": candidate["equal_timestamps"],
        },
    }
