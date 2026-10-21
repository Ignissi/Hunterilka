"""Первый ML-уровень: одна строка признаков на одну операцию.

Все признаки считаются по состоянию графа ДО добавления текущей минуты,
поэтому будущее в них попасть не может.
"""

import math

from src.context import short_return_path
from src.graph import parse_time


CATEGORICAL = ["payment_currency", "receiving_currency", "payment_format"]

# Признаки, которые требуют графа. Модель A обучается без них, модель B — с ними.
GRAPH_FEATURES = [
    "pair_count_window", "reverse_pair_exists", "pair_seen_before",
    "source_in_degree", "source_out_degree", "target_in_degree", "target_out_degree",
    "short_return_path_found", "short_return_search_complete",
]


def transaction_features(event, state):
    source = event["source_key"]
    target = event["target_key"]
    paid = float(event["amount_paid"])
    received = float(event["amount_received"])
    moment = parse_time(event["timestamp"])

    out_hour, _, _ = state.statistics(1, "out", source, event["payment_currency"])
    in_hour, _, _ = state.statistics(1, "in", target, event["receiving_currency"])
    out_day, currency_count, paid_sum = state.statistics(24, "out", source, event["payment_currency"])
    in_day, _, received_sum = state.statistics(24, "in", target, event["receiving_currency"])

    # Среднее прошлых списаний считается только в валюте текущей операции.
    past_mean = paid_sum / currency_count if currency_count else 0
    ratio_missing = past_mean <= 0
    ratio = paid / past_mean if not ratio_missing else float("nan")

    found, complete = short_return_path(state.graph, source, target, state.config["fast_scan_budget"])
    pair_count = len(state.pair_events.get((source, target), {}))

    result = {
        "log_amount_paid": math.log1p(paid),
        "log_amount_received": math.log1p(received),
        "payment_currency": event["payment_currency"],
        "receiving_currency": event["receiving_currency"],
        "payment_format": event["payment_format"],
        "same_bank": int(event["source_bank"] == event["target_bank"]),
        "same_currency": int(event["payment_currency"] == event["receiving_currency"]),
        "self_transfer": int(source == target),
        "hour": moment.hour,
        "weekday": moment.weekday(),
        "source_out_count_1h": out_hour,
        "target_in_count_1h": in_hour,
        "source_out_count_24h": out_day,
        "target_in_count_24h": in_day,
        "source_out_sum_24h_currency": math.log1p(paid_sum),
        "target_in_sum_24h_currency": math.log1p(received_sum),
        "paid_to_past_mean_ratio": ratio,
        "paid_ratio_missing": int(ratio_missing),
        "source_currency_history_count": currency_count,
        "source_is_new": int(source not in state.graph),
        "target_is_new": int(target not in state.graph),
        "pair_count_window": pair_count,
        "reverse_pair_exists": int(state.graph.has_edge(target, source)),
        "pair_seen_before": int(pair_count > 0),
        "short_return_path_found": int(found),
        "short_return_search_complete": int(complete),
    }

    for role, node in [("source", source), ("target", target)]:
        result[role + "_in_degree"] = state.graph.in_degree(node) if node in state.graph else 0
        result[role + "_out_degree"] = state.graph.out_degree(node) if node in state.graph else 0
    return result
