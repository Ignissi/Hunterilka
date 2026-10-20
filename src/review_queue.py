"""Две очереди: оценки разных объектов не конкурируют за одно место."""

import numpy as np
import pandas as pd


REVIEW_POLICY = "separate_v1"


def operation_order(scores, limit=500):
    # При равных оценках сохраняется порядок test, как в расчёте метрик.
    return np.argsort(-np.asarray(scores), kind="stable")[:limit]


def ordered_cases(rows, channel):
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    operations = table.family == "transaction"
    table = table.loc[operations if channel == "operations" else ~operations].copy()
    if "review_rank" not in table:
        table["review_rank"] = np.nan
    score = "operation_score" if channel == "operations" else "anomaly_score"
    return table.sort_values([score, "review_rank", "case_id"], ascending=[False, True, True],
                             na_position="last", kind="stable")
