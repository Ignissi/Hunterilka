"""SHAP относится к сохранённому входу выбранной операции."""

import numpy as np
import pandas as pd
from catboost import Pool

from src.features import CATEGORICAL


def explain_transaction(model, vector, feature_names):
    frame = pd.DataFrame([vector])[feature_names]
    for name in feature_names:
        if name not in CATEGORICAL:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    pool = Pool(frame, cat_features=CATEGORICAL)
    shap = model.get_feature_importance(pool, type="ShapValues")[0]
    raw = float(model.predict(pool, prediction_type="RawFormulaVal")[0])
    contributions = pd.DataFrame({"Признак": feature_names, "Вклад": shap[:-1]})
    contributions["Величина"] = contributions["Вклад"].abs()
    contributions = contributions.sort_values("Величина", ascending=False).drop(columns="Величина")
    reconstructed = float(shap[-1] + shap[:-1].sum())
    return {
        "table": contributions, "base": float(shap[-1]), "raw": raw,
        "reconstructed": reconstructed, "residual": abs(reconstructed - raw),
        "score": float(1 / (1 + np.exp(-np.clip(raw, -700, 700)))),
    }
