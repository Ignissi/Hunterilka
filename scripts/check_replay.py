"""Проверка рабочего прохода на исходном подготовленном IBM-профиле."""

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.explain import explain_transaction
from src.models import load_models
from src.replay import open_replay
from src.storage import ROOT, load_config, write_json


def main():
    config = load_config()
    models = load_models(config)
    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary) / "check"
        controller = open_replay(config, models, directory)
        controller.step()
        controller.step()
        position = controller.position
        edges = sorted(controller.state.graph.edges(data="count"))
        restored = open_replay(config, models, directory)
        restored.restore()
        if restored.position != position or sorted(restored.state.graph.edges(data="count")) != edges:
            raise ValueError("Восстановление изменило позицию или граф.")

        while restored.position < len(restored.events):
            last_time = restored.events.timestamp.iloc[-1]
            save = restored.events.timestamp.iloc[restored.position] == last_time
            restored.step(save=save)

        frame = pd.read_parquet(ROOT / "artifacts/features.parquet")
        test = frame.loc[frame.split == "test"]
        expected = models["transaction"].predict_proba(test[models["meta"]["features"]])[:, 1]
        observed = np.array([restored.scores[identifier] for identifier in test.transaction_id])
        difference = float(np.max(np.abs(expected - observed)))
        if difference > 1e-12:
            raise ValueError("Оценки рабочего прохода отличаются от таблицы эксперимента.")
        identifier = test.transaction_id.iloc[0]
        shap = explain_transaction(models["transaction"], restored.vectors[identifier], models["meta"]["features"])
        if shap["residual"] > 1e-8:
            raise ValueError("Вклады SHAP не восстанавливают raw score.")

        result = {
            "restoration": "passed", "replay_rows": len(restored.events),
            "model_id": models["meta"]["model_id"], "scope": restored.manifest["scope"],
            "compared_test_scores": len(test), "max_score_difference": difference,
            "shap_residual": shap["residual"], "persisted_cases": len(restored.cases.index), "seconds": time.perf_counter() - started,
        }
        try:
            import resource
            result["peak_rss_mib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            result["peak_rss_mib_linux"] = None
        write_json(ROOT / "reports/replay_check.json", result)
        print(result)


if __name__ == "__main__":
    main()
