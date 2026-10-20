"""Снимок случая переживает истечение активного графа."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.context import snapshot_context
from src.patterns import make_candidate
from src.storage import fingerprint, read_json, write_json


def clean_vector(vector):
    return {name: None if pd.isna(value) else value for name, value in vector.items()}


def build_case(candidate, scores, vectors, models, run_id, context=None, context_truncated=False, review_rank=None, is_example=False):
    meta = models["meta"]
    identifiers = candidate["transaction_ids"]
    available = {identifier: float(scores[identifier]) for identifier in identifiers if identifier in scores}
    maximum = max(available.values()) if available else None
    operation_signal = maximum is not None and maximum >= meta["transaction_threshold"]
    anomaly = candidate.get("anomaly_score")
    structure_signal = anomaly is not None and anomaly >= meta["structure_threshold"]
    ranked_operation = candidate["family"] == "transaction" and review_rank is not None
    if ranked_operation and (review_rank < 1 or maximum is None):
        raise ValueError("Для места в очереди нужны положительный номер и оценка операции.")
    structure_example = is_example and candidate["family"] != "transaction"
    if not operation_signal and not structure_signal and not ranked_operation and not structure_example:
        return None

    priority = 1 if operation_signal and structure_signal else (2 if structure_signal else 3)
    return {
        "case_id": candidate["case_key"], "candidate_id": candidate["candidate_id"],
        "family": candidate["family"], "detected_at": candidate["detected_at"],
        "first_time": candidate["first_time"], "last_time": candidate["last_time"],
        "priority": priority, "operation_signal": bool(operation_signal),
        "review_rank": review_rank,
        "is_example": bool(structure_example),
        "structure_signal": bool(structure_signal), "operation_score": maximum,
        "anomaly_score": anomaly, "operation_scores": available,
        "transaction_ids": identifiers, "events": candidate["events"],
        "context": context or [], "context_truncated": context_truncated,
        "features": {identifier: clean_vector(vectors[identifier]) for identifier in identifiers if identifier in vectors},
        "structure_features": candidate.get("features", {}),
        "search_truncated": candidate["search_truncated"],
        "equal_timestamps": candidate["equal_timestamps"],
        "run_id": run_id, "model_id": meta["model_id"],
        "config_hash": meta["config_hash"], "data_hash": meta["data_hash"],
        "feature_version": meta["feature_version"],
        "status": "Новый", "comment": "", "reviewed_at": None,
    }


class CaseService:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / "index.parquet"
        self.index = {}
        if self.index_path.exists():
            rows = pd.read_parquet(self.index_path).to_dict("records")
            self.index = {row["case_id"]: row for row in rows}

    def get(self, identifier):
        if identifier not in self.index:
            raise ValueError("Случай отсутствует в индексе.")
        return read_json(self.directory / (identifier + ".json"))

    def save(self, case):
        identifier = case["case_id"]
        signature = fingerprint([case["candidate_id"], case["operation_signal"], case["structure_signal"],
                                 case["model_id"], case.get("review_rank"), case.get("context_version")])
        previous = self.index.get(identifier)
        if previous and previous["signature"] == signature:
            return

        version = 1
        if previous:
            old = self.get(identifier)
            version = int(old["version"]) + 1
            history = self.directory / "versions" / (identifier + "_" + str(old["version"]) + ".json")
            write_json(history, old)
            for field in ["status", "comment", "reviewed_at", "first_detected_at"]:
                case[field] = old[field]
        else:
            case["first_detected_at"] = case["detected_at"]

        case["version"] = version
        write_json(self.directory / (identifier + ".json"), case)
        self.index[identifier] = {
            key: case[key] for key in [
                "case_id", "family", "detected_at", "priority", "operation_score",
                "anomaly_score", "status", "version", "search_truncated",
            ]
        }
        self.index[identifier]["signature"] = signature
        self.index[identifier]["review_rank"] = case.get("review_rank")
        self.index[identifier]["operations"] = len(case["transaction_ids"])
        self.index[identifier]["accounts"] = len({e[side] for e in case["events"] for side in ["source_key", "target_key"]})
        for key in ["is_example", "operation_signal", "structure_signal"]:
            self.index[identifier][key] = bool(case.get(key, False))
        self.index[identifier]["route"] = case["events"][0]["source_account"][-8:] + " → " + case["events"][-1]["target_account"][-8:]

    def flush(self):
        if self.index:
            temporary = self.index_path.with_suffix(".tmp.parquet")
            pd.DataFrame(list(self.index.values())).to_parquet(temporary, index=False)
            temporary.replace(self.index_path)

    def review(self, identifier, status, comment):
        if status not in ["Новый", "Требует проверки", "Рассмотрен"]:
            raise ValueError("Неизвестный статус рассмотрения.")
        case = self.get(identifier)
        case.update(status=status, comment=comment, reviewed_at=datetime.now(timezone.utc).isoformat())
        write_json(self.directory / (identifier + ".json"), case)
        self.index[identifier]["status"] = status
        self.flush()


def register_signals(service, state, group, candidates, scores, vectors, models, run_id):
    proposals = list(candidates)
    threshold = models["meta"]["transaction_threshold"]
    for event in group:
        if scores.get(event["transaction_id"], -1) >= threshold:
            candidate = make_candidate("transaction", [event], [event["transaction_id"]], event["timestamp"])
            proposals.append(candidate)

    for candidate in proposals:
        case = build_case(candidate, scores, vectors, models, run_id)
        if case is None:
            continue
        context, truncated = snapshot_context(state, candidate["events"], state.config)
        case["context"] = context
        case["context_truncated"] = truncated
        service.save(case)

    service.flush()
