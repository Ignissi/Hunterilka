"""Воспроизведение целых групп времени и восстановление позиции."""

import uuid

import pandas as pd

from src.cases import CaseService, register_signals
from src.graph import RollingGraph
from src.pipeline import process_group
from src.storage import ROOT, file_hash, fingerprint, read_json, write_json


class ReplayController:
    def __init__(self, events, manifest, config, models, directory=None):
        self.events = events
        self.manifest = manifest
        self.config = config
        self.models = models
        self.position = 0
        self.status = "Готов"
        self.error = ""
        self.last_diagnostics = {}
        self.state = RollingGraph(config)
        self.scores = {}
        self.vectors = {}
        self.run_id = uuid.uuid4().hex[:12] if directory is None else directory.name
        self.directory = ROOT / "runs" / self.run_id if directory is None else directory
        self.cases = CaseService(self.directory / "cases")
        self.score_from = manifest["validation_end"]

    def step(self, save=True):
        if self.position >= len(self.events):
            self.status = "Завершён"
            return 0
        timestamp = self.events.iloc[self.position]["timestamp"]
        end = int(self.events.timestamp.searchsorted(timestamp, side="right"))
        group = self.events.iloc[self.position:end].to_dict("records")
        score_enabled = timestamp >= self.score_from

        try:
            features, scores, candidates, info = process_group(
                self.state, group, self.config, self.models, score_enabled,
            )
            for identifier in self.state.expired_ids:
                self.scores.pop(identifier, None)
                self.vectors.pop(identifier, None)

            if score_enabled:
                for event, vector, score in zip(group, features, scores):
                    identifier = event["transaction_id"]
                    self.scores[identifier] = float(score)
                    self.vectors[identifier] = vector
                if save:
                    register_signals(self.cases, self.state, group, candidates, self.scores,
                                     self.vectors, self.models, self.run_id)

            self.position = end
            self.last_diagnostics = info
            self.status = "Завершён" if end == len(self.events) else "Пауза"
            if save:
                self.checkpoint()
            return len(group)
        except (ValueError, OSError) as error:
            self.status = "Ошибка"
            self.error = str(error)
            raise

    def batch(self):
        processed = 0
        while processed < self.config["batch_events"] and self.position < len(self.events):
            processed += self.step()
        return processed

    def checkpoint(self):
        write_json(self.directory / "checkpoint.json", {
            "run_id": self.run_id, "position": self.position,
            "config_hash": fingerprint(self.config), "model_id": self.models["meta"]["model_id"],
            "data_hash": self.manifest["events_sha256"],
        })

    def restore(self):
        checkpoint = read_json(self.directory / "checkpoint.json")
        expected = [fingerprint(self.config), self.models["meta"]["model_id"], self.manifest["events_sha256"]]
        actual = [checkpoint["config_hash"], checkpoint["model_id"], checkpoint["data_hash"]]
        if actual != expected:
            raise ValueError("Контрольная точка относится к другим данным, настройкам или моделям.")
        target = checkpoint["position"]
        if target < 0 or target > len(self.events):
            raise ValueError("Некорректная позиция контрольной точки.")
        if 0 < target < len(self.events) and self.events.iloc[target - 1].timestamp == self.events.iloc[target].timestamp:
            raise ValueError("Контрольная точка разрывает временную группу.")
        while self.position < target:
            self.step(save=False)
        self.status = "Пауза" if self.position < len(self.events) else "Завершён"


def open_replay(config, models, directory=None):
    processed = ROOT / "data/processed"
    manifest = read_json(processed / "dataset_manifest.json")
    if manifest["events_sha256"] != models["meta"]["data_hash"]:
        raise ValueError("Набор не совпадает с обучением. Выполните полный цикл заново.")
    if file_hash(processed / "events.parquet") != manifest["events_sha256"]:
        raise ValueError("Подготовленные события изменились после импорта.")
    events = pd.read_parquet(processed / "events.parquet")
    return ReplayController(events, manifest, config, models, directory)
