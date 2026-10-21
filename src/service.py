"""Сервис анализа: полная минута и её результат фиксируются одной транзакцией БД."""

from collections import deque
from datetime import timedelta
from itertools import groupby

from src import db
from src.cases import build_case
from src.context import snapshot_context
from src.graph import RollingGraph, parse_time
from src.patterns import make_candidate
from src.pipeline import process_group


def minute_groups(events):
    for timestamp, group in groupby(events, key=lambda event: event["timestamp"]):
        yield timestamp, list(group)


class Service:
    def __init__(self, config, models, conn):
        self.config = config
        self.models = models
        self.conn = conn
        self.saved_cases = 0
        self.reset_memory()

    def reset_memory(self):
        self.state = RollingGraph(self.config)
        self.scores = {}
        self.vectors = {}
        self.recent = deque()
        self.current_day = None
        self.needs_restore = False

    def warm_up(self, events):
        for timestamp, group in minute_groups(events):
            self.state.advance(timestamp)
            self.state.add_group(group)

    def restore(self):
        """Граф восстанавливается из окна, оценки и векторы — из сохранённых результатов."""
        self.reset_memory()
        progress = db.stream_state(self.conn)
        if progress is None or progress["last_time"] is None:
            return
        last = parse_time(progress["last_time"])
        start = last - timedelta(days=self.config["window_days"])
        end = last + timedelta(seconds=1)
        self.warm_up(db.iterate_transactions(self.conn, start.isoformat(), end.isoformat()))
        self.state.advance(progress["last_time"])
        recent_start = last - timedelta(hours=self.config["motif_window_hours"])
        for row in db.recent_scores(self.conn, recent_start.isoformat(), last.isoformat(), self.models["meta"]["model_id"]):
            identifier = row["transaction_id"]
            self.scores[identifier] = float(row["operation_score"])
            self.vectors[identifier] = row["feature_vector"]
            self.recent.append((db.time_text(row["ts"]), identifier))

    def ingest_history(self, events):
        """Прогрев тоже имеет курсор: можно продолжить после любой полной пачки минут."""
        timestamp = events[-1]["timestamp"]
        with self.conn.transaction():
            db.insert_transactions(self.conn, events)
            boundary = parse_time(timestamp) - timedelta(days=self.config["window_days"])
            db.delete_old_transactions(self.conn, boundary.isoformat())
            db.set_progress(self.conn, timestamp, history_complete=False)
        self.warm_up(events)

    def process(self, group):
        if not group:
            raise ValueError("Пустая группа операций.")
        timestamp = group[0]["timestamp"]
        if any(event["timestamp"] != timestamp for event in group):
            raise ValueError("В process передаётся одна полная группа времени.")
        if self.needs_restore:
            self.restore()
        progress = db.stream_state(self.conn)
        if not progress or not progress["history_complete"] or progress["kind"] not in {"replay", "inbox"}:
            raise ValueError("Сначала зарегистрируйте источник и завершите прогрев истории.")
        if progress["last_time"] and timestamp <= progress["last_time"]:
            if timestamp == progress["last_time"] and db.group_already_saved(self.conn, group):
                return {"saved_cases": 0, "skipped_group": True}
            raise ValueError("Опоздавшая или изменённая группа: курсор уже прошёл её время.")

        try:
            # Все записи ниже либо становятся видны вместе, либо откатываются вместе.
            with self.conn.transaction():
                features, scores, candidates, diagnostics = process_group(self.state, group, self.config, self.models)
                db.insert_transactions(self.conn, group, scores, features, self.models["meta"]["model_id"])
                self.clean_window(timestamp)
                proposals = list(candidates)
                threshold = self.models["meta"]["transaction_threshold"]
                for event, vector, score in zip(group, features, scores):
                    identifier = event["transaction_id"]
                    self.scores[identifier] = float(score)
                    self.vectors[identifier] = vector
                    self.recent.append((timestamp, identifier))
                    if score >= threshold:
                        proposals.append(make_candidate("transaction", [event], [identifier], timestamp))

                saved = 0
                for candidate in proposals:
                    case = build_case(candidate, self.scores, self.vectors, self.models["meta"])
                    if case is None:
                        continue
                    context, truncated = snapshot_context(self.state, candidate["events"], self.config)
                    case["details"]["context"] = context
                    case["details"]["context_truncated"] = truncated
                    if db.save_case(self.conn, case) != "unchanged":
                        saved += 1
                db.set_progress(self.conn, timestamp)
            self.forget_old(timestamp)
        except BaseException:
            # Память могла измениться; повтор разрешён лишь после восстановления БД.
            self.reset_memory()
            self.needs_restore = True
            raise

        self.saved_cases += saved
        diagnostics["saved_cases"] = saved
        return diagnostics

    def clean_window(self, timestamp):
        day = timestamp[:10]
        if day != self.current_day:
            boundary = parse_time(timestamp) - timedelta(days=self.config["window_days"])
            db.delete_old_transactions(self.conn, boundary.isoformat())
            scores_before = parse_time(timestamp) - timedelta(hours=self.config["motif_window_hours"])
            db.forget_old_scores(self.conn, scores_before.isoformat())
            self.current_day = day

    def forget_old(self, timestamp):
        cutoff = parse_time(timestamp) - timedelta(hours=self.config["motif_window_hours"])
        while self.recent and parse_time(self.recent[0][0]) < cutoff:
            _, identifier = self.recent.popleft()
            self.scores.pop(identifier, None)
            self.vectors.pop(identifier, None)
