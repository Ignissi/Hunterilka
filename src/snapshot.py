"""Окружение для экрана. Расчёт признаков и оценки моделей не изменяются."""

from datetime import timedelta

import numpy as np
import pandas as pd


WORKSPACE_VERSION = "graph_workspace_v1"
WORKSPACE_FILES = ("periods.parquet", "schemes.parquet", "formats.parquet", "network_sample.parquet", "network_nodes.parquet")


class HistorySlices:
    def __init__(self, events):
        if not events.timestamp.is_monotonic_increasing:
            events = events.sort_values(["timestamp", "source_row"], kind="stable")
        self.events = events.reset_index(drop=True)
        self.times = self.events.timestamp.to_numpy()
        self.incoming = self.events.groupby("target_key", sort=False).indices
        self.outgoing = self.events.groupby("source_key", sort=False).indices

    def around(self, evidence, detected_at, days=30, node_limit=100, event_limit=300):
        lower = (pd.Timestamp(detected_at) - timedelta(days=days)).isoformat()
        nodes = {e[side] for e in evidence for side in ["source_key", "target_key"]}
        evidence_ids = {e["transaction_id"] for e in evidence}
        selected = set()
        truncated = False
        frontier = sorted(nodes)
        for _ in range(2):
            following = set()
            for node in frontier:
                for index in [self.incoming, self.outgoing]:
                    positions = index.get(node, np.array([], dtype=int))
                    timestamps = self.times[positions]
                    start = timestamps.searchsorted(lower, side="left")
                    end = timestamps.searchsorted(detected_at, side="right")
                    truncated = truncated or end - start > 16
                    for position in positions[max(start, end - 16):end][::-1]:
                        row = self.events.iloc[position]
                        if row.transaction_id in evidence_ids:
                            continue
                        added = {row.source_key, row.target_key} - nodes
                        if len(nodes | added) > node_limit or len(selected) >= event_limit:
                            truncated = True
                            continue
                        selected.add(int(position))
                        nodes.update(added)
                        following.update(added)
            frontier = sorted(following)
        context = self.events.iloc[sorted(selected)].to_dict("records")
        return context, bool(truncated)


def case_title(case):
    events = case["events"]
    start, end = events[0], events[-1]
    return start["source_account"][-8:] + " → " + end["target_account"][-8:]


def graph_slice(events, first, last, seed, limit=300):
    table = events.loc[events.timestamp.between(first, last)].reset_index(drop=True)
    around = table.loc[(table.source_key == seed) | (table.target_key == seed)]
    if around.empty:
        return []
    anchor = around.tail(1).to_dict("records")
    context, _ = HistorySlices(table).around(anchor, last, days=30, event_limit=limit)
    return anchor + context
