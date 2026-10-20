"""Динамический граф и согласованное истечение трёх временных окон."""

from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal

import networkx as nx


def parse_time(value):
    return datetime.fromisoformat(str(value))


class RollingGraph:
    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()
        self.events = {}
        self.expiry = deque()
        self.pair_events = defaultdict(dict)
        self.incoming = defaultdict(dict)
        self.outgoing = defaultdict(dict)
        self.first_time = None
        self.current_time = None
        self.expired_ids = []

        # Каждое короткое окно имеет собственную очередь истечения.
        self.windows = {}
        for hours in [1, 24]:
            self.windows[hours] = {
                "queue": deque(), "count": Counter(),
                "currency_count": Counter(), "sum": defaultdict(Decimal),
            }

    def advance(self, timestamp):
        self.expired_ids = []
        moment = parse_time(timestamp)
        if self.current_time is not None and moment < self.current_time:
            raise ValueError("Получено опоздавшее событие. Пересоберите профиль по времени.")

        self.current_time = moment
        if self.first_time is None:
            self.first_time = moment

        for hours, state in self.windows.items():
            boundary = moment - timedelta(hours=hours)
            while state["queue"] and state["queue"][0][0] <= boundary:
                _, event = state["queue"].popleft()
                self.change_statistics(state, event, -1)

        boundary = moment - timedelta(days=self.config["window_days"])
        while self.expiry and self.expiry[0][0] <= boundary:
            _, identifier = self.expiry.popleft()
            self.remove(identifier)
            self.expired_ids.append(identifier)

    def change_statistics(self, state, event, sign):
        sides = [
            ("out", event["source_key"], event["payment_currency"], event["amount_paid"]),
            ("in", event["target_key"], event["receiving_currency"], event["amount_received"]),
        ]
        for direction, account, currency, amount in sides:
            key = (direction, account)
            currency_key = (direction, account, currency)
            state["count"][key] += sign
            state["currency_count"][currency_key] += sign
            state["sum"][currency_key] += sign * Decimal(amount)

            if state["count"][key] == 0:
                del state["count"][key]
            if state["currency_count"][currency_key] == 0:
                del state["currency_count"][currency_key]
                del state["sum"][currency_key]

    def add_group(self, events):
        identifiers = [event["transaction_id"] for event in events]
        if len(set(identifiers)) != len(identifiers) or any(i in self.events for i in identifiers):
            raise ValueError("Группа содержит повторный transaction_id.")
        if len(self.events) + len(events) > self.config["max_active_events"]:
            raise ValueError("Достигнут лимит активных операций. Измените профиль или явный лимит.")
        if any(parse_time(event["timestamp"]) != self.current_time for event in events):
            raise ValueError("Добавляется только полная группа текущего времени.")

        for event in sorted(events, key=lambda row: row["transaction_id"]):
            identifier = event["transaction_id"]
            source, target = event["source_key"], event["target_key"]
            self.events[identifier] = event
            self.expiry.append((self.current_time, identifier))
            self.pair_events[(source, target)][identifier] = None
            self.outgoing[source][identifier] = None
            self.incoming[target][identifier] = None
            self.graph.add_edge(source, target, count=len(self.pair_events[(source, target)]))

            for state in self.windows.values():
                state["queue"].append((self.current_time, event))
                self.change_statistics(state, event, 1)

    def remove(self, identifier):
        event = self.events.pop(identifier)
        source, target = event["source_key"], event["target_key"]
        pair = (source, target)
        del self.pair_events[pair][identifier]
        del self.outgoing[source][identifier]
        del self.incoming[target][identifier]

        if self.pair_events[pair]:
            self.graph[source][target]["count"] -= 1
        else:
            del self.pair_events[pair]
            self.graph.remove_edge(source, target)

        for index, key in [(self.outgoing, source), (self.incoming, target)]:
            if not index[key]:
                del index[key]
        for node in {source, target}:
            if node in self.graph and self.graph.degree(node) == 0:
                self.graph.remove_node(node)

    def statistics(self, hours, direction, account, currency):
        state = self.windows[hours]
        key = (direction, account, currency)
        return (
            state["count"].get((direction, account), 0),
            state["currency_count"].get(key, 0),
            float(state["sum"].get(key, 0)),
        )
