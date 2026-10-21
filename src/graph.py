"""Динамический граф операций: скользящее окно 30 дней и короткие окна 1 ч и 24 ч.

Узел графа — счёт (банк:номер). Ребро — факт переводов между двумя счетами.
Каждый перевод хранится отдельно как событие и удаляется, когда выходит из окна.
"""

from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal

import networkx as nx


SHORT_WINDOWS_HOURS = [1, 24]


def parse_time(value):
    return datetime.fromisoformat(str(value))


class RollingGraph:
    def __init__(self, config):
        self.config = config
        self.graph = nx.DiGraph()
        self.events = {}                       # transaction_id -> событие
        self.expiry = deque()                  # (время, transaction_id) для окна 30 дней
        self.pair_events = defaultdict(dict)   # (источник, получатель) -> {transaction_id}
        self.incoming = defaultdict(dict)      # счёт -> {transaction_id входящих}
        self.outgoing = defaultdict(dict)      # счёт -> {transaction_id исходящих}
        self.current_time = None

        # Короткие окна считают число операций и суммы по счёту и валюте.
        self.windows = {}
        for hours in SHORT_WINDOWS_HOURS:
            self.windows[hours] = {
                "queue": deque(), "count": Counter(),
                "currency_count": Counter(), "sum": defaultdict(Decimal),
            }

    def advance(self, timestamp):
        """Сдвигает время вперёд и удаляет всё, что вышло из окон."""
        moment = parse_time(timestamp)
        if self.current_time is not None and moment < self.current_time:
            raise ValueError("Получено опоздавшее событие: " + str(timestamp))
        self.current_time = moment

        for hours, window in self.windows.items():
            boundary = moment - timedelta(hours=hours)
            while window["queue"] and window["queue"][0][0] <= boundary:
                _, event = window["queue"].popleft()
                self.change_statistics(window, event, -1)

        boundary = moment - timedelta(days=self.config["window_days"])
        while self.expiry and self.expiry[0][0] <= boundary:
            _, identifier = self.expiry.popleft()
            self.remove(identifier)

    def change_statistics(self, window, event, sign):
        sides = [
            ("out", event["source_key"], event["payment_currency"], event["amount_paid"]),
            ("in", event["target_key"], event["receiving_currency"], event["amount_received"]),
        ]
        for direction, account, currency, amount in sides:
            key = (direction, account)
            currency_key = (direction, account, currency)
            window["count"][key] += sign
            window["currency_count"][currency_key] += sign
            window["sum"][currency_key] += sign * Decimal(str(amount))

            if window["count"][key] == 0:
                del window["count"][key]
            if window["currency_count"][currency_key] == 0:
                del window["currency_count"][currency_key]
                del window["sum"][currency_key]

    def add_group(self, events):
        """Добавляет все операции одной минуты. Группа добавляется только целиком."""
        identifiers = [event["transaction_id"] for event in events]
        if len(set(identifiers)) != len(identifiers) or any(i in self.events for i in identifiers):
            raise ValueError("Группа содержит повторный transaction_id.")
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

            for window in self.windows.values():
                window["queue"].append((self.current_time, event))
                self.change_statistics(window, event, 1)

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
        """(операций по счёту, операций по счёту в валюте, сумма в валюте) за окно."""
        window = self.windows[hours]
        key = (direction, account, currency)
        return (
            window["count"].get((direction, account), 0),
            window["currency_count"].get(key, 0),
            float(window["sum"].get(key, 0)),
        )
