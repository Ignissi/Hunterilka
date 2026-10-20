"""Кандидаты имеют форму, но ещё не имеют решения об аномальности."""

from collections import Counter, defaultdict
import heapq
from datetime import timedelta
from itertools import islice

import numpy as np

from src.graph import parse_time
from src.storage import fingerprint


FAMILY_NAMES = {
    "cycle": "Кольцо", "reciprocal": "Взаимная пара", "chain": "Транзитная цепочка",
    "fan_in": "Сбор", "fan_out": "Распределение",
    "gather_scatter": "Сбор и распределение", "scatter_gather": "Разветвление и схождение",
    "transaction": "Сигнал операции",
}


def recent_events(state, index, node, budget, cutoff, maximum):
    identifiers = index.get(node, {})
    result = []

    for identifier in reversed(identifiers):
        if budget["scanned"] >= budget["limit"] or len(result) >= maximum:
            budget["truncated"] = True
            break
        budget["scanned"] += 1
        event = state.events[identifier]
        if parse_time(event["timestamp"]) < cutoff:
            break
        if event["source_key"] != event["target_key"]:
            result.append(event)
    return result


def make_candidate(family, events, anchors, detected_at):
    unique = {event["transaction_id"]: event for event in events}
    ordered = sorted(unique.values(), key=lambda e: (e["timestamp"], e["source_row"]))
    identifiers = [event["transaction_id"] for event in ordered]
    identifier = fingerprint([family, sorted(identifiers)])[:24]
    episode = detected_at[:13]
    return {
        "candidate_id": identifier, "family": family, "detected_at": detected_at,
        "transaction_ids": identifiers, "anchors": sorted(anchors),
        "case_key": fingerprint([family, sorted(anchors), episode])[:24],
        "first_time": ordered[0]["timestamp"], "last_time": ordered[-1]["timestamp"],
        "equal_timestamps": len({e["timestamp"] for e in ordered}) < len(ordered),
        "search_truncated": False, "events": ordered,
    }


def search_patterns(state, group, config):
    detected_at = group[0]["timestamp"]
    cutoff = parse_time(detected_at) - timedelta(hours=config["motif_window_hours"])
    budgets = []
    candidates = {}
    retained = []
    seen = set()

    def new_budget(limit):
        value = {"limit": limit, "scanned": 0, "truncated": False}
        budgets.append(value)
        return value

    def keep(family, events, anchors):
        if not any(event["timestamp"] == detected_at for event in events):
            return
        candidate = make_candidate(family, events, anchors, detected_at)
        identifier = candidate["candidate_id"]
        if identifier in seen:
            return
        seen.add(identifier)

        # При переполнении сохраняем выборку по устойчивому хешу, а не первые строки.
        rank = int(identifier, 16)
        maximum = config["candidates_per_group"]
        if len(candidates) >= maximum:
            if rank >= -retained[0][0]:
                return
            _, removed = heapq.heappop(retained)
            del candidates[removed]
        candidates[identifier] = candidate
        heapq.heappush(retained, (-rank, identifier))

    # Каждая новая операция и каждый затронутый счёт получают свой обход.
    seeds = sorted(group, key=lambda e: e["transaction_id"])

    # Идём назад от новой операции: каждое предыдущее звено не позже следующего.
    for seed in seeds:
        budget = new_budget(config["path_scan_limit"])
        source, target = seed["source_key"], seed["target_key"]
        if source == target:
            continue
        stack = [(source, [seed], {source, target})]
        saved_paths = 0

        while stack and saved_paths < config["paths_per_seed"]:
            node, path, visited = stack.pop()
            previous = recent_events(state, state.incoming, node, budget, cutoff, 16)
            for event in previous:
                if saved_paths >= config["paths_per_seed"]:
                    budget["truncated"] = True
                    break
                if event["timestamp"] > path[0]["timestamp"]:
                    continue
                earlier = event["source_key"]
                extended = [event] + path

                if earlier == target and len(extended) <= config["cycle_max_length"]:
                    family = "reciprocal" if len(extended) == 2 else "cycle"
                    keep(family, extended, {e["source_key"] for e in extended})
                    saved_paths += 1
                elif earlier not in visited:
                    if 3 <= len(extended) <= config["relay_max_edges"]:
                        keep("chain", extended, [earlier, target])
                        saved_paths += 1
                    if len(extended) < max(config["relay_max_edges"], config["cycle_max_length"]):
                        stack.append((earlier, extended, visited | {earlier}))

            if budget["scanned"] >= budget["limit"]:
                break

        if stack:
            budget["truncated"] = True

    centers = sorted({e[side] for e in seeds for side in ["source_key", "target_key"]})
    for center in centers:
        budget = new_budget(config["center_scan_limit"])
        incoming = recent_events(state, state.incoming, center, budget, cutoff, config["star_max_events"])
        outgoing = recent_events(state, state.outgoing, center, budget, cutoff, config["star_max_events"])
        minimum = config["star_min_neighbors"]

        if len({e["source_key"] for e in incoming}) >= minimum:
            keep("fan_in", incoming, [center])
        if len({e["target_key"] for e in outgoing}) >= minimum:
            keep("fan_out", outgoing, [center])

        if incoming and outgoing:
            earliest_out = min(e["timestamp"] for e in outgoing)
            earlier_inputs = [e for e in incoming if e["timestamp"] <= earliest_out]
            if len({e["source_key"] for e in earlier_inputs}) >= minimum and len({e["target_key"] for e in outgoing}) >= minimum:
                keep("gather_scatter", earlier_inputs + outgoing, [center])

    # Новое второе звено B -> C раскрывает ветви A -> B -> C.
    for seed in seeds:
        budget = new_budget(config["composition_scan_limit"])
        middle, destination = seed["source_key"], seed["target_key"]
        before = recent_events(state, state.incoming, middle, budget, cutoff, 8)
        for first in before:
            origin = first["source_key"]
            if first["timestamp"] > seed["timestamp"] or len({origin, middle, destination}) < 3:
                continue
            branches = {middle: [first, seed]}
            starts = recent_events(state, state.outgoing, origin, budget, cutoff, 12)
            for start in starts:
                via = start["target_key"]
                if via in {origin, destination} or via in branches:
                    continue
                pair = state.pair_events.get((via, destination), {})
                for identifier in islice(reversed(pair), 8):
                    if budget["scanned"] >= budget["limit"]:
                        budget["truncated"] = True
                        break
                    budget["scanned"] += 1
                    end = state.events[identifier]
                    if start["timestamp"] <= end["timestamp"] <= detected_at:
                        branches[via] = [start, end]
                        break
            if len(branches) >= 2:
                evidence = [event for branch in branches.values() for event in branch]
                keep("scatter_gather", evidence, [origin, destination])

    dropped = max(0, len(seen) - len(candidates))
    incomplete = any(b["truncated"] for b in budgets) or dropped > 0
    for candidate in candidates.values():
        candidate["search_truncated"] = bool(incomplete)
        candidate["features"] = structure_features(candidate)

    diagnostics = {
        "scanned": sum(b["scanned"] for b in budgets), "search_truncated": bool(incomplete),
        "skipped_seeds": 0, "skipped_centers": 0,
        "searched_seeds": len(seeds), "searched_centers": len(centers),
        "limited_walks": sum(b["truncated"] for b in budgets),
        "generated": len(seen), "retained": len(candidates),
        "candidate_limit_dropped": dropped,
    }
    return [candidates[key] for key in sorted(candidates)], diagnostics


def structure_features(candidate):
    events = candidate["events"]
    outgoing = Counter(e["source_key"] for e in events)
    incoming = Counter(e["target_key"] for e in events)
    nodes = set(outgoing) | set(incoming)
    pairs = {(e["source_key"], e["target_key"]) for e in events}
    duration = (parse_time(candidate["last_time"]) - parse_time(candidate["first_time"])).total_seconds()

    # Сначала сравниваем суммы внутри каждой валюты, затем объединяем отношения.
    amounts = defaultdict(list)
    for event in events:
        amounts[event["payment_currency"]].append(float(event["amount_paid"]))
    variation = []
    repeats = []
    for values in amounts.values():
        mean = np.mean(values)
        if len(values) >= 2 and mean > 0:
            variation.append(float(np.std(values) / mean))
            repeats.append(1 - len(set(values)) / len(values))

    return {
        "log_nodes": float(np.log1p(len(nodes))),
        "log_events": float(np.log1p(len(events))),
        "log_pairs": float(np.log1p(len(pairs))),
        "log_duration_seconds": float(np.log1p(duration)),
        "events_per_minute": len(events) / max(1, duration / 60),
        "sources": len(set(outgoing) - set(incoming)),
        "sinks": len(set(incoming) - set(outgoing)),
        "max_out_share": max(outgoing.values()) / len(events),
        "max_in_share": max(incoming.values()) / len(events),
        "repeat_pair_fraction": 1 - len(pairs) / len(events),
        "currency_count": len(amounts),
        "cross_currency_fraction": sum(e["payment_currency"] != e["receiving_currency"] for e in events) / len(events),
        "amount_cv": float(np.median(variation)) if variation else 0.0,
        "amount_repeat_fraction": float(np.median(repeats)) if repeats else 0.0,
        "amount_comparison_missing": int(not variation),
        "equal_timestamps": int(candidate["equal_timestamps"]),
        "search_truncated": int(candidate["search_truncated"]),
        "closed_path": int(candidate["family"] in {"cycle", "reciprocal"}),
    }
