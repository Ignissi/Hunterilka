"""Ограниченные обходы большого графа: короткий обратный путь и окружение случая."""

from collections import deque
from itertools import islice


def short_return_path(graph, source, target, budget):
    """Есть ли путь target -> ... -> source длиной до двух рёбер.

    Возвращает (найден, обход завершён). Если бюджет просмотра исчерпан,
    ответ «не найден» не является доказательством отсутствия пути.
    """
    if source == target:
        return False, True

    queue = deque([(target, 0)])
    visited = {target}
    scanned = 0

    while queue:
        node, depth = queue.popleft()
        if node not in graph or depth >= 2:
            continue
        for neighbor in graph.successors(node):
            if scanned >= budget:
                return False, False
            scanned += 1
            if neighbor == source:
                return True, True
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return False, True


def snapshot_context(state, evidence, config):
    """Соседние переводы вокруг основания случая: два шага по графу, с лимитами.

    Берутся только события, которые уже есть в графе, то есть прошлое на момент
    обнаружения. Будущих переводов в окружении быть не может.
    """
    selected = {event["transaction_id"]: event.copy() for event in evidence}
    nodes = {e[side] for e in evidence for side in ["source_key", "target_key"]}
    scanned = 0
    truncated = False
    frontier = sorted(nodes)

    for _ in range(2):
        next_frontier = []
        for node in frontier:
            for index in [state.incoming, state.outgoing]:
                identifiers = index.get(node, {})
                allowance = min(32, max(0, config["context_scan_limit"] - scanned))
                if len(identifiers) > allowance:
                    truncated = True
                for identifier in islice(reversed(identifiers), allowance):
                    scanned += 1
                    event = state.events[identifier]
                    new_nodes = {event["source_key"], event["target_key"]} - nodes
                    if len(nodes | new_nodes) > config["context_nodes"] or len(selected) >= config["context_edges"]:
                        truncated = True
                        continue
                    selected[identifier] = event.copy()
                    nodes.update(new_nodes)
                    next_frontier.extend(sorted(new_nodes))
        frontier = next_frontier

    return list(selected.values()), truncated
