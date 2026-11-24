"""Trace the first path search on the unmodified application code (figure 11)."""

import ast
import inspect
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import groupby

from src import patterns
from src.graph import RollingGraph
from src.pipeline import process_group
from src.storage import load_config
from tests.helpers import event


# Orientation matches the questions in the diagram. The graph program tests
# three of them in the opposite direction as an early continue or break.
CHOICES = {
    "D1": ("source == target", True),
    "D2": ("event['timestamp'] > path[0]['timestamp']", False),
    "D3": ("earlier == target and len(extended) <= config['cycle_max_length']", True),
    "D4": ("earlier not in visited", False),
    "D5": ("3 <= len(extended) <= config['relay_max_edges']", True),
    "D6": ("len(extended) < max(config['relay_max_edges'], config['cycle_max_length'])", True),
    "L2c": ("budget['scanned'] >= budget['limit']", False),
    "L3": ("saved_paths >= config['paths_per_seed']", False),
}
DECISIONS = ("D1", "D2", "D3", "D4", "D5", "D6", "L2a", "L2b", "L2c", "L3")


def timestamp(minutes):
    return (datetime(2026, 1, 1) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def transfer(number, source, target, minute):
    return event(number, source, target, timestamp(minute))


def route(nodes):
    rows = [transfer(i + 1, a, b, i) for i, (a, b) in enumerate(zip(nodes, nodes[1:]))]
    return rows[:-1], rows[-1:]


def paths():
    reciprocal = [("reciprocal", 2)]
    chains = [("chain", n) for n in (3, 4, 5)]
    many_pairs = [transfer(i + 1, "B", "A", 0) for i in range(14)]
    many_chains = [transfer(i, f"S{i}", f"M{i}", 0) for i in range(1, 15)]
    many_chains += [transfer(100 + i, f"M{i}", "H", 1) for i in range(1, 15)]
    crowd = [transfer(i + 1, f"M{i}", "H", 1) for i in range(16)]
    crowd += [transfer(100 + i * 10 + j, f"X{i}_{j}", f"M{i}", 2) for i in range(16) for j in range(10)]
    # Each row gives its own oracle: reported structures, completeness, a
    # specific observed outcome, and (where needed) an exact boundary witness.
    return [
        ("A → A", ([], [transfer(1, "A", "A", 0)]), [], False, ("D1", True), {}),
        ("B → A, затем A → B", route("BAB"), reciprocal, False, ("D3", True), {"closure_lengths": 2}),
        ("Кольцо из трёх переводов", route("ABCA"), [("cycle", 3)], False, ("D3", True), {"closure_lengths": 3}),
        ("B → C, затем A → B, затем C → A", ([transfer(1, "B", "C", 1), transfer(2, "A", "B", 2)],
         [transfer(3, "C", "A", 3)]), [], False, ("D2", False), {}),
        ("A → B → C → D", route("ABCD"), [("chain", 3)], False, ("D5", True), {}),
        ("C → B → C → D", route("CBCD"), [], False, ("D4", True), {}),
        ("Цепь из семи переводов", route("ABCDEFGH"), chains, False, ("D6", False), {"stopped_lengths": 6}),
        ("Кольцо из шести переводов", route("ABCDEFA"), chains + [("cycle", 6)], False,
         ("D3", True), {"closure_lengths": 6}),
        ("Кольцо из семи переводов", route("ABCDEFGA"), chains, False, ("D6", False),
         {"stopped_lengths": 6, "max_extended": 6}),
        ("14 обратных переводов к одной операции", (many_pairs, [transfer(999, "A", "B", 1)]),
         reciprocal * 12, True, ("L3", False), {"inner_path_cap": 12}),
        ("14 двухзвенных путей через H, затем H → T", (many_chains, [transfer(999, "H", "T", 2)]),
         [("chain", 3)] * 12, True, ("L2b", False), {"outer_path_cap": 12}),
        ("16 входов в H, по 10 более поздних входов в их истоки", (crowd, [transfer(9999, "H", "T", 3)]),
         [], True, ("L2c", False), {"scan_cap": 128}),
        ("Встречный перевод через 24 ч 1 мин", ([transfer(1, "B", "A", 0)], [transfer(2, "A", "B", 1441)]),
         [], False, ("L2a", False), {"ages_minutes": 1441}),
        ("Встречный перевод ровно через 24 ч", ([transfer(1, "B", "A", 0)], [transfer(2, "A", "B", 1440)]),
         reciprocal, False, ("D3", True), {"ages_minutes": 1440}),
        ("Два встречных перевода одной минуты", ([], [transfer(1, "A", "B", 0), transfer(2, "B", "A", 0)]),
         reciprocal, False, ("D2", True), {"equal_time": True}),
    ]


def decision_sites():
    source, first = inspect.getsourcelines(patterns.search_patterns)
    tree = ast.parse("".join(source))
    while_node, = [n for n in ast.walk(tree) if isinstance(n, ast.While)]
    seed_loop, = [n for n in ast.walk(tree) if isinstance(n, ast.For) and while_node in n.body]
    branches = {ast.unparse(n.test): n for n in ast.walk(seed_loop) if isinstance(n, ast.If)}
    sites = {}
    for key, (expression, orientation) in CHOICES.items():
        node = branches[expression]
        sites[first + node.lineno - 1] = (key, first + node.body[0].lineno - 1, orientation)
    return sites, first + while_node.lineno - 1


class PathTrace:
    """Only follow search_patterns and its call to recent_events from the path loop."""

    def __init__(self):
        self.sites, self.loop = decision_sites()
        self.results = defaultdict(set)
        self.witness = defaultdict(set)
        self.source = patterns.search_patterns.__code__
        recent_source, first = inspect.getsourcelines(patterns.recent_events)
        self.age_line = first + next(i for i, line in enumerate(recent_source) if " < cutoff:" in line)
        source, first = inspect.getsourcelines(patterns.search_patterns)
        self.recent_call = first + next(i for i, line in enumerate(source) if "previous = recent_events" in line)

    def __call__(self, frame, event_name, arg):
        if event_name != "call":
            return None
        if frame.f_code is patterns.recent_events.__code__:
            caller = frame.f_back
            if caller.f_code is self.source and caller.f_lineno == self.recent_call:
                return self.age_trace
        if frame.f_code is not self.source:
            return None
        pending = None

        def trace(frame, event_name, arg):
            nonlocal pending
            if event_name != "line":
                return trace
            line, values = frame.f_lineno, frame.f_locals
            if pending:
                key, body, orientation, evidence = pending
                took = (line == body) == orientation
                self.results[key].add(took)
                if key == "D3" and took:
                    self.witness["closure_lengths"].add(evidence)
                if key == "D6" and not took:
                    self.witness["stopped_lengths"].add(evidence)
                if key == "L2c" and not took:
                    self.witness["scan_cap"].add(evidence)
                if key == "L3" and not took:
                    self.witness["inner_path_cap"].add(evidence)
                pending = None
            if line == self.loop:
                stack = bool(values["stack"])
                self.results["L2a"].add(stack)
                if stack:  # Python evaluates the second operand only then.
                    count = values["saved_paths"]
                    below = count < values["config"]["paths_per_seed"]
                    self.results["L2b"].add(below)
                    if not below:
                        self.witness["outer_path_cap"].add(count)
            if line in self.sites:
                key, body, orientation = self.sites[line]
                evidence = None
                if key in ("D3", "D6"):
                    evidence = len(values["extended"])
                    self.witness["extension_lengths"].add(evidence)
                elif key == "L2c":
                    evidence = values["budget"]["scanned"]
                elif key == "L3":
                    evidence = values["saved_paths"]
                elif key == "D2":
                    equal = values["event"]["timestamp"] == values["path"][0]["timestamp"]
                    self.witness["equal_time"].add(equal)
                pending = key, body, orientation, evidence
            return trace
        return trace

    def age_trace(self, frame, event_name, arg):
        if event_name == "line" and frame.f_lineno == self.age_line:
            when = datetime.fromisoformat(frame.f_locals["event"]["timestamp"])
            now = datetime.fromisoformat(frame.f_back.f_locals["detected_at"])
            self.witness["ages_minutes"].add(int((now - when).total_seconds() // 60))
        return self.age_trace


def path_checks():
    config = load_config()
    results = []
    for index, (title, (old, group), expected, cutoff, target, required) in enumerate(paths(), 1):
        state = RollingGraph(config)
        for _, batch in groupby(sorted(old, key=lambda e: e["timestamp"]), key=lambda e: e["timestamp"]):
            process_group(state, list(batch), config)
        capture = PathTrace()
        previous = sys.gettrace()
        try:
            sys.settrace(capture)
            _, _, found, diagnostics = process_group(state, group, config)
        finally:
            sys.settrace(previous)
        actual = [(c["family"], len(c["transaction_ids"])) for c in found
                  if c["family"] in ("reciprocal", "cycle", "chain")]
        subsets = capture.witness
        if subsets["extension_lengths"]:
            subsets["max_extended"].add(max(subsets["extension_lengths"]))
        decision, value = target
        passed = (Counter(expected) == Counter(actual) and diagnostics["search_truncated"] == cutoff
                  and value in capture.results[decision] and all(v in subsets[k] for k, v in required.items()))
        results.append(dict(id=f"БЯ-{index}", input=title, target=target, expected=sorted(expected),
                            observed=sorted(actual), expected_limited=cutoff, limited=diagnostics["search_truncated"],
                            decisions={k: sorted(v) for k, v in capture.results.items()},
                            required_witness=required, witness={k: sorted(v) for k, v in subsets.items()},
                            passed=passed))
    coverage = {k: {v: [r["id"] for r in results if value in r["decisions"].get(k, [])]
                    for v, value in (("yes", True), ("no", False))} for k in DECISIONS}
    return results, coverage
