"""На экран попадает только сохранённый ограниченный граф случая."""

import html
import json
import re
from collections import defaultdict
from decimal import Decimal

import networkx as nx
from pyvis.network import Network

from src.storage import fingerprint


def case_graph(case, config, expanded=False):
    node_limit = 100 if expanded else config["display_nodes"]
    edge_limit = 200 if expanded else config["display_edges"]
    evidence_ids = set(case["transaction_ids"])
    all_events = {e["transaction_id"]: e for e in case["events"] + case.get("context", [])}
    ordered = sorted(all_events.values(), key=lambda e: (e["transaction_id"] not in evidence_ids, e["timestamp"], e["source_row"]))
    nodes = set()
    pairs = defaultdict(list)
    shown = 0

    for event in ordered:
        source, target = event["source_key"], event["target_key"]
        pair = (source, target)
        if len(nodes | {source, target}) > node_limit or (pair not in pairs and len(pairs) >= edge_limit):
            continue
        nodes.update([source, target])
        pairs[pair].append(event)
        shown += 1

    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(pairs)
    if case["family"] in {"cycle", "reciprocal"}:
        positions = nx.circular_layout(graph)
    else:
        positions = nx.spring_layout(graph, seed=42, iterations=60)

    network = Network(height="480px", width="100%", directed=True, bgcolor="#0e1828", font_color="#d9e5f5", cdn_resources="in_line")
    display_ids = {node: fingerprint(node)[:16] for node in nodes}
    evidence_nodes = {e[side] for e in case["events"] for side in ["source_key", "target_key"]}
    for node in sorted(nodes):
        bank, account = json.loads(node)
        label = bank + " / " + account[-6:]
        x, y = positions[node]
        network.add_node(
            display_ids[node], label=html.escape(label), title=html.escape("Банк: " + bank + "\nСчёт: " + account),
            x=float(x * 240), y=float(y * 190), physics=False,
            color={"background": "#17263d", "border": "#f0b35c" if node in evidence_nodes else "#61738b"},
            borderWidth=2, size=min(24, 14 + graph.degree(node)), shape="dot",
        )

    for (source, target), events in pairs.items():
        red = any(e["transaction_id"] in evidence_ids for e in events)
        totals = defaultdict(Decimal)
        for event in events:
            totals[event["payment_currency"]] += Decimal(event["amount_paid"])
        amount_text = "; ".join(str(amount) + " " + currency for currency, amount in totals.items())
        network.add_edge(
            display_ids[source], display_ids[target], color="#f35c69" if red else "#536278",
            width=min(5, 1.2 + len(events) * 0.4), arrows="to",
            label=str(len(events)), title=html.escape(amount_text),
            dashes=bool(case["equal_timestamps"] and red),
        )
    network.set_options(json.dumps({
        "physics": {"enabled": False},
        "interaction": {"hover": True, "navigationButtons": False, "zoomView": True},
        "edges": {"smooth": {"enabled": True, "type": "curvedCW", "roundness": 0.12}, "font": {"color": "#acbad0", "size": 12, "strokeWidth": 0}},
        "nodes": {"font": {"size": 13, "face": "monospace"}},
    }))
    rendered = network.generate_html(notebook=False)
    # PyVis встраивает vis-network, но шаблон добавляет ненужный Bootstrap CDN.
    rendered = re.sub(r'<link\b[^>]*href=["\x27]https?://[^>]*>', "", rendered)
    rendered = re.sub(r'<script\b[^>]*src=["\x27]https?://[^>]*>.*?</script>', "", rendered, flags=re.S)
    return rendered, {"shown": shown, "total": len(all_events), "nodes": len(nodes), "pairs": len(pairs)}
