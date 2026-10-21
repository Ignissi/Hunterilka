"""Граф случая для экрана: красное основание, серое окружение, отдельный перевод по клику."""

import html
import json
import math
import re
from collections import defaultdict

import networkx as nx
from pyvis.network import Network

from src.storage import short_hash


DISPLAY_NODES = 40
DISPLAY_EDGES = 80
EXPANDED_NODES = 100
EXPANDED_EDGES = 200


def positions_for_case(graph, evidence, family):
    """Раскладка по форме схемы: звезда, кольцо, слои или пружинная."""
    core = nx.DiGraph()
    core.add_edges_from((e["source_key"], e["target_key"]) for e in evidence
                        if e["source_key"] in graph and e["target_key"] in graph)
    if not core:
        return nx.spring_layout(graph, seed=42, scale=330, iterations=70)

    if family in {"fan_in", "fan_out"}:
        center = max(core, key=lambda node: core.degree(node))
        outer = sorted(set(core) - {center})
        anchors = {center: (0, 0)}
        for index, node in enumerate(outer):
            angle = 2 * math.pi * index / len(outer)
            anchors[node] = (270 * math.cos(angle), 270 * math.sin(angle))
    elif family in {"cycle", "reciprocal"}:
        anchors = {node: tuple(point * 210) for node, point in nx.circular_layout(core).items()}
    elif nx.is_directed_acyclic_graph(core):
        layers = list(nx.topological_generations(core))
        anchors = {}
        for layer, members in enumerate(layers):
            for index, node in enumerate(sorted(members)):
                anchors[node] = ((layer - (len(layers) - 1) / 2) * 190,
                                 (index - (len(members) - 1) / 2) * min(95, 500 / max(1, len(members))))
    else:
        anchors = {node: tuple(point * 230) for node, point in nx.spring_layout(core, seed=42).items()}

    if len(anchors) == len(graph):
        return anchors
    initial = dict(anchors)
    surrounding = sorted(set(graph) - set(anchors))
    for index, node in enumerate(surrounding):
        angle = 2 * math.pi * index / max(1, len(surrounding))
        initial[node] = (450 * math.cos(angle), 330 * math.sin(angle))
    return nx.spring_layout(graph, pos=initial, fixed=list(anchors), k=95, seed=42, iterations=70)


def case_graph(case, expanded=False, show_context=True, individual=True, selected_id=None):
    """Возвращает HTML графа и счётчики того, что показано."""
    node_limit = EXPANDED_NODES if expanded else DISPLAY_NODES
    edge_limit = EXPANDED_EDGES if expanded else DISPLAY_EDGES
    evidence_ids = set(case["transaction_ids"])
    context = case.get("context", []) if show_context else []
    all_events = {e["transaction_id"]: e for e in case["events"] + context}
    ordered = sorted(all_events.values(),
                     key=lambda e: (e["transaction_id"] not in evidence_ids, e["timestamp"], e["transaction_id"]))

    nodes, shown_events = set(), []
    pairs = defaultdict(list)
    for event in ordered:
        source, target = event["source_key"], event["target_key"]
        pair = (source, target)
        edge_count = len(shown_events) if individual else len(pairs)
        needs_edge = individual or pair not in pairs
        if len(nodes | {source, target}) > node_limit or (needs_edge and edge_count >= edge_limit):
            continue
        nodes.update([source, target])
        pairs[pair].append(event)
        shown_events.append(event)

    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    graph.add_edges_from(pairs)
    positions = positions_for_case(graph, case["events"], case["family"])
    network = Network(height="570px", width="100%", directed=True, bgcolor="#0a1422", font_color="#dce8fa",
                      cdn_resources="in_line")
    evidence_nodes = {e[side] for e in case["events"] for side in ["source_key", "target_key"]}
    details = {}

    for node in sorted(nodes):
        bank, account = node.split(":", 1)
        active = node in evidence_nodes
        x, y = positions[node]
        network.add_node(node, label=account[-8:],
                         title=html.escape("Банк " + bank + " · счёт " + account),
                         x=float(x), y=float(y), physics=False,
                         color={"background": "#402233" if active else "#162438",
                                "border": "#ff617d" if active else "#52637b",
                                "highlight": {"background": "#55303e", "border": "#ffabbb"}},
                         borderWidth=2.5 if active else 1, size=19 if active else 10,
                         font={"size": 14 if active else 11, "color": "#f3e9ef" if active else "#8c9eb5"},
                         shape="dot")

    for (source, target), events in pairs.items():
        groups = [[event] for event in events] if individual else [events]
        for index, group in enumerate(groups):
            red = any(e["transaction_id"] in evidence_ids for e in group)
            selected = selected_id is not None and any(e["transaction_id"] == selected_id for e in group)
            identifier = "e_" + short_hash([e["transaction_id"] for e in group])
            details[identifier] = [{
                "id": e["transaction_id"], "time": e["timestamp"].replace("T", " "),
                "from": e["source_bank"] + " / " + e["source_account"],
                "to": e["target_bank"] + " / " + e["target_account"],
                "amount": e["amount_paid"] + " " + e["payment_currency"],
                "format": e["payment_format"], "evidence": e["transaction_id"] in evidence_ids,
            } for e in group]
            parallel = len(groups) > 1 or graph.has_edge(target, source)
            curvature = .08 + .55 * (index + 1) / (len(groups) + 1) if parallel else .05
            label = str(len(group)) + " переводов" if len(group) > 1 else ""
            if red and len(case["events"]) <= 8 and len(group) == 1:
                label = group[0]["amount_paid"] + " " + group[0]["payment_currency"]
            network.add_edge(source, target, id=identifier,
                             color={"color": "#f8c473" if selected else ("#ff617d" if red else "#465871"),
                                    "opacity": 1 if red or selected else .48},
                             width=4.5 if selected else (2.6 if red else .9),
                             arrows={"to": {"enabled": True, "scaleFactor": .7}},
                             label=label,
                             title=html.escape("Нажмите: " + group[0]["amount_paid"] + " " + group[0]["payment_currency"]),
                             smooth={"enabled": parallel, "type": "curvedCW" if index % 2 == 0 else "curvedCCW",
                                     "roundness": curvature},
                             dashes=bool(case.get("equal_timestamps") and red))

    network.set_options(json.dumps({
        "physics": {"enabled": False}, "layout": {"improvedLayout": False},
        "interaction": {"hover": True, "zoomView": True, "dragView": True, "multiselect": False, "tooltipDelay": 120},
        "edges": {"font": {"color": "#ffc2cf", "size": 11, "strokeWidth": 4, "strokeColor": "#0a1422", "align": "middle"},
                  "selectionWidth": 1.5, "hoverWidth": 1.5},
        "nodes": {"font": {"face": "monospace"}},
    }))
    rendered = network.generate_html(notebook=False)
    # Внешние ссылки на CDN убираются: страница должна работать без интернета.
    rendered = re.sub(r'<link\b[^>]*href=["\x27]https?://[^>]*>', "", rendered)
    rendered = re.sub(r'<script\b[^>]*src=["\x27]https?://[^>]*>.*?</script>', "", rendered, flags=re.S)

    panel = '''<style>
html,body{margin:0!important;background:#0a1422;color:#dce8fa;font-family:Arial,sans-serif;overflow:hidden}
.card{border:0!important;background:transparent!important}#mynetwork{border:0!important;height:570px!important;background:radial-gradient(ellipse at 48% 43%,#142338 0%,#0a1422 72%)!important}
#graph-tools{position:absolute;top:14px;left:16px;display:flex;gap:6px;z-index:9}
#graph-tools button{background:#192a40;border:1px solid #3a4c66;color:#e6effc;border-radius:7px;padding:8px 12px;cursor:pointer}
#graph-tools button:hover{background:#2a3e59}#graph-hint{position:absolute;bottom:12px;left:16px;font-size:11px;color:#8399b6;pointer-events:none}
#edge-detail{display:none;position:absolute;right:12px;bottom:32px;width:310px;max-height:200px;overflow:auto;background:#101e30ed;border:1px solid #455774;border-radius:10px;padding:14px;font-size:12px;line-height:1.7;white-space:pre-wrap;word-break:break-word;z-index:10}
</style><div id="graph-tools"><button id="graph-fit">Вписать граф</button><button id="graph-plus">+</button><button id="graph-minus">−</button></div>
<div id="edge-detail"></div><div id="graph-hint">Перетащите вершину · колёсико: масштаб · нажмите ребро: детали перевода</div>'''
    payload = json.dumps(details, ensure_ascii=False).replace("</", "<\\/")
    script = '''<script>
const transferDetails = PAYLOAD;
document.getElementById('graph-fit').onclick = () => network.fit({animation:{duration:300}});
document.getElementById('graph-plus').onclick = () => network.moveTo({scale:network.getScale()*1.25});
document.getElementById('graph-minus').onclick = () => network.moveTo({scale:network.getScale()/1.25});
network.on('click', function(params) {
  const panel = document.getElementById('edge-detail');
  const records = params.edges.length ? transferDetails[params.edges[0]] : null;
  panel.style.display = records ? 'block' : 'none';
  if (records) panel.textContent = records.map(row => (row.evidence ? 'ОСНОВАНИЕ СЛУЧАЯ' : 'ОКРУЖЕНИЕ') + '\\n' + row.from + ' → ' + row.to + '\\n' + row.amount + ' · ' + row.format + '\\n' + row.time + '\\nID: ' + row.id).join('\\n\\n');
});
network.fit({animation:false});
</script>'''.replace("PAYLOAD", payload)
    rendered = rendered.replace("</body>", panel + script + "</body>")

    counts = {
        "shown": len(shown_events), "total": len(all_events), "nodes": len(nodes),
        "evidence_shown": sum(e["transaction_id"] in evidence_ids for e in shown_events),
    }
    return rendered, counts
