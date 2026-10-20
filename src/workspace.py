"""Рабочий экран: период, каталог схем, рассмотрение и обзор участков сети."""

import json
import shutil
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.cases import CaseService
from src.patterns import FAMILY_NAMES
from src.review_queue import ordered_cases
from src.snapshot import WORKSPACE_VERSION, graph_slice
from src.storage import ROOT, fingerprint, read_json, write_json
from src.visualization import case_graph


SCHEME_TEXT = {
    "chain": "Средства проходят последовательно через несколько счетов.",
    "cycle": "Направленный путь возвращается к исходному счёту.",
    "reciprocal": "Переводы между двумя счетами идут в обоих направлениях.",
    "fan_in": "Несколько отправителей переводят средства одному получателю.",
    "fan_out": "Один отправитель распределяет средства между несколькими получателями.",
    "gather_scatter": "Средства сходятся на промежуточном счёте и затем распределяются дальше.",
    "scatter_gather": "Поток расходится по промежуточным счетам и затем снова сходится.",
    "transaction": "Выделена отдельная операция. Серые связи показывают доступное прошлое её участников.",
}


@st.cache_data(show_spinner=False)
def workspace_table(name, revision):
    return pd.read_parquet(ROOT / "demo" / name)


def open_saved_cases(demo):
    base = ROOT / "runs/demo_review" / demo["model_id"]
    destination = base / fingerprint(demo)[:12] / "cases"
    if not destination.exists():
        previous = list(base.rglob("index.parquet")) if base.exists() else []
        shutil.copytree(ROOT / "demo/cases", destination)
        service = CaseService(destination)
        # При новом оформлении сохраняем последнее решение по прежнему идентификатору случая.
        for path in previous:
            old = CaseService(path.parent)
            for identifier in set(old.index) & set(service.index):
                source = old.get(identifier)
                current = service.get(identifier)
                if source.get("reviewed_at") and source["reviewed_at"] > (current.get("reviewed_at") or ""):
                    for field in ["status", "comment", "reviewed_at"]:
                        current[field] = source[field]
                    write_json(destination / (identifier + ".json"), current)
                    service.index[identifier]["status"] = current["status"]
        service.flush()
    return CaseService(destination)


def choose_case(service, table, namespace):
    identifiers = table.case_id.tolist()
    key = namespace + "_case"
    if st.session_state.get(key) not in identifiers:
        st.session_state[key] = identifiers[0]

    def move(step):
        position = identifiers.index(st.session_state[key])
        st.session_state[key] = identifiers[max(0, min(len(identifiers) - 1, position + step))]

    def label(identifier):
        row = service.index[identifier]
        route = row.get("route", identifier[:10])
        return FAMILY_NAMES[row["family"]] + " · " + route + " · " + str(row["operations"]) + " переводов · " + row["detected_at"][5:16].replace("T", " ") + " · " + row["status"] + " · " + identifier[:6]

    current = identifiers.index(st.session_state[key])
    left, center, right = st.columns([1, 8, 1], vertical_alignment="bottom")
    left.button("← Назад", key=namespace + "_prev", disabled=current == 0, on_click=move, args=(-1,), use_container_width=True)
    identifier = center.selectbox("Открытый случай · можно искать по счёту или номеру", identifiers, key=key, format_func=label)
    right.button("Далее →", key=namespace + "_next", disabled=current == len(identifiers) - 1, on_click=move, args=(1,), use_container_width=True)
    st.caption("Карточка " + str(identifiers.index(identifier) + 1) + " из " + str(len(identifiers)) + ". Выбор сохраняется при переключении вида графа.")
    with st.expander("Список случаев: нажмите строку, чтобы открыть", expanded=False):
        visible = table.copy()
        visible["family"] = visible.family.map(FAMILY_NAMES)
        columns = [name for name in ["family", "route", "operations", "accounts", "operation_score", "anomaly_score", "status"] if name in visible]
        labels = {"family": "Схема", "route": "Участники", "operations": "Переводы", "accounts": "Счета",
                  "operation_score": "CatBoost", "anomaly_score": "Необычность", "status": "Рассмотрение"}
        table_key = namespace + "_table_" + fingerprint(identifiers)[:8]

        def select_row():
            selected = st.session_state[table_key].selection.rows
            if selected:
                st.session_state[key] = identifiers[selected[0]]

        st.dataframe(visible[columns].rename(columns=labels), hide_index=True, use_container_width=True,
                     height=300, key=table_key, on_select=select_row, selection_mode="single-row")
    return identifier


def show_network(service, config, first, last, revision):
    nodes = workspace_table("network_nodes.parquet", revision)
    sample = workspace_table("network_sample.parquet", revision)
    workspace = read_json(ROOT / "demo/workspace_manifest.json")
    st.markdown("### Карта сети: выберите счёт и раскройте его окружение")
    st.caption("Все счета доступны поиску. На рисунке ограниченный участок, а счётчики периода выше охватывают всю подготовленную историю.")
    search = st.text_input("Поиск счёта или банка", key="network_search", placeholder="Номер счёта целиком или его часть")
    if search:
        options = nodes.loc[nodes.number.str.contains(search, case=False, regex=False) | nodes.bank.str.contains(search, case=False, regex=False)].head(100)
    else:
        seeds = [row["account"] for row in workspace["seeds"] if first <= row["day"] <= last]
        options = nodes.loc[nodes.account.isin(seeds)].copy()
    if options.empty:
        st.info("Счёт не найден. Попробуйте другую часть номера.")
        return
    labels = {r.account: r.bank + " / " + r.number for r in options.itertuples()}
    seed = st.selectbox("Счёт для среза", options.account.tolist(), format_func=lambda value: labels[value], key="network_seed")
    local = (ROOT / "data/processed/events.parquet").exists()
    if not local:
        st.caption("Открыты сохранённые участки. Произвольное окружение доступно после подготовки полного CSV.")
    if st.button("Показать срез сети", type="primary", key="network_build"):
        start, end = first + "T00:00:00", last + "T23:59:59"
        with st.spinner("Извлекаю связи выбранного счёта..."):
            if local:
                path = ROOT / "data/processed/events.parquet"
                time_filter = [("timestamp", ">=", start), ("timestamp", "<=", end)]
                adjacent = pd.read_parquet(path, filters=[time_filter + [("source_key", "=", seed)], time_filter + [("target_key", "=", seed)]])
                recent = adjacent.tail(32)
                neighbors = sorted(set(recent.source_key) | set(recent.target_key))
                if neighbors:
                    sample = pd.read_parquet(path, filters=[time_filter + [("source_key", "in", neighbors)], time_filter + [("target_key", "in", neighbors)]])
                else:
                    sample = adjacent
            rows = graph_slice(sample, start, end, seed) if not sample.empty else []
            # Красным отмечаем лишь операции сохранённых случаев, наблюдаемых к концу периода.
            highlighted = set()
            for item in service.index.values():
                if first <= item["detected_at"][:10] <= last and not item.get("is_example", False):
                    highlighted.update(service.get(item["case_id"])["transaction_ids"])
            case = {"family": "network", "transaction_ids": [e["transaction_id"] for e in rows if e["transaction_id"] in highlighted],
                    "events": [e for e in rows if e["transaction_id"] in highlighted],
                    "context": [e for e in rows if e["transaction_id"] not in highlighted], "equal_timestamps": False}
            st.session_state.network_view = {"case": case, "selection": [first, last, seed]}
    current = st.session_state.get("network_view")
    if current and current["selection"] == [first, last, seed]:
        case = current["case"]
        if not case["events"] and not case["context"]:
            st.info("В выбранном периоде у этого счёта нет доступных переводов.")
        else:
            rendered, info = case_graph(case, config, expanded=True)
            components.html(rendered, height=585, scrolling=False)
            st.caption(f'В срезе {info["nodes"]} счетов и {info["shown"]} переводов. Красное: операции сохранённых случаев; серое: остальные связи. Непоказанные связи не считаются отсутствующими.')


def render_workspace(service, config, show_case, demo=None):
    st.session_state.service = service
    if not service.index:
        st.info("Случаев пока нет. История накапливается.")
        return
    all_cases = pd.DataFrame(list(service.index.values()))
    revision = fingerprint(demo) if demo else "live"
    prepared = demo is not None and demo.get("workspace_version") == WORKSPACE_VERSION and (ROOT / "demo/workspace_manifest.json").exists()
    if prepared:
        passport = read_json(ROOT / "demo/workspace_manifest.json")
        prepared = passport["model_id"] == demo["model_id"] and passport["data_hash"] == demo["data"]["events_sha256"]
    data = demo["data"] if demo else {"first_time": all_cases.detected_at.min(), "last_time": all_cases.detected_at.max()}
    lower, upper = date.fromisoformat(data["first_time"][:10]), date.fromisoformat(data["last_time"][:10])
    a, b, c = st.columns([2, 2, 3], vertical_alignment="bottom")
    start = a.date_input("Период с", lower, min_value=lower, max_value=upper, key="period_from_" + revision[:8])
    finish = b.date_input("Период по", upper, min_value=lower, max_value=upper, key="period_to_" + revision[:8])
    c.caption("Окно истории: " + str(config["window_days"]) + " дней · загружено " + (f'{data["history_days"]:.2f}' if "history_days" in data else "часть") + " дня. Случай показывает связи на момент обнаружения.")
    if "validation_end" in data:
        c.caption("Оценка новых операций на test с " + data["validation_end"].replace("T", " ") + ". Более ранняя история используется для обучения и прогрева.")
    if finish < start:
        st.info("Начало периода должно быть не позже окончания.")
        return
    first, last = start.isoformat(), finish.isoformat()
    period_cases = all_cases.loc[all_cases.detected_at.str[:10].between(first, last)]
    schemes = None
    if prepared:
        daily = workspace_table("periods.parquet", revision)
        daily = daily.loc[daily.day.between(first, last)]
        schemes = workspace_table("schemes.parquet", revision)
        schemes = schemes.loc[schemes.day.between(first, last)]
        counts = [int(daily.operations.sum()), int(daily.operation_signals.sum()),
                  int((period_cases.status == "Новый").sum()), int((period_cases.status == "Требует проверки").sum()),
                  int((period_cases.status == "Рассмотрен").sum())]
        names = ["Операций в периоде", "Сигналов CatBoost", "Новых карточек", "На проверке", "Рассмотрено"]
        for column, name, count in zip(st.columns(5), names, counts):
            column.metric(name, f"{count:,}".replace(",", " "))
        with st.expander("Динамика по дням, схемы и типы переводов"):
            left, right = st.columns(2)
            left.markdown("**Все операции по дням**")
            left.bar_chart(daily.set_index("day")[["operations"]].rename(columns={"operations": "Операции"}), color="#7197d5", height=180)
            right.markdown("**Пороговые сигналы CatBoost по дням**")
            right.bar_chart(daily.set_index("day")[["operation_signals"]].rename(columns={"operation_signals": "Сигналы"}), color="#ff617d", height=180)
            table = schemes.groupby("family")[["candidates", "episodes"]].sum().reindex([k for k in FAMILY_NAMES if k != "transaction"], fill_value=0)
            table.index = table.index.map(FAMILY_NAMES)
            st.dataframe(table.rename(columns={"candidates": "Версий кандидатов", "episodes": "Уникальных эпизодов"}), use_container_width=True)
            st.caption("Это найденные формы до отбора моделью, а не подтверждённое мошенничество. Статусы относятся к сохранённым карточкам. Оценки CatBoost считаются только на test; более ранняя история служит прогревом.")
            formats = workspace_table("formats.parquet", revision)
            formats = formats.loc[formats.day.between(first, last)]
            st.dataframe(formats.groupby("payment_format").operations.sum().rename("Операций").rename_axis("Формат перевода"), use_container_width=True)
    elif demo is not None:
        st.info("Для окружения и обзора всей истории выполните: python -m scripts.build_workspace")

    view = st.radio("Представление", ["Случаи и схемы", "Обзор сети"], horizontal=True, key="workspace_view_" + revision[:8]) if prepared else "Случаи и схемы"
    if view == "Обзор сети":
        show_network(service, config, first, last, revision)
        return
    a, b, c = st.columns([2, 3, 2])
    channel = a.radio("Очередь", ["Структуры", "Операции"], horizontal=True, key="queue_channel_" + revision[:8])
    queue = "structures" if channel == "Структуры" else "operations"
    table = ordered_cases(period_cases.to_dict("records"), queue)
    family_names = [key for key in FAMILY_NAMES if key != "transaction"] if queue == "structures" else []
    family_counts = period_cases.family.value_counts()
    family = b.selectbox("Форма связанных переводов", ["Все"] + family_names,
                        format_func=lambda value: "Все формы" if value == "Все" else FAMILY_NAMES[value] + " · " + str(family_counts.get(value, 0)) + " карточек",
                        key="scheme_" + queue + revision[:8], disabled=queue == "operations")
    status = c.selectbox("Статус рассмотрения", ["Все", "Новый", "Требует проверки", "Рассмотрен"], key="status_" + queue + revision[:8])
    if family != "Все" and not table.empty:
        table = table.loc[table.family == family]
        st.caption(SCHEME_TEXT[family])
    if status != "Все" and not table.empty:
        table = table.loc[table.status == status]
    if table.empty:
        st.info("В этом периоде и фильтре нет сохранённых карточек. Полные числа найденных форм доступны в динамике выше.")
        return
    st.caption("Структуры рассматриваются отдельно от операций. Примеры форм имеют явную пометку; необычность не означает доказанное нарушение." if queue == "structures" else "Первые 500 операций по оценке CatBoost. Карточки ниже рабочего порога отмечены отдельно.")
    identifier = choose_case(service, table, queue + "_" + revision[:8])
    show_case(service.get(identifier))
