"""Рабочее место AML-специалиста. Читает случаи из PostgreSQL; вся математика в src/."""

import json
from datetime import date, timedelta

import pandas as pd
import psycopg
import streamlit as st
import streamlit.components.v1 as components

from src import db
from src.models import explain_transaction, load_models
from src.patterns import FAMILY_NAMES
from src.report import number
from src.storage import ROOT, read_json
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
STATUSES = ["Новый", "Требует проверки", "Рассмотрен"]
STRUCTURE_FAMILIES = [key for key in FAMILY_NAMES if key != "transaction"]

st.set_page_config(page_title="Hunterилка · графовый AML", page_icon="◈", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {padding-top:1.1rem; max-width:1600px;}
h1 {font-size:2rem !important; letter-spacing:-.05em; font-weight:650 !important;}
h2 {font-size:1.3rem !important; letter-spacing:-.025em;}
h3 {font-size:1.05rem !important;}
div[data-testid="stMetric"] {border-top:1px solid #30415b; padding:14px 0 5px;}
div[data-testid="stMetricValue"] {font-size:1.6rem;}
.eyebrow {font:12px monospace;letter-spacing:.17em;color:#f35c69;margin:0 0 10px;}
.subline {color:#9aaec9;font-size:15px;margin-bottom:24px;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">HUNTER / GRAPH INTELLIGENCE / V2</div>', unsafe_allow_html=True)
st.title("Hunterилка · граф финансовых операций")
st.markdown('<div class="subline">Поток операций → динамический граф → два уровня анализа → случай → решение специалиста.</div>',
            unsafe_allow_html=True)


@st.cache_resource
def cached_models():
    return load_models()


def day_bounds(start, finish):
    """Границы периода в виде строк времени: [начало первого дня, начало дня после последнего)."""
    return start.isoformat() + "T00:00:00", (finish + timedelta(days=1)).isoformat() + "T00:00:00"


def show_time(value):
    return value.replace("T", " ") if value else "—"


# ---------- подключение ----------

try:
    with db.connect() as conn:
        db.init_schema(conn)
        total_cases = db.count_cases(conn)
        total_transactions = db.count_transactions(conn)
        first_case, last_case = db.case_time_range(conn)
except psycopg.OperationalError as error:
    st.error("База данных недоступна. Запустите PostgreSQL: docker compose up -d")
    st.code(str(error))
    st.stop()

models = None
try:
    models = cached_models()
except (FileNotFoundError, ValueError, OSError) as error:
    st.warning("Модели не загружены, объяснение SHAP недоступно: " + str(error))

with st.sidebar:
    st.markdown("## Состояние")
    st.metric("Операций в базе", f"{total_transactions:,}".replace(",", " "))
    st.metric("Случаев", f"{total_cases:,}".replace(",", " "))
    if st.button("Обновить"):
        st.rerun()
    st.divider()
    st.caption("CatBoost оценивает новую операцию по прошлому. Isolation Forest оценивает связанную структуру. "
               "Оценки относятся к разным объектам и не складываются.")
    st.caption("Красный: основание случая. Серый: соседние переводы. Янтарный: выделенный перевод.")


@st.fragment(run_every="2s")
def live_status():
    with db.connect() as conn:
        progress = db.stream_state(conn)
        count = db.count_cases(conn)
    if progress:
        mode = "Сохранённая демонстрационная выборка" if progress["kind"] == "demo" else "Поток операций"
        st.caption(mode + " · обработано по " + (progress["last_time"] or "начало прогрева") + " · случаев: " + str(count))
        if count != total_cases:
            st.info("В базе появились изменения. Обновите очередь после сохранения текущего комментария.")
            if st.button("Обновить очередь"):
                st.rerun()


live_status()


# ---------- карточка случая ----------

@st.cache_data(show_spinner=False, max_entries=48)
def render_case_graph(case, expanded, context, individual, selected):
    return case_graph(case, expanded, context, individual, selected)


def show_case(case, reviews, versions):
    family = FAMILY_NAMES.get(case["family"], case["family"])
    participants = {e[side] for e in case["events"] for side in ["source_key", "target_key"]}
    st.markdown("### " + family + " · " + str(len(participants)) + " счетов · " + str(len(case["events"])) + " переводов")
    st.caption(SCHEME_TEXT.get(case["family"], "Связанный участок сети.") + " Обнаружено " + show_time(case["detected_at"])
               + " · случай " + case["case_id"][:12] + " · версия " + str(case["version"]))

    a, b, c = st.columns([2, 2, 3])
    a.metric("CatBoost · максимальная оценка операции", "н/д" if case["operation_score"] is None else f'{case["operation_score"]:.4f}')
    b.metric("Isolation Forest · необычность структуры", "н/д" if case["anomaly_score"] is None else f'{case["anomaly_score"]:.4f}')
    grounds = []
    if case["operation_signal"]:
        grounds.append("сигнал CatBoost")
    if case["structure_signal"]:
        grounds.append("сигнал Isolation Forest")
    c.markdown("**Основание:** " + " + ".join(grounds))
    c.caption("Статус: " + case["status"])

    controls = st.columns([2, 2, 3, 5])
    context = controls[0].toggle("Серое окружение", value=True, key="context_" + case["case_id"],
                                 disabled=not bool(case.get("context")))
    expanded = controls[1].toggle("Расширенный срез", value=len(participants) > 40, key="expanded_" + case["case_id"])
    edge_mode = controls[2].selectbox("Рёбра графа", ["Каждый перевод", "Объединить пары"], key="edges_" + case["case_id"])
    labels = {e["transaction_id"]: e["source_account"][-6:] + " → " + e["target_account"][-6:] + " · "
              + e["amount_paid"] + " " + e["payment_currency"] + " · " + e["timestamp"][11:16] for e in case["events"]}
    focus = controls[3].selectbox("Выделить перевод в схеме", ["all"] + list(labels), key="focus_" + case["case_id"],
                                  format_func=lambda value: "Вся схема" if value == "all" else labels[value])

    rendered, counts = render_case_graph(case, expanded, context, edge_mode == "Каждый перевод",
                                         None if focus == "all" else focus)
    components.html(rendered, height=585, scrolling=False)
    st.caption(f'Показано {counts["nodes"]} счетов, {counts["shown"]} из {counts["total"]} переводов; '
               f'в основании {counts["evidence_shown"]} из {len(case["events"])}.')
    if case.get("context_truncated") or case["search_truncated"]:
        st.caption("Окружение или поиск ограничены бюджетом. Непоказанные связи не считаются отсутствующими.")
    if case["equal_timestamps"]:
        st.caption("Пунктир: в основании есть переводы одной минуты; их внутренний порядок неизвестен.")
    st.caption("Переводы основания: " + show_time(case["first_time"]) + " → " + show_time(case["last_time"]))

    operations, explanation, review = st.tabs(["Операции", "Объяснение", "Рассмотрение и экспорт"])

    with operations:
        frame = pd.DataFrame(case["events"])
        columns = ["transaction_id", "timestamp", "source_bank", "source_account", "target_bank", "target_account",
                   "amount_paid", "payment_currency", "amount_received", "receiving_currency", "payment_format"]
        names = {"transaction_id": "ID", "timestamp": "Время", "source_bank": "Банк отправителя",
                 "source_account": "Отправитель", "target_bank": "Банк получателя", "target_account": "Получатель",
                 "amount_paid": "Отправлено", "payment_currency": "Валюта", "amount_received": "Получено",
                 "receiving_currency": "Валюта получения", "payment_format": "Формат"}
        st.dataframe(frame[columns].rename(columns=names), hide_index=True, use_container_width=True)
        st.caption("Суммы сохранены как в источнике. Красный цвет означает основание для проверки, а не подтверждённое отмывание.")

    with explanation:
        compatible = models is not None and case["model_id"] == models["meta"]["model_id"]
        if models is None:
            st.info("Модели не загружены.")
        elif not compatible:
            st.warning("Случай сохранён другой версией модели. Для SHAP нужны те же артефакты.")
        elif not case["features"]:
            st.info("У операций основания нет сохранённых векторов признаков.")
        else:
            identifiers = list(case["features"])
            chosen = st.selectbox("Операция для SHAP", identifiers, key="shap_" + case["case_id"])
            if st.button("Рассчитать вклады признаков", key="explain_" + case["case_id"]):
                result = explain_transaction(models["transaction"], case["features"][chosen], models["meta"]["features"])
                st.caption(f'Базовое значение {result["base"]:.4f} + сумма вкладов = {result["raw"]:.4f}; после sigmoid: {result["score"]:.4f}.')
                st.bar_chart(result["table"].head(12).set_index("Признак"), color="#f35c69", horizontal=True)
                st.dataframe(result["table"], hide_index=True, use_container_width=True)
                st.caption("Вклады измеряются в raw score. Это объяснение CatBoost по сохранённому вектору.")

        if case["structure_features"] and models is not None and compatible:
            st.markdown("**Признаки структуры относительно train**")
            meta = models["meta"]
            table = pd.DataFrame([
                {"Признак": name, "В случае": value, "Медиана train": meta["structure_medians"].get(name),
                 "95-й процентиль train": meta["structure_q95"].get(name)}
                for name, value in case["structure_features"].items()
            ])
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("Необычность не является вероятностью отмывания. Эти значения описывают вход Isolation Forest.")

    with review:
        with st.form("review_" + case["case_id"]):
            status = st.selectbox("Статус", STATUSES, index=STATUSES.index(case["status"]))
            comment = st.text_area("Комментарий")
            analyst = st.text_input("Специалист", value=st.session_state.get("analyst", ""))
            if st.form_submit_button("Сохранить решение"):
                st.session_state["analyst"] = analyst
                with db.connect() as conn:
                    db.add_review(conn, case["case_id"], status, comment, analyst)
                st.rerun()

        if reviews:
            st.markdown("**Журнал решений**")
            st.dataframe(pd.DataFrame(reviews).rename(columns={"reviewed_at": "Время", "status": "Статус",
                                                              "comment": "Комментарий", "analyst": "Специалист"}),
                         hide_index=True, use_container_width=True)
        if versions:
            st.markdown("**Прежние версии случая**")
            st.dataframe(pd.DataFrame(versions).rename(columns={"version": "Версия", "detected_at": "Обнаружено",
                                                               "operation_score": "CatBoost", "anomaly_score": "Необычность"}),
                         hide_index=True, use_container_width=True)

        st.download_button("Скачать случай JSON", json.dumps(case, ensure_ascii=False, indent=2),
                           file_name="case_" + case["case_id"][:12] + ".json", mime="application/json")
        st.download_button("Скачать операции CSV", pd.DataFrame(case["events"]).to_csv(index=False).encode("utf-8-sig"),
                           file_name="operations.csv", mime="text/csv")
        st.caption("Модель: " + case["model_id"])


def choose_case(table, namespace):
    """Выбор случая: список, кнопки назад/далее и таблица очереди."""
    identifiers = [row["case_id"] for row in table]
    by_id = {row["case_id"]: row for row in table}
    key = namespace + "_case"
    if st.session_state.get(key) not in identifiers:
        st.session_state[key] = identifiers[0]

    def move(step):
        position = identifiers.index(st.session_state[key])
        st.session_state[key] = identifiers[max(0, min(len(identifiers) - 1, position + step))]

    def label(identifier):
        row = by_id[identifier]
        return (FAMILY_NAMES[row["family"]] + " · " + row["route"] + " · " + str(row["operations"]) + " переводов · "
                + row["detected_at"][5:16].replace("T", " ") + " · " + row["status"] + " · " + identifier[:6])

    current = identifiers.index(st.session_state[key])
    left, center, right = st.columns([1, 8, 1], vertical_alignment="bottom")
    left.button("← Назад", key=namespace + "_prev", disabled=current == 0, on_click=move, args=(-1,), use_container_width=True)
    identifier = center.selectbox("Открытый случай · можно искать по счёту или номеру", identifiers, key=key, format_func=label)
    right.button("Далее →", key=namespace + "_next", disabled=current == len(identifiers) - 1, on_click=move, args=(1,), use_container_width=True)
    st.caption("Карточка " + str(identifiers.index(identifier) + 1) + " из " + str(len(identifiers)))

    with st.expander("Список случаев", expanded=False):
        visible = pd.DataFrame(table)
        visible["family"] = visible.family.map(FAMILY_NAMES)
        columns = ["family", "route", "operations", "accounts", "operation_score", "anomaly_score", "status", "detected_at"]
        names = {"family": "Схема", "route": "Участники", "operations": "Переводы", "accounts": "Счета",
                 "operation_score": "CatBoost", "anomaly_score": "Необычность", "status": "Рассмотрение",
                 "detected_at": "Обнаружено"}
        st.dataframe(visible[columns].rename(columns=names), hide_index=True, use_container_width=True, height=300)
    return identifier


# ---------- вкладки ----------

workspace, network, research, help_tab = st.tabs(["Рабочее место", "Сеть", "Оценка модели", "Справка"])

with workspace:
    if total_cases == 0:
        st.info("Случаев пока нет. Запустите поток: python -m scripts.run_stream --reset "
                "или загрузите демо: python -m scripts.load_demo")
    else:
        lower, upper = date.fromisoformat(first_case[:10]), date.fromisoformat(last_case[:10])
        a, b, c = st.columns([2, 2, 3], vertical_alignment="bottom")
        start = a.date_input("Период с", lower, min_value=lower, max_value=upper, key="period_from")
        finish = b.date_input("Период по", upper, min_value=lower, max_value=upper, key="period_to")
        c.caption("Случай показывает связи на момент обнаружения. Рабочий граф хранит 30 дней; случай хранится всегда.")
        if finish < start:
            st.info("Начало периода должно быть не позже окончания.")
            st.stop()
        begin, end = day_bounds(start, finish)

        with db.connect() as conn:
            daily_transactions = db.transactions_per_day(conn, begin, end)
            daily_cases = db.cases_per_day(conn, begin, end)
            statuses = db.status_counts(conn, begin, end)

        counts = [
            sum(row["operations"] for row in daily_transactions),
            sum(row["operation_cases"] for row in daily_cases),
            sum(row["structure_cases"] for row in daily_cases),
            statuses.get("Новый", 0), statuses.get("Требует проверки", 0), statuses.get("Рассмотрен", 0),
        ]
        names = ["Операций в периоде", "Сигналов операций", "Структур", "Новых", "На проверке", "Рассмотрено"]
        for column, name, count in zip(st.columns(6), names, counts):
            column.metric(name, f"{count:,}".replace(",", " "))

        with st.expander("Динамика по дням"):
            left, right = st.columns(2)
            if daily_transactions:
                left.markdown("**Все операции по дням**")
                left.bar_chart(pd.DataFrame(daily_transactions).set_index("day").rename(columns={"operations": "Операции"}),
                               color="#7197d5", height=180)
            if daily_cases:
                right.markdown("**Случаи по дням**")
                right.bar_chart(pd.DataFrame(daily_cases).set_index("day").rename(
                    columns={"operation_cases": "Сигналы операций", "structure_cases": "Структуры"}), height=180)
            st.caption("Операции берутся из рабочего окна базы (30 дней). Случаи хранятся без ограничения по времени.")

        a, b, c = st.columns([2, 3, 2])
        channel = a.radio("Очередь", ["Структуры", "Операции"], horizontal=True, key="queue_channel")
        queue = "structures" if channel == "Структуры" else "operations"
        family = b.selectbox("Форма связанных переводов", ["Все"] + STRUCTURE_FAMILIES,
                             format_func=lambda value: "Все формы" if value == "Все" else FAMILY_NAMES[value],
                             key="scheme_" + queue, disabled=queue == "operations")
        status = c.selectbox("Статус рассмотрения", ["Все"] + STATUSES, key="status_" + queue)

        with db.connect() as conn:
            table = db.list_cases(conn, begin, end, queue,
                                  family=None if family == "Все" or queue == "operations" else family,
                                  status=None if status == "Все" else status)
        if family != "Все" and queue == "structures":
            st.caption(SCHEME_TEXT[family])

        if not table:
            st.info("В этом периоде и фильтре нет случаев.")
        else:
            st.caption("Структуры рассматриваются отдельно от операций. Необычность не означает доказанное нарушение."
                       if queue == "structures" else "Операции, прошедшие рабочий порог CatBoost, по убыванию оценки.")
            identifier = choose_case(table, queue)
            with db.connect() as conn:
                case = db.get_case(conn, identifier)
                reviews = db.case_reviews(conn, identifier)
                versions = db.case_versions(conn, identifier)
            show_case(case, reviews, versions)

with network:
    st.markdown("### Карта сети: выберите счёт и раскройте его окружение")
    st.caption("Поиск идёт по операциям рабочего окна базы. На рисунке ограниченный участок: до 100 счетов и 200 переводов.")
    search = st.text_input("Поиск счёта или банка", key="network_search", placeholder="Номер счёта целиком или его часть")
    if search:
        with db.connect() as conn:
            options = db.search_accounts(conn, search)
        if not options:
            st.info("Счёт не найден. Попробуйте другую часть номера.")
        else:
            labels = {row["bank"] + ":" + row["account"]: row["bank"] + " / " + row["account"] for row in options}
            seed = st.selectbox("Счёт для среза", list(labels), format_func=lambda value: labels[value], key="network_seed")
            if st.button("Показать срез сети", type="primary", key="network_build"):
                bank, account = seed.split(":", 1)
                with db.connect() as conn:
                    anchor = db.account_transactions(conn, bank, account, "0001-01-01T00:00:00", "9999-12-31T00:00:00")
                    keys = {e[side] for e in anchor for side in ["source_key", "target_key"]}
                    around = db.transactions_between(conn, keys, "0001-01-01T00:00:00", "9999-12-31T00:00:00") if keys else []
                    highlighted = db.signalled_transaction_ids(conn)
                rows = {e["transaction_id"]: e for e in anchor + around}
                st.session_state["network_view"] = {
                    "seed": seed,
                    "case": {"family": "network", "equal_timestamps": False,
                             "transaction_ids": [i for i in rows if i in highlighted],
                             "events": [e for i, e in rows.items() if i in highlighted],
                             "context": [e for i, e in rows.items() if i not in highlighted]},
                }
            view = st.session_state.get("network_view")
            if view and view["seed"] == seed:
                case = view["case"]
                if not case["events"] and not case["context"]:
                    st.info("У этого счёта нет операций в рабочем окне.")
                else:
                    rendered, info = case_graph(case, expanded=True)
                    components.html(rendered, height=585, scrolling=False)
                    st.caption(f'В срезе {info["nodes"]} счетов и {info["shown"]} переводов. '
                               'Красное: операции сохранённых случаев; серое: остальные связи.')

with research:
    metrics_path = ROOT / "reports" / "metrics.json"
    if not metrics_path.exists():
        st.info("Отчёт появляется после python -m scripts.evaluate.")
    else:
        report = read_json(metrics_path)
        selected = report["transaction_models"][report["selected"]]
        st.markdown("## Проверка на отложенном по времени test")
        st.caption("Основная модель: " + report["selected"] + ". Выбор по validation: находки среди первых 100, при равенстве AP.")
        st.caption("Результат отдельного эксперимента на размеченной истории. Это не измерение качества текущего входящего потока. Подробные временные срезы сохранены в docs/results.md.")
        st.markdown("**Весь test**")
        a, b, c, d = st.columns(4)
        a.metric("Найдено в первых 100", selected["top_k"]["100"]["found"])
        b.metric("Найдено в первых 500", selected["top_k"]["500"]["found"])
        c.metric("AP", f'{selected["ap"]:.4f}')
        d.metric("ROC AUC", f'{selected["roc_auc"]:.4f}')

        rows = [{"Модель": name, "AP": result["ap"], "ROC AUC": result["roc_auc"],
                 "Найдено в первых 100": result["top_k"]["100"]["found"],
                 "Найдено в первых 500": result["top_k"]["500"]["found"], "Сигналов": result["alerts"]}
                for name, result in report["transaction_models"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("A: поля операции и история счетов. B: A + графовые признаки. plain: без весов; balanced: веса классов.")

        coverage = report["structure"].get("coverage")
        if coverage:
            st.markdown("**Где находятся положительные операции test**")
            a, b, c, d = st.columns(4)
            a.metric("Всего положительных", coverage["test_positive_operations"])
            b.metric("Попали в структуры", coverage["in_any_candidate"])
            c.metric("Нашёл первый уровень", coverage["in_operation_alerts"])
            d.metric("Добавил второй", coverage["additional_from_structures"])
        st.caption("Метки доступны только оценке. Они не управляют рабочей очередью.")

with help_tab:
    st.markdown("""
## Как рассмотреть случай
1. Выберите период, очередь операций или структур, затем карточку.
2. Прочитайте основание. Две оценки относятся к разным объектам и не складываются.
3. Проверьте направление связей, время и суммы в таблице операций.
4. Откройте SHAP выбранной операции или признаки структуры.
5. Сохраните статус и комментарий. Каждое решение попадает в журнал.

**Счёт** — вершина графа с составным ключом банка и номера. **Перевод** — событие.
Повторные переводы одной пары остаются отдельными операциями, хотя на рисунке могут объединяться.
Красный цвет и необычность служат основанием дальнейшего анализа. Структура может быть законной.

Поступлением данных и настройкой моделей управляет администратор. Специалист работает с сохранённой очередью и журналом решений.
""")
