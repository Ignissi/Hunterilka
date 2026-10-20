"""Рабочее место. Математика и изменение графа находятся в src."""

import json
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.cases import CaseService
from src.explain import explain_transaction
from src.models import load_models
from src.patterns import FAMILY_NAMES
from src.replay import open_replay
from src.storage import ROOT, load_config, read_json
from src.visualization import case_graph
from src.workspace import SCHEME_TEXT, open_saved_cases, render_workspace


st.set_page_config(page_title="Hunter | Графовый AML", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container {padding-top:1.1rem; max-width:1600px;}
[data-testid="stSidebar"] {min-width:250px;max-width:250px;}
h1 {font-size:2rem !important; letter-spacing:-.05em; font-weight:650 !important;}
h2 {font-size:1.3rem !important; letter-spacing:-.025em;}
h3 {font-size:1.05rem !important;}
div[data-testid="stMetric"] {border-top:1px solid #30415b; padding:14px 0 5px;}
div[data-testid="stMetricValue"] {font-size:1.6rem;}
div[data-testid="stDataFrame"] {border:1px solid #24334a; border-radius:8px;}
.eyebrow {font:12px monospace;letter-spacing:.17em;color:#f35c69;margin:0 0 10px;}
.subline {color:#9aaec9;font-size:15px;margin-bottom:24px;}
.tag {border:1px solid #3b4b62;padding:4px 9px;border-radius:4px;font:11px monospace;color:#b2c1d7;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">HUNTER / GRAPH INTELLIGENCE / V1.1</div>', unsafe_allow_html=True)
st.title("Hunterилка · граф финансовых операций")
st.markdown('<div class="subline">История сети → связанные схемы → объяснимый случай → решение специалиста.</div>', unsafe_allow_html=True)

config = load_config()
if not (ROOT / "artifacts/model_meta.json").exists():
    st.info("Модели ещё не подготовлены. Выполните команды из RUN_GUIDE.md: подготовка, признаки, обучение, оценка.")
    st.code("python -m scripts.run_pipeline --device GPU\npython -m streamlit run app.py")
    st.stop()


@st.cache_resource
def cached_models(config_text):
    return load_models(json.loads(config_text))


try:
    models = cached_models(json.dumps(config, sort_keys=True))
except (ValueError, OSError, KeyError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.markdown("## Режим работы")
    mode = st.radio("Источник экрана", ["Сохранённые случаи", "Воспроизведение"])
    st.caption("IBM HI-Small · синтетический исследовательский набор · объединённая видимость банков")
    st.divider()
    st.markdown("**Два уровня анализа**")
    st.caption("CatBoost оценивает новую операцию по прошлому контексту. Isolation Forest оценивает связанную структуру.")
    st.markdown("**Три обозначения**")
    st.caption("Красный: связи выбранного случая. Серый: соседние переводы. Янтарный: выделенная операция. Нажмите ребро, чтобы увидеть участников, сумму и время.")
    st.divider()
    st.caption("Максимальная история: " + str(config["window_days"]) + " дней. Смена окна требует нового обучения.")


@st.cache_data(show_spinner=False, max_entries=48)
def render_case_graph(case, settings, expanded, context, individual, selected):
    return case_graph(case, settings, expanded, context, individual, selected)


def show_case(case):
    family = FAMILY_NAMES.get(case["family"], case["family"])
    participants = {e[side] for e in case["events"] for side in ["source_key", "target_key"]}
    st.markdown("### " + family + " · " + str(len(participants)) + " счетов · " + str(len(case["events"])) + " переводов")
    st.caption(SCHEME_TEXT.get(case["family"], "Связанный участок сети.") + " Обнаружено " + case["detected_at"].replace("T", " ") + " · случай " + case["case_id"][:12])
    if case.get("is_example"):
        st.info("Пример найденной формы из test. Он выбран для изучения механизма, без отбора по метке. Срабатывания моделей указаны отдельно.")
    a, b, c = st.columns([2, 2, 3])
    a.metric("CatBoost · максимальная оценка операции", "н/д" if case["operation_score"] is None else f'{case["operation_score"]:.4f}')
    b.metric("Isolation Forest · необычность структуры", "н/д" if case["anomaly_score"] is None else f'{case["anomaly_score"]:.4f}')
    grounds = []
    if case["operation_signal"]:
        grounds.append("сигнал CatBoost")
    if case["structure_signal"]:
        grounds.append("сигнал Isolation Forest")
    c.markdown("**Основание:** " + (" + ".join(grounds) if grounds else "пороги моделей не пройдены"))
    if case.get("review_rank") is not None:
        c.caption("Место " + str(case["review_rank"]) + " в очереди операций test.")
    c.caption("Статус: " + case["status"])
    controls = st.columns([2, 2, 3, 5])
    context = controls[0].toggle("Серое окружение", value=True, key="context_" + case["case_id"], disabled=not bool(case.get("context")))
    available = case["events"] + case.get("context", [])
    can_expand = len({e[side] for e in available for side in ["source_key", "target_key"]}) > config["display_nodes"] or len(available) > config["display_edges"]
    expanded = controls[1].toggle("Расширенный срез", value=len(participants) > config["display_nodes"], key="expanded_" + case["case_id"], disabled=not can_expand)
    edge_mode = controls[2].selectbox("Рёбра графа", ["Каждый перевод", "Объединить пары"], key="edges_" + case["case_id"])
    event_labels = {e["transaction_id"]: e["source_account"][-6:] + " → " + e["target_account"][-6:] + " · " + e["amount_paid"] + " " + e["payment_currency"] + " · " + e["timestamp"][11:16] for e in case["events"]}
    focus = controls[3].selectbox("Выделить перевод в схеме", ["all"] + list(event_labels), key="focus_" + case["case_id"], format_func=lambda value: "Вся схема" if value == "all" else event_labels[value])
    rendered, counts = render_case_graph(case, config, expanded, context, edge_mode == "Каждый перевод", None if focus == "all" else focus)
    components.html(rendered, height=585, scrolling=False)
    st.caption(f'Красное: связи выбранного случая · серое: окружение · янтарное: выделенный перевод. Показано {counts["nodes"]} счетов, {counts["shown"]} из {counts["total"]} переводов; в основании {counts["evidence_shown"]} из {len(case["events"])}.')
    if not case.get("context"):
        st.caption("В этой карточке нет сохранённых соседних переводов. Расширение не создаёт новых связей.")
    if counts["shown"] < counts["total"]:
        st.caption("Часть связей скрыта лимитом экрана. Расширенный срез показывает до 100 счетов и 200 рёбер; полный список основания доступен ниже.")
    if case.get("context_truncated") or case["search_truncated"]:
        st.caption("Окружение или поиск ограничены бюджетом. Непоказанные связи не считаются отсутствующими.")
    if case["equal_timestamps"]:
        st.caption("Пунктир: в основании есть переводы одной минуты; их внутренний порядок неизвестен.")
    st.caption("Переводы основания: " + case["first_time"].replace("T", " ") + " → " + case["last_time"].replace("T", " "))

    operations, explanation, review = st.tabs(["Операции", "Объяснение", "Рассмотрение и экспорт"])
    with operations:
        frame = pd.DataFrame(case["events"])
        columns = ["transaction_id", "timestamp", "source_bank", "source_account", "target_bank", "target_account", "amount_paid", "payment_currency", "amount_received", "receiving_currency", "payment_format"]
        st.dataframe(frame[columns].rename(columns={"transaction_id": "ID", "timestamp": "Время", "source_bank": "Банк отправителя", "source_account": "Отправитель", "target_bank": "Банк получателя", "target_account": "Получатель", "amount_paid": "Отправлено", "payment_currency": "Валюта отправления", "amount_received": "Получено", "receiving_currency": "Валюта получения", "payment_format": "Формат"}), hide_index=True, use_container_width=True)
        st.caption("Суммы сохранены как в источнике. Валюты не складываются. Красный цвет означает основание просмотра, а не подтверждённое отмывание.")

    with explanation:
        compatible = case["model_id"] == models["meta"]["model_id"]
        if not compatible:
            st.warning("Карточка сохранена с другой версией модели. Для её SHAP нужен соответствующий артефакт.")
        identifiers = list(case["features"]) if compatible else []
        if identifiers:
            chosen = st.selectbox("Операция для SHAP", identifiers, key="shap_" + case["case_id"])
            if st.button("Рассчитать вклады признаков", key="explain_" + case["case_id"]):
                result = explain_transaction(models["transaction"], case["features"][chosen], models["meta"]["features"])
                st.caption(f'Базовое значение {result["base"]:.4f} + сумма вкладов = {result["raw"]:.4f}; после sigmoid: {result["score"]:.4f}.')
                st.bar_chart(result["table"].head(12).set_index("Признак"), color="#f35c69", horizontal=True)
                st.dataframe(result["table"], hide_index=True, use_container_width=True)
                st.caption("Вклады измеряются в raw score. Это объяснение CatBoost по сохранённому вектору, не объяснение второго уровня.")
        else:
            st.info("У операций основания нет оценок этого test-запуска. Они наблюдались в периоде прогрева.")

        if case["structure_features"]:
            st.markdown("**Описание структуры относительно train**")
            meta = models["meta"]
            table = pd.DataFrame([
                {"Признак": name, "В случае": value, "Медиана train": meta["structure_medians"][name], "95-й процентиль train": meta["structure_q95"][name]}
                for name, value in case["structure_features"].items()
            ])
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("Необычность не является вероятностью отмывания. Эти значения описывают вход Isolation Forest.")

    with review:
        with st.form("review_" + case["case_id"]):
            statuses = ["Новый", "Требует проверки", "Рассмотрен"]
            status = st.selectbox("Статус", statuses, index=statuses.index(case["status"]))
            comment = st.text_area("Комментарий", value=case["comment"])
            if st.form_submit_button("Сохранить решение"):
                st.session_state.service.review(case["case_id"], status, comment)
                st.rerun()
        current = st.session_state.service.get(case["case_id"])
        st.download_button("Скачать случай JSON", json.dumps(current, ensure_ascii=False, indent=2), file_name="case_" + case["case_id"][:12] + ".json", mime="application/json")
        st.download_button("Скачать операции CSV", pd.DataFrame(case["events"]).to_csv(index=False).encode("utf-8-sig"), file_name="operations.csv", mime="text/csv")
        st.caption("Модель: " + case["model_id"] + " · признаки: " + case["feature_version"])


def show_queue(service):
    render_workspace(service, config, show_case)


workspace, research, help_tab = st.tabs(["Рабочее место", "Проверка моделей", "Справка"])
with workspace:
    if mode == "Сохранённые случаи":
        if not (ROOT / "demo/demo_manifest.json").exists():
            st.info("Для сохранённых случаев выполните python -m scripts.evaluate.")
        else:
            demo = read_json(ROOT / "demo/demo_manifest.json")
            if demo["model_id"] != models["meta"]["model_id"]:
                st.error("Карточки относятся к другой модели. Выполните scripts.evaluate после обучения.")
                st.stop()
            service = open_saved_cases(demo)
            if demo["data"].get("scope") != "full_source":
                st.warning("Техническая проверка на укороченном профиле. Полный эксперимент ещё не рассчитан.")
            render_workspace(service, config, show_case, demo)

    else:
        if not (ROOT / "data/processed/events.parquet").exists():
            st.info("Для воспроизведения подготовьте IBM CSV по RUN_GUIDE.md. Сохранённые случаи доступны сразу.")
        else:
            runs = sorted(p.parent for p in (ROOT / "runs").glob("*/checkpoint.json"))
            a, b, c = st.columns([1, 2, 1])
            if a.button("Новый запуск"):
                st.session_state.controller = open_replay(config, models)
                st.session_state.playing = False
            previous = b.selectbox("Сохранённый запуск", runs, format_func=lambda path: path.name) if runs else None
            if previous is not None and c.button("Восстановить"):
                with st.spinner("Восстановление повторным проходом до границы группы..."):
                    controller = open_replay(config, models, previous)
                    controller.restore()
                    st.session_state.controller = controller
                    st.session_state.playing = False

            @st.fragment(run_every=1.0 if st.session_state.get("playing", False) else None)
            def live_workspace():
                controller = st.session_state.get("controller")
                if controller is None:
                    st.info("Создайте новый запуск. Ранние периоды прогревают граф; сигналы создаются только на test.")
                    return
                a, b, c, d = st.columns(4)
                disabled = controller.status in ["Ошибка", "Завершён"]
                if a.button("Запуск", disabled=disabled):
                    st.session_state.playing = True
                    st.rerun()
                if b.button("Пауза"):
                    st.session_state.playing = False
                    st.rerun()
                single = c.button("Одна группа", disabled=disabled)
                packet = d.button("Пакет", disabled=disabled)
                try:
                    if single:
                        st.session_state.playing = False
                        controller.step()
                    elif packet or (st.session_state.get("playing", False) and not disabled):
                        controller.batch()
                except (ValueError, OSError) as error:
                    st.session_state.playing = False
                    st.error(str(error))
                state = "Выполняется" if st.session_state.get("playing", False) and controller.status != "Завершён" else controller.status
                st.caption(state + " · позиция " + str(controller.position) + " / " + str(len(controller.events)))
                st.progress(controller.position / len(controller.events))
                a, b, c = st.columns(3)
                a.metric("Активных операций", len(controller.state.events))
                b.metric("Связей", controller.state.graph.number_of_edges())
                c.metric("Случаев", len(controller.cases.index))
                st.caption("Время событий: " + str(controller.state.current_time or "ещё не начато"))
                if controller.last_diagnostics:
                    st.caption("Просмотрено записей поиска: " + str(controller.last_diagnostics["scanned"]) + "; пропущено опорных операций: " + str(controller.last_diagnostics["skipped_seeds"]))
                show_queue(controller.cases)

            live_workspace()

with research:
    metrics_path = ROOT / "reports/metrics.json"
    if metrics_path.exists():
        report = read_json(metrics_path)
        if report["model_id"] != models["meta"]["model_id"]:
            st.error("Отчёт относится к другой модели. Выполните scripts.evaluate.")
            st.stop()
        st.markdown("## Временная проверка A/B")
        st.caption("Выбор по validation: находки среди первых 100, при равенстве AP. Модель: " + report["selected"] + ". Ниже итоговый test.")
        selected = report["transaction_models"][report["selected"]]
        a, b, c, d = st.columns(4)
        a.metric("Найдено в первых 100", str(selected["top_k"]["100"]["found"]))
        b.metric("Найдено в первых 500", str(selected["top_k"]["500"]["found"]))
        c.metric("AP основной модели", f'{selected["ap"]:.4f}')
        d.metric("ROC AUC основной модели", f'{selected["roc_auc"]:.4f}')
        st.caption(f'Случайный выбор 500 операций дал бы в среднем {500 * selected["prevalence"]:.2f} положительных. Первые 500 по модели охватывают {selected["top_k"]["500"]["recall"]:.1%} всех положительных test.')
        st.caption("Ниже precision и recall рассчитаны при рабочих порогах, выбранных на validation. Это другой режим отбора, чем просмотр первых 500.")
        rows = [{"Модель": name, "AP": result["ap"], "Precision": result["precision"], "Recall": result["recall"], "F1": result["f1"], "Сигналов": result["alerts"]} for name, result in report["transaction_models"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("A: поля операции и история счетов. B: A + связи и графовые признаки. plain: без весов; balanced: веса по train.")
        coverage = report.get("search_coverage")
        if coverage:
            st.markdown("**Где находятся и где теряются положительные операции**")
            a, b, c, d = st.columns(4)
            a.metric("Всего в test", coverage["test_positive_operations"])
            b.metric("Захватил поиск", coverage["in_any_candidate"])
            c.metric("Нашёл первый уровень", coverage["in_operation_alerts"])
            d.metric("Добавил второй", coverage["additional_from_structures"])
            top = [{"Модель": name, "Найдено в первых 100": value["top_k"]["100"]["found"],
                    "Найдено в первых 500": value["top_k"]["500"]["found"]}
                   for name, value in report["transaction_models"].items()]
            st.dataframe(pd.DataFrame(top), hide_index=True, use_container_width=True)
        st.markdown("**Структуры: самостоятельная очередь исследования**")
        st.caption("Необычность не равна вероятности отмывания. Качество отбора структур показано отдельно от качества операций.")
        st.json(report["structure"], expanded=False)
        with st.expander("Почему отказались от смешивания очередей"):
            st.caption("Прежняя общая очередь вытесняла сильные сигналы операций структурными аномалиями. Результат этой проверки сохранён ниже. Теперь просмотр структур требует отдельного бюджета.")
            st.json(report.get("review_budgets", {"100": report["equal_budget"]}), expanded=False)
        st.caption("Метки доступны только оценке. Они не управляют рабочей очередью и подсветкой.")
    else:
        st.info("Отчёт появляется после python -m scripts.evaluate.")

with help_tab:
    st.markdown("""
## Как рассмотреть случай
1. Выберите очередь операций или структур, затем карточку.
2. Прочитайте основания. Две оценки относятся к разным объектам и не складываются.
3. Проверьте направление связей, время и суммы в таблице операций.
4. Откройте SHAP выбранной операции или признаки структуры.
5. Сохраните статус, комментарий и при необходимости экспорт.

**Сохранённые случаи** позволяют открыть готовый результат без загрузки всего IBM CSV.
В очереди операций сохранены первые 500 по оценке модели на test. Карточки ниже порога обозначены отдельно.
Структурные примеры рассматриваются независимо и не вытесняют операции.
**Воспроизведение** последовательно обрабатывает подготовленный профиль: шаг выполняет полную временную группу.
Перезапуск страницы не означает продолжения фоновой обработки. Кнопка восстановления пересчитывает прошлое до контрольной точки.

**Счёт** является вершиной с составным ключом банка и номера. **Перевод** является событием.
Повторные переводы одной пары остаются отдельными операциями, хотя на рисунке объединяются в связь.
Красный цвет и необычность служат основанием дальнейшего анализа. Структура может быть законной.

Запуск: `RUN_GUIDE.md`. Архитектура: `docs/architecture.md`. Эксперимент: `docs/results.md`.
""")
