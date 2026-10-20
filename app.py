"""Рабочее место. Математика и изменение графа находятся в src."""

import json
import shutil
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.cases import CaseService
from src.explain import explain_transaction
from src.models import load_models
from src.patterns import FAMILY_NAMES
from src.replay import open_replay
from src.review_queue import ordered_cases
from src.storage import ROOT, fingerprint, load_config, read_json
from src.visualization import case_graph


st.set_page_config(page_title="Hunter | Графовый AML", page_icon="◈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:2rem; max-width:1500px;}
h1 {font-size:2.35rem !important; letter-spacing:-.05em; font-weight:650 !important;}
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
st.title("Большая сеть. Объяснимый случай.")
st.markdown('<div class="subline">Связанные переводы, два независимых сигнала и сохранённые основания рассмотрения.</div>', unsafe_allow_html=True)

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
    st.caption("Красный: операции сигнала. Янтарный: участники. Серый: дополнительный контекст. Подпись связи: число переводов.")
    st.divider()
    st.caption("Максимальная история: " + str(config["window_days"]) + " дней. Смена окна требует нового обучения.")


def show_case(case):
    family = FAMILY_NAMES.get(case["family"], case["family"])
    st.markdown("## " + family)
    st.caption("Случай " + case["case_id"][:12] + " · версия " + str(case["version"]) + " · обнаружен " + case["detected_at"].replace("T", " "))
    a, b, c = st.columns(3)
    a.metric("Оценка операции", "н/д" if case["operation_score"] is None else f'{case["operation_score"]:.4f}')
    b.metric("Необычность структуры", "н/д" if case["anomaly_score"] is None else f'{case["anomaly_score"]:.4f}')
    c.metric("Операций в основании", str(len(case["events"])))

    expanded = st.checkbox("Расширить граф до 100 счетов и 200 связей", key="expanded_" + case["case_id"])
    rendered, counts = case_graph(case, config, expanded)
    components.html(rendered, height=495, scrolling=False)
    st.caption(f'Показано {counts["shown"]} из {counts["total"]} операций снимка · {counts["nodes"]} счетов · {counts["pairs"]} направленных связей')
    grounds = []
    if case["operation_signal"]:
        grounds.append("оценка операции выше validation-порога")
    if case["structure_signal"]:
        grounds.append("необычность структуры выше validation-порога")
    if case.get("review_rank") is not None:
        grounds.append("место " + str(case["review_rank"]) + " среди операций test по оценке CatBoost")
    st.markdown("**Основания:** " + "; ".join(grounds) + ".")
    if not case["operation_signal"] and not case["structure_signal"]:
        st.caption("Операция выбрана по месту в очереди. Рабочий порог сигнала не пройден.")
    st.caption("Интервал операций: " + case["first_time"].replace("T", " ") + " → " + case["last_time"].replace("T", " "))
    if case["search_truncated"]:
        st.warning("Поиск ограничен бюджетом. Этот случай не означает, что все структуры вокруг него просмотрены.")
    if case["equal_timestamps"]:
        st.info("Есть совпадающие временные метки. Порядок внутри одной минуты неизвестен; связи обозначены пунктиром.")
    if case.get("context_truncated"):
        st.caption("Дополнительный вычислительный контекст получен с ограничениями.")

    operations, explanation, review = st.tabs(["Операции", "Объяснение", "Рассмотрение и экспорт"])
    with operations:
        frame = pd.DataFrame(case["events"])
        columns = ["transaction_id", "timestamp", "source_bank", "source_account", "target_bank", "target_account", "amount_paid", "payment_currency", "amount_received", "receiving_currency", "payment_format"]
        st.dataframe(frame[columns], hide_index=True, use_container_width=True)
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
                st.success("Решение сохранено.")
        current = st.session_state.service.get(case["case_id"])
        st.download_button("Скачать случай JSON", json.dumps(current, ensure_ascii=False, indent=2), file_name="case_" + case["case_id"][:12] + ".json", mime="application/json")
        st.download_button("Скачать операции CSV", pd.DataFrame(case["events"]).to_csv(index=False).encode("utf-8-sig"), file_name="operations.csv", mime="text/csv")
        st.caption("Модель: " + case["model_id"] + " · признаки: " + case["feature_version"])


def show_queue(service):
    st.session_state.service = service
    if not service.index:
        st.info("Очередь пока пуста. История накапливается; отсутствие случая не является ошибкой.")
        return
    left, right = st.columns([1, 2.8], gap="large")
    with left:
        st.markdown("## Очереди рассмотрения")
        channel = st.radio("Что рассматриваем", ["Операции", "Структуры"], horizontal=True)
        queue = "operations" if channel == "Операции" else "structures"
        table = ordered_cases(list(service.index.values()), queue)
        st.caption("Операции идут по оценке CatBoost. Структуры имеют отдельную очередь по необычности.")
        if table.empty:
            st.info("В этой очереди пока нет карточек. Можно выбрать другую.")
            return
        family = st.selectbox("Структура", ["Все"] + sorted(table.family.unique()), format_func=lambda value: FAMILY_NAMES.get(value, value))
        status = st.selectbox("Рассмотрение", ["Все", "Новый", "Требует проверки", "Рассмотрен"])
        dates = sorted({value[:10] for value in table.detected_at})
        day = st.selectbox("Дата обнаружения", ["Все"] + dates)
        if family != "Все":
            table = table.loc[table.family == family]
        if status != "Все":
            table = table.loc[table.status == status]
        if day != "Все":
            table = table.loc[table.detected_at.str.startswith(day)]
        if table.empty:
            st.info("Случаев по фильтрам нет.")
            return
        if len(table) > 5000:
            st.warning("Найдено " + str(len(table)) + " случаев. Сузьте дату, тип или статус.")
            return
        pages = max(1, (len(table) + 49) // 50)
        page = st.selectbox("Страница", range(1, pages + 1))
        visible = table.iloc[(page - 1) * 50:page * 50]
        st.caption(str(len(table)) + " случаев · на странице " + str(len(visible)))
        identifiers = visible.case_id.tolist()

        def label(identifier):
            row = service.index[identifier]
            return FAMILY_NAMES.get(row["family"], row["family"]) + " · " + str(row["operations"]) + " оп. · " + identifier[:6]

        previous_id = st.session_state.get("selected_case_id")
        initial = identifiers.index(previous_id) if previous_id in identifiers else 0
        identifier = st.selectbox("Открыть случай", identifiers, index=initial, format_func=label)
        st.session_state.selected_case_id = identifier
        score = "operation_score" if queue == "operations" else "anomaly_score"
        overview = visible[["family", "operations", score]].copy()
        overview.family = overview.family.map(FAMILY_NAMES)
        overview.columns = ["Тип", "Операций", "Оценка CatBoost" if queue == "operations" else "Необычность"]
        st.dataframe(overview, hide_index=True, use_container_width=True)
    with right:
        show_case(service.get(identifier))


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
            # Решения демонстратора хранятся отдельно от поставленных исходных карточек.
            destination = ROOT / "runs/demo_review" / demo["model_id"] / fingerprint(demo)[:12] / "cases"
            if not destination.exists():
                shutil.copytree(ROOT / "demo/cases", destination)
            service = CaseService(destination)
            st.caption("СОХРАНЁННЫЙ TEST · выборка карточек; поток сейчас не воспроизводится")
            if demo.get("review_policy"):
                st.caption("Операции: первые 500 по оценке модели, включая оценки ниже рабочего порога. Структуры: отдельная выборка примеров. Отбор без меток.")
            if demo["data"].get("scope") != "full_source":
                st.warning("Техническая проверка на укороченном профиле. Полный эксперимент ещё не рассчитан.")
            a, b, c, d = st.columns(4)
            a.metric("Обработано операций", f'{demo["data"]["rows"]:,}'.replace(",", " "))
            b.metric("Счетов профиля", f'{demo["data"]["accounts"]:,}'.replace(",", " "))
            c.metric("Карточек в демонстрации", str(demo["cases"]))
            d.metric("Окно истории", str(config["window_days"]) + " дней")
            st.caption("Доступная история: " + demo["data"]["first_time"] + " → " + demo["data"]["last_time"])
            show_queue(service)
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

            @st.fragment(run_every=1.0)
            def live_workspace():
                controller = st.session_state.get("controller")
                if controller is None:
                    st.info("Создайте новый запуск. Ранние периоды прогревают граф; сигналы создаются только на test.")
                    return
                a, b, c, d = st.columns(4)
                disabled = controller.status in ["Ошибка", "Завершён"]
                if a.button("Запуск", disabled=disabled):
                    st.session_state.playing = True
                if b.button("Пауза"):
                    st.session_state.playing = False
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
