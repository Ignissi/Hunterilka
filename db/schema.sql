-- Hunterилка: рабочие данные системы.
-- Аналитические таблицы (признаки, обучающие выборки) живут в Parquet, не здесь.

-- Поток операций за последние 30 дней. Старые строки удаляются сервисом.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      BIGINT PRIMARY KEY,
    ts                  TIMESTAMP NOT NULL,
    source_bank         TEXT NOT NULL,
    source_account      TEXT NOT NULL,
    target_bank         TEXT NOT NULL,
    target_account      TEXT NOT NULL,
    amount_paid         NUMERIC NOT NULL,
    payment_currency    TEXT NOT NULL,
    amount_received     NUMERIC NOT NULL,
    receiving_currency  TEXT NOT NULL,
    payment_format      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS transactions_ts ON transactions (ts);
CREATE INDEX IF NOT EXISTS transactions_source ON transactions (source_bank, source_account);
CREATE INDEX IF NOT EXISTS transactions_target ON transactions (target_bank, target_account);

-- Оценка и точный вектор нужны для восстановления уже оценённой истории.
-- У операций прогрева эти поля NULL. Меток is_laundering здесь нет.
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS operation_score DOUBLE PRECISION;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS feature_vector JSONB;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS model_id TEXT;

-- Один источник и один сервис записи на базу. Курсор фиксируется вместе с результатами.
CREATE TABLE IF NOT EXISTS stream_state (
    singleton          INTEGER PRIMARY KEY CHECK (singleton = 1),
    kind               TEXT NOT NULL CONSTRAINT stream_state_kind_v22_check CHECK (kind IN ('replay', 'demo', 'inbox')),
    dataset_id         TEXT NOT NULL,
    model_id           TEXT NOT NULL,
    feature_contract   JSONB NOT NULL,
    score_from         TIMESTAMP NOT NULL,
    last_time          TIMESTAMP,
    history_complete   BOOLEAN NOT NULL DEFAULT FALSE
);

-- Совместимое обновление схемы 2.1: данные и курсор сохраняются.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'stream_state'::regclass
               AND conname = 'stream_state_kind_check') THEN
        LOCK TABLE stream_state IN ACCESS EXCLUSIVE MODE;
        IF EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid = 'stream_state'::regclass
                   AND conname = 'stream_state_kind_check') THEN
            ALTER TABLE stream_state DROP CONSTRAINT stream_state_kind_check;
            ALTER TABLE stream_state ADD CONSTRAINT stream_state_kind_v22_check
                CHECK (kind IN ('replay', 'demo', 'inbox'));
        END IF;
    END IF;
END $$;

-- Случай: сигнал системы с замороженным снимком подграфа на момент обнаружения.
CREATE TABLE IF NOT EXISTS cases (
    case_id             TEXT PRIMARY KEY,
    candidate_id        TEXT NOT NULL,
    family              TEXT NOT NULL,
    detected_at         TIMESTAMP NOT NULL,
    first_detected_at   TIMESTAMP NOT NULL,
    first_time          TIMESTAMP NOT NULL,
    last_time           TIMESTAMP NOT NULL,
    operation_score     DOUBLE PRECISION,
    anomaly_score       DOUBLE PRECISION,
    operation_signal    BOOLEAN NOT NULL,
    structure_signal    BOOLEAN NOT NULL,
    operations          INTEGER NOT NULL,
    accounts            INTEGER NOT NULL,
    route               TEXT NOT NULL,
    version             INTEGER NOT NULL DEFAULT 1,
    status              TEXT NOT NULL DEFAULT 'Новый',
    model_id            TEXT NOT NULL,
    details             JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS cases_detected_at ON cases (detected_at);
CREATE INDEX IF NOT EXISTS cases_family ON cases (family);

-- Прежние версии случая: схема выросла, а старое основание нужно сохранить.
CREATE TABLE IF NOT EXISTS case_versions (
    case_id             TEXT NOT NULL REFERENCES cases (case_id),
    version             INTEGER NOT NULL,
    detected_at         TIMESTAMP NOT NULL,
    operation_score     DOUBLE PRECISION,
    anomaly_score       DOUBLE PRECISION,
    details             JSONB NOT NULL,
    PRIMARY KEY (case_id, version)
);

-- Журнал решений специалиста. Строки только добавляются, текущий статус хранится в cases.
CREATE TABLE IF NOT EXISTS reviews (
    review_id           BIGSERIAL PRIMARY KEY,
    case_id             TEXT NOT NULL REFERENCES cases (case_id),
    reviewed_at         TIMESTAMP NOT NULL DEFAULT now(),
    status              TEXT NOT NULL,
    comment             TEXT NOT NULL DEFAULT '',
    analyst             TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS reviews_case ON reviews (case_id, reviewed_at);
