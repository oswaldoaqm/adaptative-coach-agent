-- AdaptiveCoach Agent — PostgreSQL schema
-- Long-term memory layer

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ─── Users ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    timezone    TEXT NOT NULL DEFAULT 'UTC',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Sessions ─────────────────────────────────────────────────────────────────
-- Each coaching conversation is one session.
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    summary     TEXT,                   -- LLM-generated summary after session ends
    mood_score  SMALLINT CHECK (mood_score BETWEEN 1 AND 10)
);

-- ─── Messages ─────────────────────────────────────────────────────────────────
-- Full turn-by-turn log; source of truth for short-term memory reconstruction.
CREATE TABLE IF NOT EXISTS messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);

-- ─── Goals ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS goals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    description TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'completed', 'abandoned')),
    due_date    DATE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_goals_user ON goals(user_id, status);

-- ─── User Patterns ────────────────────────────────────────────────────────────
-- LLM-extracted behavioral patterns that persist across sessions.
-- Examples: "user is most productive in the morning",
--           "user avoids deep work on Fridays"
CREATE TABLE IF NOT EXISTS user_patterns (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category    TEXT NOT NULL,          -- e.g. 'productivity', 'habit', 'blocker'
    pattern     TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 0.5 CHECK (confidence BETWEEN 0 AND 1),
    evidence_count INT NOT NULL DEFAULT 1,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, category, pattern)
);

CREATE INDEX IF NOT EXISTS idx_patterns_user ON user_patterns(user_id, category);

-- ─── Memory Promotions ────────────────────────────────────────────────────────
-- Audit trail: when a ChromaDB (mid-term) memory was promoted to PostgreSQL.
CREATE TABLE IF NOT EXISTS memory_promotions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chroma_doc_id   TEXT NOT NULL,
    promoted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    target_table    TEXT NOT NULL,      -- 'user_patterns' | 'goals'
    target_id       UUID NOT NULL
);

-- ─── Helpers ──────────────────────────────────────────────────────────────────
-- Auto-update updated_at columns
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

CREATE OR REPLACE TRIGGER trg_goals_updated_at
    BEFORE UPDATE ON goals
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
