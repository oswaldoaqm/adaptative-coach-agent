"""MemoryManager: orchestrates the 3-layer memory system."""

from __future__ import annotations

from typing import Any

from .short_term import ShortTermMemory
from .mid_term import MidTermMemory
from .long_term import LongTermMemory
from app.db.models import Session as DBSession, User


SYSTEM_PROMPT_TEMPLATE = """You are AdaptiveCoach, a productivity coach with deep memory of your user.

## User Profile
Name: {name}
Timezone: {timezone}

## Active Goals
{goals}

## Known Patterns
{patterns}

## Relevant Past Sessions
{past_sessions}

Use this context to give personalised, actionable coaching. Be concise and direct.
"""


class MemoryManager:
    """
    Coordinates all three memory layers for a single user session.

    Lifecycle:
        1. `start_session(user_id)` — open DB session, load long-term context.
        2. `add_turn(role, content)` — append to short-term + persist to DB.
        3. `build_context(query)` — assemble the LLM messages list.
        4. `end_session(summary, mood)` — close DB session, push summary to ChromaDB.
        5. `promote_patterns(patterns)` — move LLM-extracted patterns to PostgreSQL.
    """

    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        mid_term: MidTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ) -> None:
        self.st = short_term or ShortTermMemory()
        self.mt = mid_term or MidTermMemory()
        self.lt = long_term or LongTermMemory()

        self._user: User | None = None
        self._db_session: DBSession | None = None

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def start_session(self, user_id: str) -> str:
        """Open a new coaching session. Returns the session_id."""
        self._user = self._get_user(user_id)
        db_session = self.lt.open_session(user_id)
        self._db_session = db_session
        self.st.clear()
        return db_session.id

    def end_session(self, summary: str, mood_score: int | None = None) -> str:
        """Close session, persist summary to DB and ChromaDB. Returns chroma doc_id."""
        if not self._db_session or not self._user:
            raise RuntimeError("No active session. Call start_session first.")

        self.lt.close_session(self._db_session.id, summary, mood_score)

        doc_id = self.mt.store(
            user_id=self._user.id,
            session_id=self._db_session.id,
            summary=summary,
            metadata={"mood_score": mood_score} if mood_score else {},
        )

        self._db_session = None
        self.st.clear()
        return doc_id

    # ── Turn management ────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Add a message to short-term memory and persist it to PostgreSQL."""
        if not self._db_session:
            raise RuntimeError("No active session. Call start_session first.")
        self.st.add(role, content)  # type: ignore[arg-type]
        self.lt.save_message(self._db_session.id, role, content)

    # ── Context assembly ───────────────────────────────────────────────────────

    def build_context(self, query: str, n_past: int = 3) -> list[dict]:
        """
        Build the full messages list to send to the LLM:
        system prompt (long-term facts + semantically relevant past sessions)
        + short-term turns.
        """
        if not self._user:
            raise RuntimeError("No active session.")

        goals = self.lt.get_active_goals(self._user.id)
        patterns = self.lt.get_patterns(self._user.id)
        past = self.mt.query(self._user.id, query, n_results=n_past)

        system = SYSTEM_PROMPT_TEMPLATE.format(
            name=self._user.name,
            timezone=self._user.timezone,
            goals=self._fmt_goals(goals),
            patterns=self._fmt_patterns(patterns),
            past_sessions=self._fmt_past(past),
        )

        self.st.system_prompt = system
        return self.st.to_messages()

    # ── Pattern promotion ──────────────────────────────────────────────────────

    def promote_patterns(
        self,
        patterns: list[dict[str, Any]],
        source_doc_id: str | None = None,
    ) -> None:
        """
        Persist LLM-extracted patterns to PostgreSQL.

        Each pattern dict: {"category": str, "pattern": str, "confidence": float}
        """
        if not self._user:
            raise RuntimeError("No active session.")

        for p in patterns:
            row = self.lt.upsert_pattern(
                user_id=self._user.id,
                category=p["category"],
                pattern=p["pattern"],
                confidence=p.get("confidence", 0.5),
            )
            if source_doc_id:
                self.lt.record_promotion(
                    user_id=self._user.id,
                    chroma_doc_id=source_doc_id,
                    target_table="user_patterns",
                    target_id=row.id,
                )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_user(self, user_id: str) -> User:
        from sqlalchemy import select
        from sqlalchemy.orm import Session as ORM
        with ORM(self.lt._engine) as db:
            user = db.get(User, user_id)
            if not user:
                raise ValueError(f"User {user_id!r} not found. Create with lt.get_or_create_user().")
            return user

    @staticmethod
    def _fmt_goals(goals) -> str:
        if not goals:
            return "None set yet."
        return "\n".join(f"- [{g.status}] {g.title}" + (f" (due {g.due_date})" if g.due_date else "") for g in goals)

    @staticmethod
    def _fmt_patterns(patterns) -> str:
        if not patterns:
            return "None identified yet."
        return "\n".join(
            f"- [{p.category}] {p.pattern} (confidence {p.confidence:.0%}, seen {p.evidence_count}x)"
            for p in patterns
        )

    @staticmethod
    def _fmt_past(past) -> str:
        if not past:
            return "No relevant past sessions."
        return "\n\n".join(
            f"Session {i + 1} (similarity {1 - r['distance']:.0%}):\n{r['summary']}"
            for i, r in enumerate(past)
        )
