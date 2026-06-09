"""Long-term memory: PostgreSQL — goals, patterns, session records."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session as DBSession

from app.db.models import Base, Goal, MemoryPromotion, Message, Session, User, UserPattern


def _engine(url: str | None = None):
    url = url or os.getenv("POSTGRES_URL", "postgresql://coach:coach@localhost:5432/adaptivecoach")
    return create_engine(url, pool_pre_ping=True)


class LongTermMemory:
    def __init__(self, db_url: str | None = None) -> None:
        self._engine = _engine(db_url)

    # ── Bootstrap ──────────────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Create all tables (idempotent — use for tests or first-run outside Docker)."""
        Base.metadata.create_all(self._engine)

    # ── Users ──────────────────────────────────────────────────────────────────

    def get_or_create_user(self, name: str, timezone: str = "UTC") -> User:
        with DBSession(self._engine) as db:
            user = db.execute(select(User).where(User.name == name)).scalar_one_or_none()
            if not user:
                user = User(name=name, timezone=timezone)
                db.add(user)
                db.commit()
                db.refresh(user)
            return user

    # ── Sessions ───────────────────────────────────────────────────────────────

    def open_session(self, user_id: str) -> Session:
        with DBSession(self._engine) as db:
            session = Session(user_id=user_id)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    def close_session(self, session_id: str, summary: str, mood_score: int | None = None) -> None:
        with DBSession(self._engine) as db:
            db.execute(
                update(Session)
                .where(Session.id == session_id)
                .values(ended_at=datetime.now(timezone.utc), summary=summary, mood_score=mood_score)
            )
            db.commit()

    # ── Messages ───────────────────────────────────────────────────────────────

    def save_message(self, session_id: str, role: str, content: str) -> Message:
        with DBSession(self._engine) as db:
            msg = Message(session_id=session_id, role=role, content=content)
            db.add(msg)
            db.commit()
            db.refresh(msg)
            return msg

    def get_session_messages(self, session_id: str) -> list[Message]:
        with DBSession(self._engine) as db:
            return list(
                db.execute(
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.created_at)
                ).scalars()
            )

    # ── Goals ──────────────────────────────────────────────────────────────────

    def upsert_goal(self, user_id: str, title: str, description: str = "", due_date=None) -> Goal:
        with DBSession(self._engine) as db:
            existing = db.execute(
                select(Goal).where(Goal.user_id == user_id, Goal.title == title, Goal.status == "active")
            ).scalar_one_or_none()
            if existing:
                return existing
            goal = Goal(user_id=user_id, title=title, description=description, due_date=due_date)
            db.add(goal)
            db.commit()
            db.refresh(goal)
            return goal

    def get_active_goals(self, user_id: str) -> list[Goal]:
        with DBSession(self._engine) as db:
            return list(
                db.execute(
                    select(Goal).where(Goal.user_id == user_id, Goal.status == "active")
                ).scalars()
            )

    def complete_goal(self, goal_id: str) -> None:
        with DBSession(self._engine) as db:
            db.execute(update(Goal).where(Goal.id == goal_id).values(status="completed"))
            db.commit()

    # ── Patterns ───────────────────────────────────────────────────────────────

    def upsert_pattern(
        self,
        user_id: str,
        category: str,
        pattern: str,
        confidence: float = 0.5,
    ) -> UserPattern:
        """Insert or strengthen an existing pattern."""
        with DBSession(self._engine) as db:
            stmt = (
                insert(UserPattern)
                .values(
                    user_id=user_id,
                    category=category,
                    pattern=pattern,
                    confidence=confidence,
                    evidence_count=1,
                )
                .on_conflict_do_update(
                    constraint="uq_pattern",
                    set_={
                        "confidence": (UserPattern.confidence + confidence) / 2,
                        "evidence_count": UserPattern.evidence_count + 1,
                        "last_seen": datetime.now(timezone.utc),
                    },
                )
                .returning(UserPattern)
            )
            result = db.execute(stmt).scalar_one()
            db.commit()
            return result

    def get_patterns(self, user_id: str, category: str | None = None) -> list[UserPattern]:
        with DBSession(self._engine) as db:
            q = select(UserPattern).where(UserPattern.user_id == user_id)
            if category:
                q = q.where(UserPattern.category == category)
            return list(db.execute(q.order_by(UserPattern.confidence.desc())).scalars())

    # ── Memory Promotion ───────────────────────────────────────────────────────

    def record_promotion(
        self, user_id: str, chroma_doc_id: str, target_table: str, target_id: str
    ) -> None:
        with DBSession(self._engine) as db:
            db.add(MemoryPromotion(
                user_id=user_id,
                chroma_doc_id=chroma_doc_id,
                target_table=target_table,
                target_id=target_id,
            ))
            db.commit()
