from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name: Mapped[str] = Column(Text, nullable=False)
    timezone: Mapped[str] = Column(Text, nullable=False, default="UTC")
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    sessions: Mapped[list[Session]] = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    goals: Mapped[list[Goal]] = relationship("Goal", back_populates="user", cascade="all, delete-orphan")
    patterns: Mapped[list[UserPattern]] = relationship("UserPattern", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id: Mapped[str] = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    started_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[Optional[datetime]] = Column(DateTime(timezone=True))
    summary: Mapped[Optional[str]] = Column(Text)
    mood_score: Mapped[Optional[int]] = Column(SmallInteger, CheckConstraint("mood_score BETWEEN 1 AND 10"))

    user: Mapped[User] = relationship("User", back_populates="sessions")
    messages: Mapped[list[Message]] = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("idx_messages_session", "session_id", "created_at"),
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="chk_message_role"),
    )

    id: Mapped[str] = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    session_id: Mapped[str] = Column(UUID(as_uuid=False), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = Column(Text, nullable=False)
    content: Mapped[str] = Column(Text, nullable=False)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[Session] = relationship("Session", back_populates="messages")


class Goal(Base):
    __tablename__ = "goals"
    __table_args__ = (
        Index("idx_goals_user", "user_id", "status"),
        CheckConstraint("status IN ('active', 'completed', 'abandoned')", name="chk_goal_status"),
    )

    id: Mapped[str] = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id: Mapped[str] = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = Column(Text, nullable=False)
    description: Mapped[Optional[str]] = Column(Text)
    status: Mapped[str] = Column(Text, nullable=False, default="active")
    due_date: Mapped[Optional[datetime]] = Column(Date)
    created_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped[User] = relationship("User", back_populates="goals")


class UserPattern(Base):
    __tablename__ = "user_patterns"
    __table_args__ = (
        Index("idx_patterns_user", "user_id", "category"),
        UniqueConstraint("user_id", "category", "pattern", name="uq_pattern"),
    )

    id: Mapped[str] = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id: Mapped[str] = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    category: Mapped[str] = Column(Text, nullable=False)
    pattern: Mapped[str] = Column(Text, nullable=False)
    confidence: Mapped[float] = Column(Float, nullable=False, default=0.5)
    evidence_count: Mapped[int] = Column(Integer, nullable=False, default=1)
    first_seen: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())
    last_seen: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped[User] = relationship("User", back_populates="patterns")


class MemoryPromotion(Base):
    __tablename__ = "memory_promotions"

    id: Mapped[str] = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    user_id: Mapped[str] = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    chroma_doc_id: Mapped[str] = Column(Text, nullable=False)
    promoted_at: Mapped[datetime] = Column(DateTime(timezone=True), server_default=func.now())
    target_table: Mapped[str] = Column(Text, nullable=False)
    target_id: Mapped[str] = Column(UUID(as_uuid=False), nullable=False)
