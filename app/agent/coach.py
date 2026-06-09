"""AdaptiveCoach Agent — Qwen-powered coaching loop with 3-layer memory."""

from __future__ import annotations

import json
from typing import Iterator

from openai import OpenAI

from app.config import get_settings
from app.memory.manager import MemoryManager

PATTERN_EXTRACTION_PROMPT = """Analyse the coaching session below and extract behavioural patterns about the user.
Return ONLY a JSON array. Each element must have:
  "category": one of ["productivity", "habit", "blocker", "emotion", "goal"]
  "pattern":  a concise, third-person statement (max 15 words)
  "confidence": float 0–1

Session summary:
{summary}

JSON array:"""


class AdaptiveCoachAgent:
    """
    Single-user coaching agent.

    Usage:
        agent = AdaptiveCoachAgent(user_id="<uuid>")
        for chunk in agent.chat("I keep procrastinating on deep work"):
            print(chunk, end="", flush=True)
        agent.end_session()
    """

    def __init__(self, user_id: str, memory: MemoryManager | None = None) -> None:
        self.user_id = user_id
        self.memory = memory or MemoryManager()
        cfg = get_settings()
        self._model = cfg.qwen_model
        self._client = OpenAI(
            api_key=cfg.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self._session_id: str | None = None

    # ── Session ────────────────────────────────────────────────────────────────

    def start_session(self) -> str:
        self._session_id = self.memory.start_session(self.user_id)
        return self._session_id

    def end_session(self, mood_score: int | None = None) -> str:
        """Summarise session, push to ChromaDB, extract and promote patterns."""
        if not self._session_id:
            raise RuntimeError("No active session.")

        summary = self._summarise_session()
        doc_id = self.memory.end_session(summary, mood_score)
        patterns = self._extract_patterns(summary)
        if patterns:
            self.memory.promote_patterns(patterns, source_doc_id=doc_id)
        self._session_id = None
        return summary

    # ── Chat ───────────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> Iterator[str]:
        """Stream a coaching response to the user's message."""
        if not self._session_id:
            self.start_session()

        self.memory.add_turn("user", user_message)
        messages = self.memory.build_context(user_message)

        stream = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024,
        )

        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            yield delta

        self.memory.add_turn("assistant", full_response)

    def chat_sync(self, user_message: str) -> str:
        """Non-streaming version — returns the full response string."""
        return "".join(self.chat(user_message))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _summarise_session(self) -> str:
        """Ask the LLM to summarise the current session in 3–5 sentences."""
        messages = self.memory.build_context("summarise this session")
        messages.append({
            "role": "user",
            "content": (
                "Summarise this coaching session in 3–5 sentences. "
                "Focus on topics discussed, progress made, and any commitments."
            ),
        })
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()

    def _extract_patterns(self, summary: str) -> list[dict]:
        """Extract behavioural patterns from the session summary via LLM."""
        prompt = PATTERN_EXTRACTION_PROMPT.format(summary=summary)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        try:
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except (json.JSONDecodeError, IndexError):
            return []
