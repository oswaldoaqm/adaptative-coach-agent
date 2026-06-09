"""Short-term memory: in-context sliding window of messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Role = Literal["user", "assistant", "system"]


@dataclass
class Turn:
    role: Role
    content: str


class ShortTermMemory:
    """Keeps the last `max_turns` turns in memory for the LLM context window."""

    def __init__(self, max_turns: int = 20, system_prompt: str = "") -> None:
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self._turns: list[Turn] = []

    def add(self, role: Role, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        if len(self._turns) > self.max_turns:
            # Drop oldest non-system turns
            self._turns = self._turns[-self.max_turns :]

    def to_messages(self) -> list[dict]:
        """Return OpenAI-compatible messages list."""
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend({"role": t.role, "content": t.content} for t in self._turns)
        return messages

    def clear(self) -> list[Turn]:
        """Flush and return all turns (used when promoting to mid-term)."""
        turns, self._turns = self._turns, []
        return turns

    def __len__(self) -> int:
        return len(self._turns)
