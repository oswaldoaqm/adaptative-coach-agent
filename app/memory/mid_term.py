"""Mid-term memory: ChromaDB semantic store of session summaries."""

from __future__ import annotations

import os
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings


class MidTermMemory:
    """Stores and retrieves session summaries as embeddings in ChromaDB."""

    COLLECTION = "session_summaries"

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        host = host or os.getenv("CHROMA_HOST", "localhost")
        port = port or int(os.getenv("CHROMA_PORT", "8001"))
        self._client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Store ──────────────────────────────────────────────────────────────────

    def store(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upsert a session summary. Returns the document id."""
        doc_id = f"{user_id}:{session_id}"
        meta = {"user_id": user_id, "session_id": session_id, **(metadata or {})}
        self._col.upsert(ids=[doc_id], documents=[summary], metadatas=[meta])
        return doc_id

    # ── Retrieve ───────────────────────────────────────────────────────────────

    def query(
        self,
        user_id: str,
        text: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k most similar session summaries for a user."""
        results = self._col.query(
            query_texts=[text],
            n_results=n_results,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"summary": doc, "metadata": meta, "distance": dist})
        return out

    def get_recent(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch the most recent summaries for a user (by get, no embedding needed)."""
        result = self._col.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"],
            limit=limit,
        )
        return [
            {"id": id_, "summary": doc, "metadata": meta}
            for id_, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def delete(self, doc_id: str) -> None:
        self._col.delete(ids=[doc_id])
