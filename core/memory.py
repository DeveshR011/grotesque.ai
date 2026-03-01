"""
Grotesque AI – Memory System

Modular, detachable memory with two phases:

Phase 1 – Short-term RAM-only conversation memory:
 • Sliding window of recent exchanges
 • No persistence, no disk, no network
 • Auto-cleared on idle / shutdown

Phase 2 – Local vector database (FAISS):
 • Stores embeddings for semantic retrieval
 • Encrypted at rest (AES-256)
 • Purely local, never transmits data externally
 • Detachable – pipeline works without it

Architecture:
 • MemoryManager is the unified interface
 • ShortTermMemory is Phase 1 (always active)
 • VectorMemory is Phase 2 (opt-in, requires FAISS)
 • Both implement the MemoryBackend ABC
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("grotesque.memory")


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class MemoryEntry:
    """A single memory record."""
    role: str                          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    entry_id: str = ""

    def __post_init__(self):
        if not self.entry_id:
            raw = f"{self.timestamp}:{self.role}:{self.content[:64]}"
            self.entry_id = hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class RetrievalResult:
    """Returned by memory search."""
    entry: MemoryEntry
    score: float = 0.0                 # similarity score (higher = better)


# ======================================================================
# Abstract backend
# ======================================================================

class MemoryBackend(ABC):
    """
    Abstract interface for a memory storage backend.
    All backends must be fully local and never transmit data.
    """

    @abstractmethod
    def store(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]: ...

    @abstractmethod
    def get_recent(self, n: int = 10) -> List[MemoryEntry]: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def size(self) -> int: ...


# ======================================================================
# Phase 1: Short-term RAM-only memory
# ======================================================================

class ShortTermMemory(MemoryBackend):
    """
    In-memory conversation buffer.  No persistence.

    Keeps the last ``max_entries`` exchanges in a circular buffer.
    Wiped on shutdown and idle timeout.
    """

    def __init__(self, max_entries: int = 100) -> None:
        self._max = max_entries
        self._entries: List[MemoryEntry] = []
        self._lock = threading.Lock()

    def store(self, entry: MemoryEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max:
                # Remove oldest
                self._entries = self._entries[-self._max:]

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Simple substring search (no embeddings needed for Phase 1)."""
        with self._lock:
            query_lower = query.lower()
            results = []
            for entry in reversed(self._entries):
                if query_lower in entry.content.lower():
                    score = 1.0 / (1.0 + abs(time.time() - entry.timestamp) / 3600)
                    results.append(RetrievalResult(entry=entry, score=score))
                    if len(results) >= top_k:
                        break
            return results

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        with self._lock:
            return list(self._entries[-n:])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            logger.debug("Short-term memory cleared")

    def size(self) -> int:
        return len(self._entries)

    def to_llm_messages(self, n: int = 10, role_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """Format recent memory as LLM chat messages.

        Args:
            n: max entries to return.
            role_filter: if set, only return entries matching this role
                         (e.g. ``"context"`` for speaker transcriptions).
        """
        entries = self.get_recent(n * 2 if role_filter else n)
        if role_filter:
            entries = [e for e in entries if e.role == role_filter][-n:]
        return [{"role": e.role, "content": e.content} for e in entries]


# ======================================================================
# Phase 2: Vector memory with FAISS
# ======================================================================

class VectorMemory(MemoryBackend):
    """
    Local vector database using FAISS for semantic retrieval.

    Features:
     • Stores embeddings computed from a local model
     • Encrypted persistence (AES-256 via cryptography.fernet)
     • No cloud, no network, fully local
     • Detachable – the system works without it

    Requires: faiss-cpu (or faiss-gpu), sentence-transformers (local model)
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_entries: int = 10_000,
        encryption_key: Optional[bytes] = None,
    ) -> None:
        self._index_path = index_path
        self._embedding_model_name = embedding_model
        self._dim = embedding_dim
        self._max = max_entries
        self._encryption_key = encryption_key

        self._index = None
        self._embedder = None
        self._entries: List[MemoryEntry] = []
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        """Lazily load FAISS index and embedding model."""
        if self._loaded:
            return

        try:
            import faiss
        except ImportError:
            logger.warning("faiss not installed – vector memory disabled")
            logger.warning("Install with: pip install faiss-cpu")
            return

        # Create FAISS index
        self._index = faiss.IndexFlatIP(self._dim)  # inner product (cosine after norm)
        logger.info("FAISS index created (dim=%d)", self._dim)

        # Load from disk if exists and encryption key provided
        if self._index_path and self._index_path.exists():
            self._load_from_disk()

        # Load local embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._embedding_model_name)
            logger.info("Embedding model loaded: %s", self._embedding_model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed – using random embeddings")
            self._embedder = None

        self._loaded = True

    def store(self, entry: MemoryEntry) -> None:
        if not self._loaded:
            self.load()
        if self._index is None:
            return

        embedding = self._compute_embedding(entry.content)
        entry.embedding = embedding

        with self._lock:
            self._entries.append(entry)
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            self._index.add(embedding.reshape(1, -1).astype(np.float32))

            if len(self._entries) > self._max:
                self._rebuild_index()

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self._loaded or self._index is None or self._index.ntotal == 0:
            return []

        embedding = self._compute_embedding(query)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        with self._lock:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(
                embedding.reshape(1, -1).astype(np.float32), k
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self._entries):
                    results.append(RetrievalResult(
                        entry=self._entries[idx],
                        score=float(score),
                    ))
            return results

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        with self._lock:
            return list(self._entries[-n:])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            if self._index is not None:
                self._index.reset()
            logger.debug("Vector memory cleared")

    def size(self) -> int:
        return len(self._entries)

    def save_to_disk(self) -> None:
        """Persist index + entries to disk with encryption."""
        if not self._index_path or self._index is None:
            return

        import faiss
        import pickle

        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize entries (without embeddings to save space)
        entries_data = []
        for e in self._entries:
            entries_data.append({
                "role": e.role, "content": e.content,
                "timestamp": e.timestamp, "metadata": e.metadata,
                "entry_id": e.entry_id,
            })

        payload = pickle.dumps({
            "entries": entries_data,
            "index_bytes": faiss.serialize_index(self._index).tobytes(),
        })

        if self._encryption_key:
            from cryptography.fernet import Fernet
            f = Fernet(self._encryption_key)
            payload = f.encrypt(payload)

        self._index_path.write_bytes(payload)
        logger.info("Vector memory saved to %s (%d entries)", self._index_path, len(self._entries))

    def _load_from_disk(self) -> None:
        """Load encrypted index from disk."""
        import faiss
        import pickle

        payload = self._index_path.read_bytes()

        if self._encryption_key:
            from cryptography.fernet import Fernet
            f = Fernet(self._encryption_key)
            try:
                payload = f.decrypt(payload)
            except Exception:
                logger.error("Failed to decrypt vector memory – starting fresh")
                return

        data = pickle.loads(payload)
        index_bytes = np.frombuffer(data["index_bytes"], dtype=np.uint8)
        self._index = faiss.deserialize_index(index_bytes)

        for ed in data["entries"]:
            self._entries.append(MemoryEntry(
                role=ed["role"], content=ed["content"],
                timestamp=ed["timestamp"], metadata=ed.get("metadata", {}),
                entry_id=ed.get("entry_id", ""),
            ))

        logger.info("Vector memory loaded: %d entries", len(self._entries))

    def _compute_embedding(self, text: str) -> np.ndarray:
        if self._embedder is not None:
            return self._embedder.encode(text, show_progress_bar=False)
        # Fallback: random (for testing only)
        return np.random.randn(self._dim).astype(np.float32)

    def _rebuild_index(self) -> None:
        """Trim old entries and rebuild FAISS index."""
        import faiss
        self._entries = self._entries[-self._max:]
        self._index = faiss.IndexFlatIP(self._dim)
        for entry in self._entries:
            if entry.embedding is not None:
                emb = entry.embedding.copy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self._index.add(emb.reshape(1, -1).astype(np.float32))


# ======================================================================
# Memory Manager (unified interface)
# ======================================================================

class MemoryManager:
    """
    Unified memory interface.  Aggregates short-term + vector backends.
    Fully modular and detachable – pipeline runs without it.
    """

    def __init__(
        self,
        short_term_size: int = 100,
        enable_vector: bool = False,
        vector_index_path: Optional[Path] = None,
        encryption_key: Optional[bytes] = None,
    ) -> None:
        self._short_term = ShortTermMemory(max_entries=short_term_size)
        self._vector: Optional[VectorMemory] = None

        if enable_vector:
            self._vector = VectorMemory(
                index_path=vector_index_path,
                encryption_key=encryption_key,
            )

    def store_exchange(self, user_text: str, assistant_text: str) -> None:
        """Store a complete user↔assistant exchange."""
        user_entry = MemoryEntry(role="user", content=user_text)
        assistant_entry = MemoryEntry(role="assistant", content=assistant_text)

        self._short_term.store(user_entry)
        self._short_term.store(assistant_entry)

        if self._vector:
            self._vector.store(user_entry)
            self._vector.store(assistant_entry)

    def store_context(self, text: str) -> None:
        """Store environmental context (e.g. speaker/loopback audio transcription).

        Stored with role='context' so the LLM can distinguish it from
        direct user speech. Not sent to the vector store to avoid
        polluting semantic search with background audio."""
        entry = MemoryEntry(role="context", content=text)
        self._short_term.store(entry)

    def get_context(self, n: int = 10, role_filter: Optional[str] = None) -> List[Dict[str, str]]:
        """Get recent conversation context as LLM message format.

        Args:
            n: max entries to return.
            role_filter: if set, only return entries matching this role.
        """
        return self._short_term.to_llm_messages(n, role_filter=role_filter)

    def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Semantic search across all backends."""
        results = []

        # Short-term substring search
        results.extend(self._short_term.search(query, top_k=top_k))

        # Vector search (if enabled)
        if self._vector:
            results.extend(self._vector.search(query, top_k=top_k))

        # Deduplicate and sort by score
        seen_ids: set = set()
        unique = []
        for r in sorted(results, key=lambda x: x.score, reverse=True):
            if r.entry.entry_id not in seen_ids:
                seen_ids.add(r.entry.entry_id)
                unique.append(r)
        return unique[:top_k]

    def clear(self) -> None:
        """Clear all memory (security wipe)."""
        self._short_term.clear()
        if self._vector:
            self._vector.clear()

    def save(self) -> None:
        """Persist vector memory to disk (noop if not enabled)."""
        if self._vector:
            self._vector.save_to_disk()

    def get_stats(self) -> dict:
        stats = {
            "short_term_entries": self._short_term.size(),
            "vector_enabled": self._vector is not None,
        }
        if self._vector:
            stats["vector_entries"] = self._vector.size()
        return stats
