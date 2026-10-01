"""Text embeddings from an Ollama embedding model, for the semantic half of
the folder shortlist (see candidates.rank_candidates).

The lexical shortlist compares words; it cannot see that "Curriculum Vitae"
belongs with "Lebenslauf" or an English manual with "Manuals". An embedding
model can, at the price of blurring exact names - so the two are combined,
not replaced (candidates.fuse).

The embedding model runs on the CPU (`num_gpu: 0`): the 6-GB card is full
with the chat model, and a second model there would evict it - a cold
reload of the chat model costs more than every embedding of a document put
together. Measured on the server: ~0.1 s per text on the CPU.

Vectors are cached by (model, text) in a small sqlite file: folder texts
only change when the tree does, so after the first run nearly every folder
is a cache hit and one document costs one embedding call.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from pathlib import Path

import ollama

log = logging.getLogger(__name__)

EMBED_KEEP_ALIVE = "30m"
_OPTIONS = {"num_gpu": 0}
_BATCH = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
"""


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


class Embedder:
    def __init__(self, host: str, model: str, cache_path: str | Path | None = None, timeout: float = 120.0):
        self.host = host
        self.model = model
        self._timeout = timeout
        self._conn: sqlite3.Connection | None = None
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _cached(self, keys: list[str]) -> dict[str, list[float]]:
        if self._conn is None or not keys:
            return {}
        found: dict[str, list[float]] = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i:i + 500]
            rows = self._conn.execute(
                f"SELECT text_hash, vector FROM vectors WHERE model = ? AND text_hash IN ({','.join('?' * len(chunk))})",
                (self.model, *chunk),
            ).fetchall()
            found.update((k, _unpack(v)) for k, v in rows)
        return found

    def _store(self, items: list[tuple[str, list[float]]]) -> None:
        if self._conn is None or not items:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vectors (model, text_hash, vector) VALUES (?, ?, ?)",
                [(self.model, key, _pack(vector)) for key, vector in items],
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order. Raises on an Ollama failure (the
        caller decides whether to go on without the semantic half)."""
        keys = [self._key(t) for t in texts]
        vectors = self._cached(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in vectors))
        if missing:
            client = ollama.Client(host=self.host, timeout=self._timeout)
            fresh: list[tuple[str, list[float]]] = []
            for i in range(0, len(missing), _BATCH):
                batch = missing[i:i + _BATCH]
                response = client.embed(model=self.model, input=batch, options=_OPTIONS, keep_alive=EMBED_KEEP_ALIVE)
                for text, vector in zip(batch, response["embeddings"]):
                    key = self._key(text)
                    vectors[key] = vector
                    fresh.append((key, vector))
            self._store(fresh)
            log.info("Embedded %d new text(s) with %s", len(missing), self.model)
        return [vectors[k] for k in keys]

    def preload(self) -> None:
        """Load the model (a cold load takes a few seconds). Best effort."""
        try:
            ollama.Client(host=self.host, timeout=self._timeout).embed(
                model=self.model, input="DEPOT", options=_OPTIONS, keep_alive=EMBED_KEEP_ALIVE
            )
        except Exception as exc:
            log.debug("Embedding model preload failed (ignored): %s", exc)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0
