"""HybridMemoryAgent — episodic memory (Vector Store) + stable profile (Feature Store).

A minimal POC for the Day-19 bonus challenge. Combines:
  - Qdrant in-memory (episodic memory) — what the user has read / said, per-user.
  - rank-bm25 + vector + RRF (hybrid retrieval, reused from NB2).
  - Feast online store (stable profile + recent activity, reused from NB4).

The agent does NOT call a real LLM — `recall()` returns the *assembled context
string* a real LLM prompt would be built from. That keeps the POC about the
memory architecture, not generation.

Run the demo: `python bonus/demo.py`
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi

REPO_ROOT = Path(__file__).resolve().parent.parent
FEAST_DIR = REPO_ROOT / "app" / "feast_repo"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION = "episodic_memory"
RRF_K = 60


def chunk_text(text: str, max_chars: int = 280) -> list[str]:
    """Sentence-aware chunking with a char cap.

    Per-message granularity is too fine (loses context); per-conversation too
    coarse (dilutes the embedding). We split on sentence boundaries and pack up
    to ~280 chars/chunk — a tradeoff documented in ARCHITECTURE.md §1.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + len(s) + 1 > max_chars:
            chunks.append(buf)
            buf = s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks or [text.strip()]


class HybridMemoryAgent:
    """Per-user episodic memory + shared feature-store profile."""

    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        # Local mirror of stored chunks for BM25 (Qdrant has no built-in sparse here).
        self.chunks: list[dict] = []  # {id, user_id, text}
        self._next_id = 0
        # Feast is optional — degrade gracefully if the repo hasn't been applied.
        self.fs = None
        try:
            from feast import FeatureStore

            self.fs = FeatureStore(repo_path=str(FEAST_DIR))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Feast unavailable ({exc}); profile context disabled.", file=sys.stderr)

    # ── write path ──────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001") -> None:
        """Add a new piece of episodic memory for this user."""
        pieces = chunk_text(text)
        vectors = list(self.embedder.embed(pieces))
        points = []
        for piece, vec in zip(pieces, vectors):
            cid = self._next_id
            self._next_id += 1
            self.chunks.append({"id": cid, "user_id": user_id, "text": piece})
            points.append(PointStruct(
                id=cid,
                vector=vec.tolist(),
                payload={"user_id": user_id, "text": piece},
            ))
        self.client.upsert(collection_name=COLLECTION, points=points)

    # ── read path ───────────────────────────────────────────────────────
    def _hybrid_search(self, query: str, user_id: str, top_k: int = 3) -> list[str]:
        """RRF fusion of BM25 + vector, scoped to one user's memories."""
        mine = [c for c in self.chunks if c["user_id"] == user_id]
        if not mine:
            return []

        # Vector side — filter by user_id payload so users never see each other's memory.
        q_vec = next(self.embedder.embed([query])).tolist()
        sem = self.client.query_points(
            collection_name=COLLECTION,
            query=q_vec,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
            limit=max(top_k * 5, 20),
        ).points
        sem_ids = [p.id for p in sem]

        # Keyword side — BM25 over this user's chunks.
        bm25 = BM25Okapi([c["text"].lower().split() for c in mine])
        scores = bm25.get_scores(query.lower().split())
        kw_local = sorted(range(len(mine)), key=lambda i: -scores[i])[: max(top_k * 5, 20)]
        kw_ids = [mine[i]["id"] for i in kw_local]

        # Reciprocal Rank Fusion (k=60, rank 1-based) — same formula as NB2.
        rrf: dict[int, float] = {}
        for ids in (kw_ids, sem_ids):
            for rank, cid in enumerate(ids, start=1):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        by_id = {c["id"]: c["text"] for c in mine}
        top = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [by_id[cid] for cid, _ in top]

    def _profile(self, user_id: str) -> dict:
        """Pull stable profile + recent activity from the Feast online store."""
        if self.fs is None:
            return {}
        feats = self.fs.get_online_features(
            features=[
                "user_profile_features:reading_speed_wpm",
                "user_profile_features:preferred_language",
                "user_profile_features:topic_affinity",
                "query_velocity_features:queries_last_hour",
                "query_velocity_features:distinct_topics_24h",
            ],
            entity_rows=[{"user_id": user_id}],
        ).to_dict()
        return {k: v[0] for k, v in feats.items()}

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Retrieve top-K memories + profile features → assembled context string."""
        prof = self._profile(user_id)
        memories = self._hybrid_search(query, user_id)

        lines = [f"### Context for query: {query!r} (user={user_id})"]
        if prof:
            lines.append(
                f"Profile: thích chủ đề '{prof.get('topic_affinity')}', "
                f"ngôn ngữ '{prof.get('preferred_language')}', "
                f"đọc {prof.get('reading_speed_wpm')} wpm."
            )
            lines.append(
                f"Recent activity: {prof.get('queries_last_hour')} query/giờ qua, "
                f"{prof.get('distinct_topics_24h')} chủ đề khác nhau /24h."
            )
        else:
            lines.append("Profile: (không có — Feast chưa apply/materialize).")

        if memories:
            lines.append("Top memories liên quan:")
            for i, m in enumerate(memories, 1):
                lines.append(f"  {i}. {m}")
        else:
            lines.append("Top memories: (chưa có ký ức nào cho user này).")
        return "\n".join(lines)
