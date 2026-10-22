from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from .config import Track
from .embeddings import VectorChild


@dataclass(frozen=True, slots=True)
class VectorCandidate:
    memory_id: str
    child_id: str
    score: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class VectorStore:
    def __init__(self, root: Path, track: Track) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.track = track
        self.collection_name = f"aml_{track.value}_v1"
        self.client = chromadb.PersistentClient(path=str(self.root))
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine", "aml_track": track.value},
            embedding_function=None,
        )

    def upsert(
        self,
        children: Sequence[VectorChild],
        vectors: Sequence[Sequence[float]],
        user_id: str,
    ) -> None:
        if len(children) != len(vectors):
            raise ValueError("children and vectors must have the same length")
        if not children:
            return
        created_at = _utc_now()
        self.collection.upsert(
            ids=[child.id for child in children],
            embeddings=[list(vector) for vector in vectors],
            documents=["" for _ in children],
            metadatas=[
                {
                    "memory_id": child.memory_id,
                    "user_id": user_id,
                    "track": self.track.value,
                    "kind": child.kind,
                    "created_at": created_at,
                }
                for child in children
            ],
        )

    def query(
        self, query_vectors: Sequence[Sequence[float]], user_id: str, limit: int
    ) -> list[VectorCandidate]:
        if not query_vectors or limit < 1:
            return []
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(vector) for vector in query_vectors],
            n_results=min(limit, count),
            where={"user_id": user_id},
            include=["metadatas", "distances"],
        )
        candidates: list[VectorCandidate] = []
        for ids, metadatas, distances in zip(
            result.get("ids") or [],
            result.get("metadatas") or [],
            result.get("distances") or [],
        ):
            for child_id, metadata, distance in zip(ids, metadatas, distances):
                if metadata is None or distance is None:
                    continue
                score = 1.0 - float(distance)
                if not math.isfinite(score):
                    continue
                candidates.append(
                    VectorCandidate(
                        memory_id=str(metadata["memory_id"]),
                        child_id=str(child_id),
                        score=score,
                    )
                )
        return sorted(
            candidates,
            key=lambda item: (-item.score, item.memory_id, item.child_id),
        )

    def contains_all(self, child_ids: Sequence[str]) -> bool:
        if not child_ids:
            return True
        result = self.collection.get(ids=list(child_ids), include=[])
        return set(result.get("ids") or []) == set(child_ids)
