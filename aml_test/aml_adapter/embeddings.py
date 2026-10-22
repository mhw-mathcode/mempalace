from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mempalace.embedding import get_embedding_function

from .config import Track
from .schemas import ContentPart, ImagePart, TextPart


class EmbeddingOutputError(RuntimeError):
    """An embedding backend returned vectors that cannot be indexed safely."""


@dataclass(frozen=True, slots=True)
class VectorChild:
    id: str
    memory_id: str
    kind: Literal["text", "image"]
    value: str


def _normalize_rows(raw, expected_rows: int) -> list[list[float]]:
    try:
        rows = [list(row) for row in raw]
    except (TypeError, ValueError) as exc:
        raise EmbeddingOutputError("embedding output must be a two-dimensional matrix") from exc
    if len(rows) != expected_rows:
        raise EmbeddingOutputError(
            f"embedding output returned {len(rows)} rows for {expected_rows} inputs"
        )
    if not rows:
        raise EmbeddingOutputError("embedding output is empty")
    dimension = len(rows[0])
    if dimension < 1 or any(len(row) != dimension for row in rows):
        raise EmbeddingOutputError("embedding output has an inconsistent dimension")

    normalized: list[list[float]] = []
    for row in rows:
        try:
            vector = [float(value) for value in row]
        except (TypeError, ValueError) as exc:
            raise EmbeddingOutputError("embedding output contains a non-numeric value") from exc
        if not all(math.isfinite(value) for value in vector):
            raise EmbeddingOutputError("embedding output contains NaN or infinity")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise EmbeddingOutputError("embedding output contains a zero or invalid vector")
        normalized.append([value / norm for value in vector])
    return normalized


def _options_context(text: str, options: Sequence[str] | None) -> str:
    if not options:
        return text
    return f"{text}\n\nOptions:\n" + "\n".join(options)


class EmbeddingAdapter:
    def __init__(self, track: Track, function) -> None:
        self.track = track
        self.function = function

    @classmethod
    def for_track(cls, track: Track) -> EmbeddingAdapter:
        model = "embeddinggemma2" if track is Track.MULTIMODAL else "minilm"
        return cls(track, get_embedding_function(model=model))

    def embed_documents(self, children: Sequence[VectorChild]) -> list[list[float]]:
        if not children:
            return []
        text_positions = [index for index, child in enumerate(children) if child.kind == "text"]
        image_positions = [index for index, child in enumerate(children) if child.kind == "image"]
        if image_positions and self.track is not Track.MULTIMODAL:
            raise ValueError("image embedding is available only for the multimodal track")

        vectors: list[list[float] | None] = [None] * len(children)
        if text_positions:
            values = [children[index].value for index in text_positions]
            rows = _normalize_rows(self.function.embed_documents(values), len(values))
            for index, vector in zip(text_positions, rows):
                vectors[index] = vector
        for index in image_positions:
            raw = self.function.embed_image(children[index].value)
            vectors[index] = _normalize_rows([raw], 1)[0]
        return self._validate_common_dimension(vectors)

    def embed_query(
        self, query: str | Sequence[ContentPart], options: Sequence[str] | None
    ) -> list[list[float]]:
        if isinstance(query, str):
            text = _options_context(query, options)
            return _normalize_rows(self.function.embed_query([text]), 1)
        if self.track is not Track.MULTIMODAL:
            raise ValueError("content-part queries are available only for the multimodal track")

        vectors: list[list[float]] = []
        text_seen = False
        for part in query:
            if isinstance(part, TextPart):
                text = _options_context(part.text, options if not text_seen else None)
                vectors.extend(_normalize_rows(self.function.embed_query([text]), 1))
                text_seen = True
            elif isinstance(part, ImagePart):
                vectors.extend(_normalize_rows([self.function.embed_image(part.image_url.url)], 1))
        if options and not text_seen:
            vectors.extend(
                _normalize_rows(
                    self.function.embed_query([_options_context("", options)]),
                    1,
                )
            )
        if not vectors:
            raise EmbeddingOutputError("embedding output is empty")
        return self._validate_common_dimension(vectors)

    @staticmethod
    def _validate_common_dimension(
        vectors: Sequence[list[float] | None],
    ) -> list[list[float]]:
        if not vectors or any(vector is None for vector in vectors):
            raise EmbeddingOutputError("embedding output is incomplete")
        complete = [vector for vector in vectors if vector is not None]
        dimension = len(complete[0])
        if any(len(vector) != dimension for vector in complete):
            raise EmbeddingOutputError("embedding output has an inconsistent dimension")
        return complete
