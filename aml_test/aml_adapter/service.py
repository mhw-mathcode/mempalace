from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .config import Track
from .embeddings import EmbeddingAdapter, VectorChild
from .media import (
    MAX_SEARCH_IMAGE_BYTES,
    MediaStore,
    search_content,
    validate_request_media,
)
from .schemas import (
    AddRequest,
    AddResponse,
    ImagePart,
    ImageURL,
    SearchItem,
    SearchRequest,
    SearchResponse,
    TextPart,
    validate_add_for_track,
    validate_search_for_track,
)
from .state_store import StateStore, child_id_for, memory_id_for_request
from .vector_store import VectorStore


class VectorVisibilityError(RuntimeError):
    """The vector backend did not expose every child after an upsert."""


@dataclass
class _LockEntry:
    lock: Lock
    users: int = 0


class _RequestLocks:
    def __init__(self) -> None:
        self._guard = Lock()
        self._entries: dict[str, _LockEntry] = {}

    @contextmanager
    def acquire(self, request_id: str):
        with self._guard:
            entry = self._entries.get(request_id)
            if entry is None:
                entry = _LockEntry(lock=Lock())
                self._entries[request_id] = entry
            entry.users += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._guard:
                entry.users -= 1
                if entry.users == 0:
                    self._entries.pop(request_id, None)


class MemoryService:
    def __init__(
        self,
        track: Track,
        state: StateStore,
        media_store: MediaStore,
        embedding: EmbeddingAdapter,
        vector_store: VectorStore,
    ) -> None:
        self.track = track
        self.state = state
        self.media_store = media_store
        self.embedding = embedding
        self.vector_store = vector_store
        self._locks = _RequestLocks()

    def add(self, request: AddRequest) -> AddResponse:
        validate_add_for_track(request, self.track)
        decoded_images = validate_request_media(request.messages)
        content = search_content(request.messages, self.track)
        text = self._render_text(request)
        memory_id = memory_id_for_request(request.request_id)
        child_ids = self._child_ids(memory_id, bool(text), len(decoded_images))

        with self._locks.acquire(request.request_id):
            record, status = self.state.begin_add(
                request,
                content,
                child_ids,
                [image.ref() for image in decoded_images],
            )
            if status == "committed":
                return self._response(request)

            image_paths = [self.media_store.put(image) for image in decoded_images]
            children = self._vector_children(record.id, child_ids, text, image_paths)
            vectors = self.embedding.embed_documents(children)
            self.vector_store.upsert(children, vectors, request.user_id)
            if not self.vector_store.contains_all(child_ids):
                raise VectorVisibilityError("vector children are not immediately visible")
            self.state.mark_committed(request.request_id)
            return self._response(request)

    def recover_pending(self, request_id: str) -> AddResponse:
        request = self.state.load_add_request(request_id)
        if request is None:
            raise KeyError(request_id)
        return self.add(request)

    def search(self, request: SearchRequest) -> SearchResponse:
        validate_search_for_track(request, self.track)
        query = self._localize_query(request.query)
        query_vectors = self.embedding.embed_query(query, request.options)
        candidate_limit = min(max(request.top_k * 4, request.top_k), 1000)
        candidates = self.vector_store.query(
            query_vectors,
            request.user_id,
            candidate_limit,
        )

        best_scores: dict[str, float] = {}
        for candidate in candidates:
            score = float(candidate.score)
            if not math.isfinite(score):
                continue
            previous = best_scores.get(candidate.memory_id)
            if previous is None or score > previous:
                best_scores[candidate.memory_id] = score
        ranked_ids = sorted(best_scores, key=lambda memory_id: (-best_scores[memory_id], memory_id))
        hydrated = self.state.hydrate_committed(ranked_ids, request.user_id)

        items: list[SearchItem] = []
        response_media_bytes = 0
        for memory_id in ranked_ids:
            record = hydrated.get(memory_id)
            if record is None:
                continue
            memory_media_bytes = sum(ref.byte_size for ref in record.media_refs)
            if response_media_bytes + memory_media_bytes > MAX_SEARCH_IMAGE_BYTES:
                break
            items.append(
                SearchItem(
                    id=record.id,
                    content=record.content,
                    score=best_scores[memory_id],
                    created_at=record.created_at,
                )
            )
            response_media_bytes += memory_media_bytes
            if len(items) >= request.top_k:
                break
        return SearchResponse(data=items)

    @staticmethod
    def _response(request: AddRequest) -> AddResponse:
        return AddResponse(
            success=True,
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    def _render_text(self, request: AddRequest) -> str:
        rendered: list[str] = []
        for message in request.messages:
            if isinstance(message.content, str):
                text = message.content
            else:
                text = "\n".join(
                    part.text for part in message.content if isinstance(part, TextPart)
                )
            if text:
                rendered.append(f"role={message.role}\n{text}")
        return "\n\n".join(rendered)

    def _localize_query(self, query):
        if isinstance(query, str):
            return query
        decoded = iter(validate_request_media(query))
        localized = []
        for part in query:
            if isinstance(part, ImagePart):
                path = self.media_store.put(next(decoded))
                localized.append(
                    ImagePart(type="image_url", image_url=ImageURL(url=str(path)))
                )
            else:
                localized.append(part)
        return localized

    @staticmethod
    def _child_ids(memory_id: str, has_text: bool, image_count: int) -> list[str]:
        child_ids: list[str] = []
        if has_text:
            child_ids.append(child_id_for(memory_id, "text", 0))
        child_ids.extend(
            child_id_for(memory_id, "image", index) for index in range(image_count)
        )
        return child_ids

    @staticmethod
    def _vector_children(
        memory_id: str,
        child_ids: Sequence[str],
        text: str,
        image_paths: Sequence[Path],
    ) -> list[VectorChild]:
        children: list[VectorChild] = []
        offset = 0
        if text:
            children.append(
                VectorChild(id=child_ids[0], memory_id=memory_id, kind="text", value=text)
            )
            offset = 1
        children.extend(
            VectorChild(
                id=child_ids[offset + index],
                memory_id=memory_id,
                kind="image",
                value=str(path),
            )
            for index, path in enumerate(image_paths)
        )
        return children
