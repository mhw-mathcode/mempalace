from __future__ import annotations

from dataclasses import dataclass

from mempalace.embedding import EmbeddingFunctionUnavailableError

from .media import MediaError, MediaLimitError
from .state_store import IdempotencyConflict


class ServiceUnavailableError(RuntimeError):
    """A required model or vector service is temporarily unavailable."""


@dataclass(frozen=True, slots=True)
class ErrorMapping:
    status: int
    reason: str


def classify_error(error: Exception) -> ErrorMapping:
    if isinstance(error, MediaLimitError):
        return ErrorMapping(413, "media_limit_exceeded")
    if isinstance(error, MediaError):
        return ErrorMapping(400, "invalid_media")
    if isinstance(error, IdempotencyConflict):
        return ErrorMapping(409, "idempotency_conflict")
    if isinstance(error, (ServiceUnavailableError, EmbeddingFunctionUnavailableError)):
        return ErrorMapping(503, "service_unavailable")
    if isinstance(error, ValueError):
        return ErrorMapping(400, "invalid_request")
    return ErrorMapping(500, "internal_error")
