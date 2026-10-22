from pathlib import Path

import pytest

from aml_adapter.config import Settings, Track
from aml_adapter.embeddings import VectorChild
from aml_adapter.media import MediaStore
from aml_adapter.service import MemoryService
from aml_adapter.state_store import StateStore
from aml_adapter.vector_store import VectorStore


class DeterministicEmbedding:
    """Small fixed-width embedding used only by storage-level integration tests."""

    def embed_documents(self, children: list[VectorChild]) -> list[list[float]]:
        return [[1.0, 0.0] for _child in children]

    def embed_query(self, query, options) -> list[list[float]]:
        del query, options
        return [[1.0, 0.0]]


@pytest.fixture
def textual_runtime(tmp_path: Path) -> tuple[Settings, MemoryService, VectorStore]:
    settings = Settings.from_env(
        {"AML_TRACK": Track.TEXTUAL.value, "AML_DATA_DIR": str(tmp_path.resolve())}
    )
    state = StateStore(settings.state_db)
    vectors = VectorStore(settings.vectors_dir, settings.track)
    service = MemoryService(
        settings.track,
        state,
        MediaStore(settings.media_dir),
        DeterministicEmbedding(),
        vectors,
    )
    return settings, service, vectors
