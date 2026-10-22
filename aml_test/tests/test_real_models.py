import base64
import os
import sys
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from aml_adapter.config import Settings, Track
from aml_adapter.embeddings import EmbeddingAdapter
from aml_adapter.media import MediaStore
from aml_adapter.schemas import AddRequest, SearchRequest
from aml_adapter.service import MemoryService
from aml_adapter.state_store import StateStore
from aml_adapter.vector_store import VectorStore

pytestmark = pytest.mark.real_models
NO_LLM_MODULES = {
    "mempalace.closet_llm",
    "mempalace.llm_client",
    "mempalace.llm_refine",
}


def _require_real_models() -> None:
    if os.environ.get("AML_RUN_REAL_MODELS") != "1":
        pytest.skip("set AML_RUN_REAL_MODELS=1 after caching the required model weights")


def _service(root: Path, track: Track) -> MemoryService:
    settings = Settings.from_env({"AML_TRACK": track.value, "AML_DATA_DIR": str(root.resolve())})
    try:
        embedding = EmbeddingAdapter.for_track(track)
    except (ImportError, ModuleNotFoundError, RuntimeError) as error:
        pytest.skip(f"cached {track.value} embedding runtime is unavailable: {error}")
    return MemoryService(
        track,
        StateStore(settings.state_db),
        MediaStore(settings.media_dir),
        embedding,
        VectorStore(settings.vectors_dir, track),
    )


def _png_uri() -> str:
    output = BytesIO()
    Image.new("RGB", (8, 8), color=(240, 128, 32)).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _assert_no_llm_imported() -> None:
    assert NO_LLM_MODULES.isdisjoint(sys.modules)


def test_minilm_text_add_and_search_with_cached_weights(tmp_path: Path) -> None:
    _require_real_models()
    _assert_no_llm_imported()
    service = _service(tmp_path, Track.TEXTUAL)
    request = AddRequest.model_validate(
        {
            "request_id": "real-minilm-1",
            "messages": [{"role": "user", "content": "A copper lighthouse at dusk."}],
            "user_id": "real-user",
            "session_id": "real-session",
        }
    )

    service.add(request)
    response = service.search(
        SearchRequest.model_validate({"query": "lighthouse", "user_id": "real-user", "top_k": 3})
    )

    assert response.data and response.data[0].content == "A copper lighthouse at dusk."
    _assert_no_llm_imported()


def test_embeddinggemma2_text_and_png_add_and_search_with_cached_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_real_models()
    monkeypatch.setenv("MEMPALACE_EMBEDDINGGEMMA2_MODALITIES", "text+vision")
    _assert_no_llm_imported()
    service = _service(tmp_path, Track.MULTIMODAL)
    image = _png_uri()
    request = AddRequest.model_validate(
        {
            "request_id": "real-gemma2-1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "An orange square."},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                }
            ],
            "user_id": "real-user",
            "session_id": "real-session",
        }
    )

    service.add(request)
    response = service.search(
        SearchRequest.model_validate({"query": "orange square", "user_id": "real-user", "top_k": 3})
    )

    assert response.data and response.data[0].content[1].image_url.url == image
    _assert_no_llm_imported()
