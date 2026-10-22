import math

import pytest

from aml_adapter.config import Track
from aml_adapter.embeddings import EmbeddingAdapter, EmbeddingOutputError, VectorChild
from aml_adapter.schemas import ImagePart, ImageURL, TextPart


class FakeEmbeddingFunction:
    def __init__(self) -> None:
        self.dimension = 2
        self.document_inputs = []
        self.query_inputs = []
        self.image_inputs = []

    def embed_documents(self, values):
        self.document_inputs.append(list(values))
        return [[3.0, 4.0] for _ in values]

    def embed_query(self, values):
        self.query_inputs.append(list(values))
        return [[0.0, 2.0] for _ in values]

    def embed_image(self, value):
        self.image_inputs.append(value)
        return [5.0, 0.0]


def test_for_track_selects_mempalace_original_models(monkeypatch) -> None:
    requested = []

    def factory(*, model):
        requested.append(model)
        return FakeEmbeddingFunction()

    monkeypatch.setattr("aml_adapter.embeddings.get_embedding_function", factory)

    EmbeddingAdapter.for_track(Track.TEXTUAL)
    EmbeddingAdapter.for_track(Track.CODING)
    EmbeddingAdapter.for_track(Track.MULTIMODAL)

    assert requested == ["minilm", "minilm", "embeddinggemma2"]


def test_real_minilm_factory_uses_callable_document_protocol_without_inference(
    monkeypatch,
) -> None:
    function = EmbeddingAdapter.for_track(Track.TEXTUAL).function
    assert callable(function)
    assert not hasattr(function, "embed_documents")
    monkeypatch.setattr(
        type(function),
        "__call__",
        lambda _self, values: [[1.0] * 384 for _value in values],
    )
    adapter = EmbeddingAdapter(Track.TEXTUAL, function)

    vectors = adapter.embed_documents(
        [VectorChild(id="v", memory_id="m", kind="text", value="document")]
    )

    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_document_embedding_routes_text_and_native_images_in_source_order(monkeypatch) -> None:
    function = FakeEmbeddingFunction()
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: function
    )
    adapter = EmbeddingAdapter.for_track(Track.MULTIMODAL)
    children = [
        VectorChild(id="v1", memory_id="m", kind="text", value="document text"),
        VectorChild(id="v2", memory_id="m", kind="image", value="C:/media/image.png"),
        VectorChild(id="v3", memory_id="m", kind="text", value="more text"),
    ]

    vectors = adapter.embed_documents(children)

    assert function.document_inputs == [["document text", "more text"]]
    assert function.image_inputs == ["C:/media/image.png"]
    assert vectors == [[0.6, 0.8], [1.0, 0.0], [0.6, 0.8]]


def test_text_query_appends_ordered_options_exactly(monkeypatch) -> None:
    function = FakeEmbeddingFunction()
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: function
    )
    adapter = EmbeddingAdapter.for_track(Track.CODING)

    vectors = adapter.embed_query("question", ["A. first", "B. second"])

    assert function.query_inputs == [["question\n\nOptions:\nA. first\nB. second"]]
    assert vectors == [[0.0, 1.0]]


def test_multimodal_query_routes_ordered_text_and_image_parts(monkeypatch) -> None:
    function = FakeEmbeddingFunction()
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: function
    )
    adapter = EmbeddingAdapter.for_track(Track.MULTIMODAL)
    parts = [
        TextPart(type="text", text="question"),
        ImagePart(type="image_url", image_url=ImageURL(url="C:/media/query.png")),
    ]

    vectors = adapter.embed_query(parts, ["A", "B"])

    assert function.query_inputs == [["question\n\nOptions:\nA\nB"]]
    assert function.image_inputs == ["C:/media/query.png"]
    assert vectors == [[0.0, 1.0], [1.0, 0.0]]


def test_non_multimodal_adapter_rejects_image_children(monkeypatch) -> None:
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: FakeEmbeddingFunction()
    )
    adapter = EmbeddingAdapter.for_track(Track.TEXTUAL)

    with pytest.raises(ValueError, match="multimodal"):
        adapter.embed_documents(
            [VectorChild(id="v", memory_id="m", kind="image", value="image.png")]
        )


@pytest.mark.parametrize(
    "result",
    [
        [[math.nan, 1.0]],
        [[math.inf, 1.0]],
        [[0.0, 0.0]],
        [[1.0], [1.0, 2.0]],
        [],
    ],
)
def test_malformed_document_outputs_fail_closed(monkeypatch, result) -> None:
    function = FakeEmbeddingFunction()
    function.embed_documents = lambda _values: result
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: function
    )
    adapter = EmbeddingAdapter.for_track(Track.TEXTUAL)
    children = [VectorChild(id="v", memory_id="m", kind="text", value="text")]

    with pytest.raises(EmbeddingOutputError):
        adapter.embed_documents(children)


def test_mixed_output_dimensions_fail_closed(monkeypatch) -> None:
    function = FakeEmbeddingFunction()
    function.embed_documents = lambda _values: [[1.0, 0.0]]
    function.embed_image = lambda _value: [1.0, 0.0, 0.0]
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: function
    )
    adapter = EmbeddingAdapter.for_track(Track.MULTIMODAL)

    with pytest.raises(EmbeddingOutputError, match="dimension"):
        adapter.embed_documents(
            [
                VectorChild(id="v1", memory_id="m", kind="text", value="text"),
                VectorChild(id="v2", memory_id="m", kind="image", value="image.png"),
            ]
        )


def test_empty_query_output_fails_closed(monkeypatch) -> None:
    function = FakeEmbeddingFunction()
    function.embed_query = lambda _values: []
    monkeypatch.setattr(
        "aml_adapter.embeddings.get_embedding_function", lambda **_kwargs: function
    )

    with pytest.raises(EmbeddingOutputError):
        EmbeddingAdapter.for_track(Track.TEXTUAL).embed_query("question", None)


def test_model_dimension_is_enforced_across_document_and_query_calls() -> None:
    function = FakeEmbeddingFunction()
    adapter = EmbeddingAdapter(Track.TEXTUAL, function)
    child = VectorChild(id="v", memory_id="m", kind="text", value="text")
    assert adapter.embed_documents([child]) == [[0.6, 0.8]]

    function.embed_query = lambda _values: [[1.0, 0.0, 0.0]]
    with pytest.raises(EmbeddingOutputError, match="expected dimension 2"):
        adapter.embed_query("question", None)
