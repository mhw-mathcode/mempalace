import base64
import struct
import zlib
from io import BytesIO

import pytest
from PIL import Image

from aml_adapter.config import Track
from aml_adapter.media import (
    MAX_IMAGE_BYTES,
    MediaError,
    MediaStore,
    decode_image_data_uri,
    search_content,
    validate_request_media,
)
from aml_adapter.schemas import AddRequest


def image_bytes(format_name: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=(20, 40, 60)).save(output, format=format_name)
    return output.getvalue()


def data_uri(data: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@pytest.mark.parametrize(
    ("format_name", "mime_type", "suffix"),
    [("PNG", "image/png", ".png"), ("JPEG", "image/jpeg", ".jpg"), ("WEBP", "image/webp", ".webp")],
)
def test_decode_accepts_supported_image_formats(
    format_name: str, mime_type: str, suffix: str
) -> None:
    raw = image_bytes(format_name)

    decoded = decode_image_data_uri(data_uri(raw, mime_type))

    assert decoded.data == raw
    assert decoded.mime_type == mime_type
    assert decoded.suffix == suffix
    assert decoded.byte_size == len(raw)
    assert len(decoded.sha256) == 64


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/image.png",
        "file:///tmp/image.png",
        "data:image/gif;base64,R0lGODlh",
        "data:image/png;base64,not base64!",
        "data:image/png;utf8,hello",
    ],
)
def test_decode_rejects_remote_unsupported_and_non_strict_data(url: str) -> None:
    with pytest.raises(MediaError):
        decode_image_data_uri(url)


def test_decode_accepts_exact_byte_limit_and_rejects_one_byte_more() -> None:
    png = image_bytes("PNG")
    exact = png + b"\0" * (MAX_IMAGE_BYTES - len(png))

    assert decode_image_data_uri(data_uri(exact)).byte_size == MAX_IMAGE_BYTES
    with pytest.raises(MediaError, match="10 MiB"):
        decode_image_data_uri(data_uri(exact + b"x"))


def test_decode_rejects_corrupt_and_mime_mismatched_images() -> None:
    with pytest.raises(MediaError, match="decode"):
        decode_image_data_uri(data_uri(b"not an image"))
    with pytest.raises(MediaError, match="does not match"):
        decode_image_data_uri(data_uri(image_bytes("PNG"), "image/jpeg"))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def test_decode_rejects_decompression_bomb_dimensions() -> None:
    ihdr = struct.pack(">IIBBBBB", 100_000, 100_000, 8, 2, 0, 0, 0)
    huge_png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")

    with pytest.raises(MediaError, match="unsafe dimensions"):
        decode_image_data_uri(data_uri(huge_png))


def test_validate_request_media_preserves_order_and_enforces_aggregate_limit() -> None:
    first = data_uri(image_bytes("PNG"))
    second = data_uri(image_bytes("JPEG"), "image/jpeg")
    request = AddRequest.model_validate(
        {
            "request_id": "r",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": first}},
                        {"type": "text", "text": "between"},
                        {"type": "image_url", "image_url": {"url": second}},
                    ],
                }
            ],
            "user_id": "u",
            "session_id": "s",
        }
    )

    decoded = validate_request_media(request.messages)
    assert [item.data_uri for item in decoded] == [first, second]

    total = sum(item.byte_size for item in decoded)
    with pytest.raises(MediaError, match="30 MiB"):
        validate_request_media(request.messages, limit_bytes=total - 1)


def test_media_store_uses_digest_filename_and_skips_existing_write(tmp_path, monkeypatch) -> None:
    decoded = decode_image_data_uri(data_uri(image_bytes("PNG")))
    store = MediaStore(tmp_path)

    path = store.put(decoded)
    assert path == tmp_path / f"{decoded.sha256}.png"
    assert path.read_bytes() == decoded.data

    monkeypatch.setattr("aml_adapter.media.os.replace", lambda *_: pytest.fail("rewrote blob"))
    assert store.put(decoded) == path


def test_media_store_removes_temporary_file_when_replace_fails(tmp_path, monkeypatch) -> None:
    decoded = decode_image_data_uri(data_uri(image_bytes("PNG")))
    store = MediaStore(tmp_path)

    def fail_replace(*_args) -> None:
        raise OSError("disk failed")

    monkeypatch.setattr("aml_adapter.media.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failed"):
        store.put(decoded)

    assert list(tmp_path.iterdir()) == []


def test_search_content_round_trips_text_and_preserves_multimodal_order() -> None:
    text_request = AddRequest.model_validate(
        {
            "request_id": "r1",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
            "user_id": "u",
            "session_id": "s",
        }
    )
    assert search_content(text_request.messages[:1], Track.TEXTUAL) == "first"
    assert search_content(text_request.messages, Track.CODING) == "first\n\nsecond"

    image = data_uri(image_bytes("PNG"))
    multimodal_request = AddRequest.model_validate(
        {
            "request_id": "r2",
            "messages": [
                {"role": "user", "content": "before"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": "after"},
                    ],
                },
            ],
            "user_id": "u",
            "session_id": "s",
        }
    )
    content = search_content(multimodal_request.messages, Track.MULTIMODAL)
    assert [part.model_dump(mode="json") for part in content] == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": "after"},
    ]
