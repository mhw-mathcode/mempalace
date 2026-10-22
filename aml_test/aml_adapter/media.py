from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .config import Track
from .schemas import ContentPart, ImagePart, Message, TextPart

MIB = 1024 * 1024
MAX_IMAGE_BYTES = 10 * MIB
MAX_ADD_IMAGE_BYTES = 30 * MIB

_DATA_URI = re.compile(
    r"\Adata:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/]*={0,2})\Z"
)
_MIME_FORMAT = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}
_MIME_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class MediaError(ValueError):
    """An AML media value is invalid or exceeds a published limit."""


@dataclass(frozen=True, slots=True)
class MediaRef:
    sha256: str
    mime_type: str
    suffix: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class DecodedImage:
    mime_type: str
    data_uri: str
    data: bytes
    sha256: str
    suffix: str
    byte_size: int

    def ref(self) -> MediaRef:
        return MediaRef(
            sha256=self.sha256,
            mime_type=self.mime_type,
            suffix=self.suffix,
            byte_size=self.byte_size,
        )


def decode_image_data_uri(url: str) -> DecodedImage:
    match = _DATA_URI.fullmatch(url)
    if match is None:
        raise MediaError("image_url.url must be an inline Base64 JPEG, PNG, or WebP Data URI")
    mime_type, encoded = match.groups()
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 4:
        raise MediaError("decoded image exceeds the 10 MiB limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaError("image Data URI contains invalid Base64") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise MediaError("decoded image exceeds the 10 MiB limit")
    if not data:
        raise MediaError("decoded image is empty")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                actual_format = image.format
                image.verify()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MediaError("image has unsafe dimensions") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MediaError("image bytes could not be decoded") from exc

    if actual_format != _MIME_FORMAT[mime_type]:
        raise MediaError("declared image MIME type does not match decoded image")

    return DecodedImage(
        mime_type=mime_type,
        data_uri=url,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        suffix=_MIME_SUFFIX[mime_type],
        byte_size=len(data),
    )


def _parts(values: Sequence[object]):
    for value in values:
        if isinstance(value, Message):
            if isinstance(value.content, str):
                continue
            yield from value.content
        else:
            yield value


def validate_request_media(
    messages_or_parts: Sequence[object], limit_bytes: int = MAX_ADD_IMAGE_BYTES
) -> list[DecodedImage]:
    decoded: list[DecodedImage] = []
    total = 0
    for part in _parts(messages_or_parts):
        if not isinstance(part, ImagePart):
            continue
        image = decode_image_data_uri(part.image_url.url)
        total += image.byte_size
        if total > limit_bytes:
            raise MediaError("decoded images exceed the 30 MiB Add-request limit")
        decoded.append(image)
    return decoded


class MediaStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, image: DecodedImage) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{image.sha256}{image.suffix}"
        if destination.is_file():
            return destination

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".blob-", suffix=".tmp", dir=self.root, delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(image.data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
            return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def search_content(messages: Sequence[Message], track: Track) -> str | list[ContentPart]:
    if track in {Track.TEXTUAL, Track.CODING}:
        return "\n\n".join(
            message.content for message in messages if isinstance(message.content, str)
        )

    result: list[ContentPart] = []
    for message in messages:
        if isinstance(message.content, str):
            result.append(TextPart(type="text", text=message.content))
        else:
            result.extend(message.content)
    return result
