from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from .config import Track


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ImageURL(StrictModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return _non_blank(value, "image_url.url")


class TextPart(StrictModel):
    type: Literal["text"]
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _non_blank(value, "text")


class ImagePart(StrictModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = str | Annotated[list[ContentPart], Field(min_length=1)]


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: Content
    timestamp: StrictInt | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: Content) -> Content:
        if isinstance(value, str):
            return _non_blank(value, "content")
        return value


class AddRequest(StrictModel):
    request_id: str
    messages: Annotated[list[Message], Field(min_length=1)]
    user_id: str
    session_id: str

    @field_validator("request_id", "user_id", "session_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _non_blank(value, info.field_name)


class AddResponse(StrictModel):
    success: Literal[True]
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(StrictModel):
    query: Content
    options: Annotated[list[str], Field(min_length=1)] | None = None
    user_id: str
    top_k: Annotated[StrictInt, Field(ge=1, le=100)]

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: Content) -> Content:
        if isinstance(value, str):
            return _non_blank(value, "query")
        return value

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        return _non_blank(value, "user_id")

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            for option in value:
                _non_blank(option, "options item")
        return value


class SearchItem(StrictModel):
    id: str
    content: Content
    score: float | None = None
    created_at: str | None = None


class SearchResponse(StrictModel):
    data: list[SearchItem]


def validate_add_for_track(request: AddRequest, track: Track) -> None:
    if track in {Track.TEXTUAL, Track.CODING} and any(
        not isinstance(message.content, str) for message in request.messages
    ):
        raise ValueError(f"{track.value} Add requires string content")


def validate_search_for_track(request: SearchRequest, track: Track) -> None:
    if track in {Track.TEXTUAL, Track.CODING} and not isinstance(request.query, str):
        raise ValueError(f"{track.value} Search requires a string query")
