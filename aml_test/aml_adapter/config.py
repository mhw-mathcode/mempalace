from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility

    class StrEnum(str, Enum):
        pass


class Track(StrEnum):
    TEXTUAL = "textual"
    CODING = "coding"
    MULTIMODAL = "multimodal"


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _route(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value.startswith("/") or value.startswith("//"):
        raise ValueError(f"{name} must be an absolute route path")
    return value


@dataclass(frozen=True)
class Settings:
    track: Track
    data_dir: Path
    host: str
    port: int
    health_path: str
    add_path: str
    search_path: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        raw_track = _required(source, "AML_TRACK").lower()
        try:
            track = Track(raw_track)
        except ValueError as exc:
            choices = ", ".join(item.value for item in Track)
            raise ValueError(f"AML_TRACK must be one of: {choices}") from exc

        raw_data_dir = Path(_required(source, "AML_DATA_DIR")).expanduser()
        if not raw_data_dir.is_absolute():
            raise ValueError("AML_DATA_DIR must be an absolute path")
        data_dir = raw_data_dir.resolve()

        raw_port = source.get("AML_PORT", "8000").strip()
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError("AML_PORT must be an integer from 1 through 65535") from exc
        if not 1 <= port <= 65535:
            raise ValueError("AML_PORT must be an integer from 1 through 65535")

        health_path = _route(source, "AML_HEALTH_PATH", "/health")
        add_path = _route(source, "AML_ADD_PATH", "/add")
        search_path = _route(source, "AML_SEARCH_PATH", "/search")
        if len({health_path, add_path, search_path}) != 3:
            raise ValueError("AML route paths must be distinct")

        return cls(
            track=track,
            data_dir=data_dir,
            host=source.get("AML_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=port,
            health_path=health_path,
            add_path=add_path,
            search_path=search_path,
        )

    @property
    def state_db(self) -> Path:
        return self.data_dir / "state.sqlite3"

    @property
    def vectors_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"
