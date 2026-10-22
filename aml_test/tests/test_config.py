from pathlib import Path

import pytest

from aml_adapter.config import Settings, Track


def test_settings_require_track_and_absolute_data_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AML_TRACK"):
        Settings.from_env({"AML_DATA_DIR": str(tmp_path)})

    with pytest.raises(ValueError, match="AML_DATA_DIR"):
        Settings.from_env({"AML_TRACK": "textual"})

    with pytest.raises(ValueError, match="absolute"):
        Settings.from_env({"AML_TRACK": "textual", "AML_DATA_DIR": "relative/data"})


def test_settings_apply_defaults_and_derive_storage_paths(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {"AML_TRACK": "textual", "AML_DATA_DIR": str(tmp_path.resolve())}
    )

    assert settings.track is Track.TEXTUAL
    assert settings.data_dir == tmp_path.resolve()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.health_path == "/health"
    assert settings.add_path == "/add"
    assert settings.search_path == "/search"
    assert settings.state_db == tmp_path.resolve() / "state.sqlite3"
    assert settings.vectors_dir == tmp_path.resolve() / "vectors"
    assert settings.media_dir == tmp_path.resolve() / "media"


def test_settings_reject_unknown_track(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AML_TRACK"):
        Settings.from_env({"AML_TRACK": "academic", "AML_DATA_DIR": str(tmp_path.resolve())})


def test_settings_accept_route_overrides_and_runtime_values(tmp_path: Path) -> None:
    settings = Settings.from_env(
        {
            "AML_TRACK": "multimodal",
            "AML_DATA_DIR": str(tmp_path.resolve()),
            "AML_HOST": "127.0.0.1",
            "AML_PORT": "9010",
            "AML_HEALTH_PATH": "/ready",
            "AML_ADD_PATH": "/v1/memory/add",
            "AML_SEARCH_PATH": "/v1/memory/search",
        }
    )

    assert settings.track is Track.MULTIMODAL
    assert settings.host == "127.0.0.1"
    assert settings.port == 9010
    assert (settings.health_path, settings.add_path, settings.search_path) == (
        "/ready",
        "/v1/memory/add",
        "/v1/memory/search",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"AML_ADD_PATH": "v1/add"}, "absolute route"),
        ({"AML_ADD_PATH": "/same", "AML_SEARCH_PATH": "/same"}, "distinct"),
        ({"AML_PORT": "0"}, "AML_PORT"),
        ({"AML_PORT": "true"}, "AML_PORT"),
    ],
)
def test_settings_reject_invalid_routes_and_port(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    env = {"AML_TRACK": "coding", "AML_DATA_DIR": str(tmp_path.resolve()), **overrides}

    with pytest.raises(ValueError, match=message):
        Settings.from_env(env)
