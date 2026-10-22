from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import Settings


def configure_request_logging(handler: logging.Handler | None = None) -> None:
    request_logger = logging.getLogger("aml_adapter")
    if handler is not None:
        request_logger.handlers[:] = [handler]
    elif not request_logger.handlers:
        request_logger.addHandler(logging.StreamHandler())
    for configured_handler in request_logger.handlers:
        configured_handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.setLevel(logging.INFO)
    request_logger.propagate = False


def main() -> None:
    settings = Settings.from_env()
    configure_request_logging()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
