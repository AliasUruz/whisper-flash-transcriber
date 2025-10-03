"""Logging helpers that keep terminal output lean and copy-friendly."""
from __future__ import annotations

import logging
import os
import re
from typing import Iterable

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class _StripAnsiFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        if isinstance(record.msg, str):
            record.msg = ANSI_ESCAPE_RE.sub('', record.msg)
        if record.args:
            record.args = tuple(
                ANSI_ESCAPE_RE.sub('', arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def _determine_level() -> int:
    env_level = os.getenv("WHISPER_LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


def setup_logging(*, extra_filters: Iterable[logging.Filter] | None = None) -> None:
    """Configure root logging with a compact, copy-friendly format."""

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    filters: list[logging.Filter] = [_StripAnsiFilter()]
    if extra_filters:
        filters.extend(extra_filters)
    for filt in filters:
        handler.addFilter(filt)

    logging.basicConfig(level=_determine_level(), handlers=[handler], force=True)

    for noisy_logger in ("google", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
