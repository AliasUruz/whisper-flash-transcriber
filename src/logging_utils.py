"""Logging helpers that keep terminal output lean and copy-friendly."""
from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Sequence

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


class _ContextAugmentFilter(logging.Filter):
    """Ensure records include the standard structured logging context."""

    _FALLBACK_COMPONENT = "app"

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple enrichment
        if not getattr(record, "component", None):
            # Derive component from the logger name to avoid empty placeholders.
            component = record.name.rsplit(".", maxsplit=1)[-1] if record.name else self._FALLBACK_COMPONENT
            record.component = component
        if not getattr(record, "structured_context", None):
            record.structured_context = ""
        return True


def _render_structured_context(record: logging.LogRecord, *, keys: Sequence[str]) -> str:
    """Serialize known contextual attributes into a key=value representation."""

    parts: list[str] = []
    for key in keys:
        value = getattr(record, key, None)
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


class StructuredFormatter(logging.Formatter):
    """Formatter that appends structured metadata when available."""

    _CONTEXT_KEYS: Sequence[str] = (
        "event",
        "stage",
        "state",
        "action",
        "status",
        "details",
        "path",
        "duration_ms",
    )

    def format(self, record: logging.LogRecord) -> str:
        context = _render_structured_context(record, keys=self._CONTEXT_KEYS)
        record.structured_context = f" | {context}" if context else ""
        return super().format(record)


def _determine_level() -> int:
    env_level = os.getenv("WHISPER_LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


def setup_logging(*, extra_filters: Iterable[logging.Filter] | None = None) -> None:
    """Configure root logging with a compact, copy-friendly format."""

    handler = logging.StreamHandler()
    handler.setFormatter(
        StructuredFormatter(
            fmt="%(asctime)s | %(levelname)s | %(component)s | %(message)s%(structured_context)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    filters: list[logging.Filter] = [_StripAnsiFilter(), _ContextAugmentFilter()]
    if extra_filters:
        filters.extend(extra_filters)
    for filt in filters:
        handler.addFilter(filt)

    logging.basicConfig(level=_determine_level(), handlers=[handler], force=True)

    for noisy_logger in ("google", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
