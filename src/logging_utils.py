"""Logging helpers with structured, copy-friendly terminal output."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, MutableMapping

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


class _RuntimeContextFilter(logging.Filter):
    """Inject lightweight runtime metadata into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        record.process_id = os.getpid()
        record.thread_name = threading.current_thread().name
        return True


def _stringify_detail(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.4f}"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, set, frozenset)):
        inner = ', '.join(_stringify_detail(item) for item in value)
        return f"[{inner}]"
    if value is None:
        return "<none>"
    return str(value)


class StructuredMessage:
    """Represent a log message with a concise headline and structured details."""

    __slots__ = ("headline", "event", "details")

    def __init__(
        self,
        headline: str,
        /,
        *,
        event: str | None = None,
        details: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> None:
        combined_details: dict[str, Any] = {}
        if details:
            combined_details.update(details)
        for key, value in fields.items():
            if value is not None:
                combined_details[key] = value
        self.headline = headline
        self.event = event
        self.details = combined_details

    def __str__(self) -> str:  # pragma: no cover - string formatting helper
        segments = [self.headline]
        if self.event:
            segments.append(f"event={self.event}")
        if self.details:
            detail_pairs = " ".join(
                f"{key}={_stringify_detail(value)}" for key, value in self.details.items()
            )
            if detail_pairs:
                segments.append(detail_pairs)
        return " | ".join(segments)


def _determine_level() -> int:
    env_level = os.getenv("WHISPER_LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


@dataclass(slots=True, frozen=True)
class StructuredMessage:
    """Container for log messages enriched with contextual metadata."""

    message: str
    fields: Mapping[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """Formatter that appends structured context to the base log message."""

    _ORDERED_FIELD_NAMES = (
        "component",
        "event",
        "stage",
        "action",
        "status",
        "details",
    )

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - cosmetic
        base_message = super().format(record)
        structured_fields: Mapping[str, Any] | None = getattr(record, "_structured_fields", None)

        merged_fields: list[tuple[str, Any]] = []
        seen: set[str] = set()

        for field_name in self._ORDERED_FIELD_NAMES:
            if field_name in seen:
                continue
            if hasattr(record, field_name):
                value = getattr(record, field_name)
            else:
                value = None
            if value not in (None, ""):
                merged_fields.append((field_name, value))
                seen.add(field_name)

        if structured_fields:
            for field_name in sorted(structured_fields):
                if field_name in seen:
                    continue
                value = structured_fields[field_name]
                if value in (None, ""):
                    continue
                merged_fields.append((field_name, value))

        if not merged_fields:
            return base_message

        formatted_fields = " ".join(
            f"{key}={self._stringify(value)}" for key, value in merged_fields
        )
        return f"{base_message} | {formatted_fields}"

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".") or "0"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that preserves structured metadata across log calls."""

    def bind(self, **extra_fields: Any) -> "StructuredLoggerAdapter":
        merged_extra: dict[str, Any] = {**self.extra, **extra_fields}
        return self.__class__(self.logger, merged_extra)

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]):  # pragma: no cover - trivial
        log_kwargs = dict(kwargs)
        extra: dict[str, Any]
        if "extra" in log_kwargs:
            extra = dict(log_kwargs["extra"])
        else:
            extra = {}

        for key, value in self.extra.items():
            extra.setdefault(key, value)

        if isinstance(msg, StructuredMessage):
            structured_fields = dict(msg.fields)
            existing = extra.get("_structured_fields")
            if isinstance(existing, dict):
                structured_fields = {**existing, **structured_fields}
            extra["_structured_fields"] = structured_fields

            for key in StructuredFormatter._ORDERED_FIELD_NAMES:
                if key in structured_fields:
                    extra.setdefault(key, structured_fields[key])

            processed_message = msg.message
        else:
            processed_message = msg

        log_kwargs["extra"] = extra
        return processed_message, log_kwargs


def log_context(message: str, /, **fields: Any) -> StructuredMessage:
    """Create a structured log payload with contextual metadata."""

    return StructuredMessage(message=message, fields=fields)


def get_logger(name: str, *, component: str | None = None, **context: Any) -> StructuredLoggerAdapter:
    """Return a structured logger adapter preloaded with contextual fields."""

    base_logger = logging.getLogger(name)
    if component is not None:
        context.setdefault("component", component)
    return StructuredLoggerAdapter(base_logger, context)


def setup_logging(*, extra_filters: Iterable[logging.Filter] | None = None) -> None:
    """Configure root logging with a structured, copy-friendly format."""

    handler = logging.StreamHandler()
    handler.setFormatter(
        StructuredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | thread=%(threadName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    filters: list[logging.Filter] = [_StripAnsiFilter(), _RuntimeContextFilter()]
    if extra_filters:
        filters.extend(extra_filters)
    for filt in filters:
        handler.addFilter(filt)

    logging.basicConfig(level=_determine_level(), handlers=[handler], force=True)

    for noisy_logger in ("google", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


__all__ = [
    "StructuredLoggerAdapter",
    "StructuredMessage",
    "get_logger",
    "log_context",
    "setup_logging",
]
