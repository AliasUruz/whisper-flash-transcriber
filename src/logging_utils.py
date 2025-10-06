"""Logging helpers that keep terminal output structured and copy-friendly."""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Iterable, Mapping

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


class _StructuredLogFormatter(logging.Formatter):
    """Formatter that produces explicit, copy-friendly log lines."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname.upper()
        name = record.name
        message = super().format(record)

        extras: list[str] = []
        if record.processName:
            extras.append(f"process={record.processName}")
        if record.threadName and record.threadName != "MainThread":
            extras.append(f"thread={record.threadName}")
        if record.funcName:
            extras.append(f"func={record.funcName}")

        context = f" [{', '.join(extras)}]" if extras else ""

        if "\n" in message:
            head, *rest = message.splitlines()
            body = "\n".join(f"    {line}" for line in rest)
            message = head + ("\n" + body if body else "")

        return f"{timestamp} | {level:<8} | {name}{context} | {message}"


def setup_logging(*, extra_filters: Iterable[logging.Filter] | None = None) -> None:
    """Configure root logging with a structured, copy-friendly format."""

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt=(
                "%(asctime)s | %(levelname)-8s | %(name)s | "
                "pid=%(process_id)s thread=%(thread_name)s | %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    filters: list[logging.Filter] = [_StripAnsiFilter(), _RuntimeContextFilter()]
    if extra_filters:
        filters.extend(extra_filters)
    for filt in filters:
        handler.addFilter(filt)

    logging.basicConfig(level=_determine_level(), handlers=[handler], force=True)
    logging.captureWarnings(True)

    for noisy_logger in ("google", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger,
    level: int,
    headline: str,
    /,
    *,
    event: str | None = None,
    details: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Emit a :class:`StructuredMessage` through ``logger``."""

    logger.log(level, StructuredMessage(headline, event=event, details=details, **fields))


__all__ = ["StructuredMessage", "log_event", "setup_logging"]
