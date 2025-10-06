"""Logging helpers that keep terminal output structured and copy-friendly."""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
import contextvars
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

LOG_DIR_ENV = "WHISPER_LOG_DIR"
LOG_MAX_BYTES_ENV = "WHISPER_LOG_MAX_BYTES"
LOG_BACKUP_COUNT_ENV = "WHISPER_LOG_BACKUP_COUNT"
LOG_RUN_ID_ENV = "WHISPER_LOG_RUN_ID"
LOG_FORMAT_ENV = "WHISPER_LOG_FORMAT"
DEFAULT_LOG_FILENAME = "whisper-flash-transcriber.log"
DEFAULT_LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
DEFAULT_LOG_BACKUP_COUNT = 5

_FORMAT_STRUCTURED = "structured"
_FORMAT_JSON = "json"
_SUPPORTED_FORMATS = {_FORMAT_STRUCTURED, _FORMAT_JSON}

_SESSION_START = datetime.now(timezone.utc)
_SESSION_MONOTONIC = time.monotonic()
_RUN_ID = (
    os.getenv(LOG_RUN_ID_ENV, "").strip()
    or f"{_SESSION_START:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
)
_LAST_LOG_DIRECTORY: Path | None = None
_LAST_LOG_FILE: Path | None = None
_ACTIVE_FORMAT: str = _FORMAT_STRUCTURED


_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "whisper_flash_correlation_id",
    default=None,
)
_operation_stack_var: contextvars.ContextVar[tuple[tuple[str, str | None], ...]] = (
    contextvars.ContextVar("whisper_flash_operation_stack", default=())
)


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
        record.run_id = _RUN_ID
        try:
            record.uptime_ms = int((time.monotonic() - _SESSION_MONOTONIC) * 1000)
        except Exception:  # pragma: no cover - defensive guard
            record.uptime_ms = None
        return True


class _CorrelationContextFilter(logging.Filter):
    """Attach correlation metadata stored in context variables."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - trivial
        correlation_id = _correlation_id_var.get()
        if correlation_id is not None and not getattr(record, "correlation_id", None):
            record.correlation_id = correlation_id

        operation_stack = _operation_stack_var.get()
        if operation_stack:
            operation_id, operation_name = operation_stack[-1]
            record.operation_id = getattr(record, "operation_id", operation_id)
            if operation_name is not None and not getattr(record, "operation_name", None):
                record.operation_name = operation_name
            record.operation_depth = getattr(record, "operation_depth", len(operation_stack))
        else:
            record.operation_depth = getattr(record, "operation_depth", 0)
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


def current_correlation_id() -> str | None:
    """Return the correlation identifier active in the current context."""

    return _correlation_id_var.get()


@contextmanager
def scoped_correlation_id(
    value: str | None = None,
    *,
    preserve_existing: bool = False,
) -> Iterator[str | None]:
    """Temporarily bind ``value`` as the active correlation identifier."""

    current_value = _correlation_id_var.get()
    if preserve_existing and current_value is not None:
        yield current_value
        return

    new_value = value or uuid.uuid4().hex[:8]
    token = _correlation_id_var.set(new_value)
    try:
        yield new_value
    finally:
        _correlation_id_var.reset(token)


@contextmanager
def log_operation(
    logger: logging.Logger | logging.LoggerAdapter,
    headline: str,
    /,
    *,
    event: str | None = None,
    details: Mapping[str, Any] | None = None,
    start_level: int = logging.INFO,
    success_level: int | None = None,
    failure_level: int | None = None,
    success_message: str | None = None,
    failure_message: str | None = None,
    success_details: Mapping[str, Any] | None = None,
    failure_details: Mapping[str, Any] | None = None,
    include_duration: bool = True,
) -> Iterator[str]:
    """Log the lifespan of a structured operation with automatic timing."""

    operation_id = uuid.uuid4().hex[:8]
    operation_name = event if event else headline
    stack = _operation_stack_var.get()
    token = _operation_stack_var.set(stack + ((operation_id, operation_name),))

    start_event = f"{event}.start" if event else None
    success_event = f"{event}.success" if event else None
    failure_event = f"{event}.error" if event else None

    start_payload = dict(details or {})
    start_payload.setdefault("operation_id", operation_id)

    logger.log(start_level, StructuredMessage(headline, event=start_event, details=start_payload))

    started_at = time.perf_counter()

    try:
        yield operation_id
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        detail_payload = dict(details or {})
        if failure_details:
            detail_payload.update(failure_details)
        detail_payload.setdefault("operation_id", operation_id)
        if include_duration:
            detail_payload.setdefault("duration_ms", round(duration_ms, 2))

        logger.log(
            failure_level or logging.ERROR,
            StructuredMessage(
                failure_message or f"{headline} failed.",
                event=failure_event,
                details=detail_payload,
            ),
        )
        raise
    else:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        detail_payload = dict(details or {})
        if success_details:
            detail_payload.update(success_details)
        detail_payload.setdefault("operation_id", operation_id)
        if include_duration:
            detail_payload.setdefault("duration_ms", round(duration_ms, 2))

        logger.log(
            success_level or start_level,
            StructuredMessage(
                success_message or f"{headline} completed.",
                event=success_event,
                details=detail_payload,
            ),
        )
    finally:
        _operation_stack_var.reset(token)


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


def _determine_log_format() -> tuple[str, str | None]:
    """Return the desired log format and any warning about invalid choices."""

    raw_value = (os.getenv(LOG_FORMAT_ENV) or "").strip().lower()
    if not raw_value:
        return _FORMAT_STRUCTURED, None
    if raw_value in _SUPPORTED_FORMATS:
        return raw_value, None
    return _FORMAT_STRUCTURED, raw_value


def _build_formatter(log_format: str) -> logging.Formatter:
    if log_format == _FORMAT_JSON:
        return _JsonLogFormatter()
    return _StructuredLogFormatter()


class _StructuredLogFormatter(logging.Formatter):
    """Formatter that produces explicit, copy-friendly log lines."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname.upper()
        component = getattr(record, "component", None)
        name = record.name
        if component:
            name = f"{name}:{component}"

        message = super().format(record)
        if "\n" in message:
            head, *rest = message.splitlines()
            body = "\n".join(f"    {line}" for line in rest)
            message = head + ("\n" + body if body else "")

        extras: list[str] = []
        process_id = getattr(record, "process_id", None)
        if process_id is not None:
            extras.append(f"pid={process_id}")
        thread_name = getattr(record, "thread_name", None) or getattr(record, "threadName", None)
        if thread_name and thread_name != "MainThread":
            extras.append(f"thread={thread_name}")
        run_id = getattr(record, "run_id", None)
        if run_id:
            extras.append(f"run={run_id}")
        uptime_ms = getattr(record, "uptime_ms", None)
        if isinstance(uptime_ms, int) and uptime_ms >= 0:
            extras.append(f"uptime_ms={uptime_ms}")
        if record.funcName:
            extras.append(f"func={record.funcName}")

        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            extras.append(f"corr={correlation_id}")
        operation_id = getattr(record, "operation_id", None)
        if operation_id:
            extras.append(f"op={operation_id}")
            depth = getattr(record, "operation_depth", None)
            if isinstance(depth, int) and depth > 1:
                extras.append(f"depth={depth}")
        operation_name = getattr(record, "operation_name", None)
        if operation_name:
            extras.append(f"op_name={operation_name}")

        context = f" [{', '.join(extras)}]" if extras else ""

        event = getattr(record, "event", None)
        details = getattr(record, "details", None)
        detail_segment = ""
        if event:
            detail_segment = f" | event={event}"

        if isinstance(details, Mapping) and details:
            detail_pairs = " ".join(
                f"{key}={_stringify_detail(value)}" for key, value in sorted(details.items())
            )
            if detail_pairs:
                detail_segment += f" | {detail_pairs}"

        return f"{timestamp} | {level:<8} | {name}{context} | {message}{detail_segment}"


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_json_value(item) for item in value]
    return str(value)


class _JsonLogFormatter(logging.Formatter):
    """Formatter that encodes log records as JSON payloads."""

    default_time_format = "%Y-%m-%dT%H:%M:%S"
    default_msec_format = "%s.%03d"

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname.upper(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        component = getattr(record, "component", None)
        if component:
            payload["component"] = component

        event = getattr(record, "event", None)
        if event:
            payload["event"] = event

        details = getattr(record, "details", None)
        if isinstance(details, Mapping) and details:
            payload["details"] = {
                str(key): _normalize_json_value(value) for key, value in details.items()
            }

        extras: dict[str, Any] = {}
        process_id = getattr(record, "process_id", None)
        if process_id is not None:
            extras["pid"] = process_id
        thread_name = getattr(record, "thread_name", None) or getattr(record, "threadName", None)
        if thread_name and thread_name != "MainThread":
            extras["thread"] = thread_name
        run_id = getattr(record, "run_id", None)
        if run_id:
            extras["run"] = run_id
        uptime_ms = getattr(record, "uptime_ms", None)
        if isinstance(uptime_ms, int) and uptime_ms >= 0:
            extras["uptime_ms"] = uptime_ms
        if record.funcName:
            extras["func"] = record.funcName
        if record.module:
            extras["module"] = record.module
        if record.filename:
            extras["filename"] = record.filename
        if record.lineno:
            extras["lineno"] = record.lineno

        correlation_id = getattr(record, "correlation_id", None)
        if correlation_id:
            extras["correlation_id"] = correlation_id
        operation_id = getattr(record, "operation_id", None)
        if operation_id:
            extras["operation_id"] = operation_id
            depth = getattr(record, "operation_depth", None)
            if isinstance(depth, int) and depth > 0:
                extras["operation_depth"] = depth
        operation_name = getattr(record, "operation_name", None)
        if operation_name:
            extras["operation_name"] = operation_name

        if extras:
            payload["extra"] = extras

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class ContextualLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that normalizes structured metadata across log records."""

    __slots__ = ("_component", "_defaults", "_default_event")

    def __init__(
        self,
        logger: logging.Logger,
        *,
        component: str | None = None,
        defaults: Mapping[str, Any] | None = None,
        default_event: str | None = None,
    ) -> None:
        super().__init__(logger, extra={})
        self._component = component
        self._defaults = dict(defaults or {})
        self._default_event = default_event

    def bind(self, *, event: str | None = None, **fields: Any) -> "ContextualLoggerAdapter":
        merged_defaults = dict(self._defaults)
        merged_defaults.update(fields)
        default_event = event or self._default_event
        return ContextualLoggerAdapter(
            self.logger,
            component=self._component,
            defaults=merged_defaults,
            default_event=default_event,
        )

    def process(self, msg: Any, kwargs: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
        if not isinstance(kwargs, dict):
            kwargs = dict(kwargs)
        event_override = kwargs.pop("event", None)
        detail_overrides = kwargs.pop("details", None)

        if isinstance(msg, StructuredMessage):
            headline = msg.headline
            event = msg.event
            details = dict(msg.details)
        else:
            headline = str(msg)
            event = None
            details: dict[str, Any] = {}

        details.update(self._defaults)
        if isinstance(detail_overrides, Mapping):
            details.update(detail_overrides)

        event = event_override or event or self._default_event

        extra = dict(kwargs.get("extra", {}))
        if self._component and "component" not in extra:
            extra["component"] = self._component
        if event is not None:
            extra.setdefault("event", event)
        if details:
            existing_details = extra.get("details")
            if isinstance(existing_details, Mapping):
                merged_details = dict(existing_details)
                merged_details.update(details)
            else:
                merged_details = details
            extra["details"] = merged_details
        kwargs = dict(kwargs)
        if extra:
            kwargs["extra"] = extra

        if isinstance(msg, StructuredMessage):
            msg = headline

        return msg, kwargs


def _resolve_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _build_rotating_file_handler(
    filters: Iterable[logging.Filter],
    formatter: logging.Formatter,
) -> logging.Handler | None:
    log_directory = Path(os.getenv(LOG_DIR_ENV, "logs")).expanduser()
    try:
        log_directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        logging.getLogger(__name__).warning(
            "Unable to create log directory at %s; file logging disabled.",
            log_directory,
            exc_info=True,
        )
        return None

    global _LAST_LOG_DIRECTORY, _LAST_LOG_FILE
    _LAST_LOG_DIRECTORY = log_directory

    max_bytes = _resolve_int(os.getenv(LOG_MAX_BYTES_ENV), DEFAULT_LOG_MAX_BYTES)
    backup_count = _resolve_int(os.getenv(LOG_BACKUP_COUNT_ENV), DEFAULT_LOG_BACKUP_COUNT)

    try:
        handler = RotatingFileHandler(
            log_directory / DEFAULT_LOG_FILENAME,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "Unable to configure rotating file handler in %s; file logging disabled.",
            log_directory,
            exc_info=True,
        )
        return None

    handler.setFormatter(formatter)
    for filt in filters:
        handler.addFilter(filt)
    try:
        _LAST_LOG_FILE = Path(handler.baseFilename).resolve()
    except Exception:
        _LAST_LOG_FILE = Path(handler.baseFilename)
    return handler


def setup_logging(*, extra_filters: Iterable[logging.Filter] | None = None) -> None:
    """Configure root logging with a structured, copy-friendly format."""

    level = _determine_level()
    log_format, invalid_choice = _determine_log_format()
    global _ACTIVE_FORMAT
    _ACTIVE_FORMAT = log_format

    filters: list[logging.Filter] = [
        _StripAnsiFilter(),
        _RuntimeContextFilter(),
        _CorrelationContextFilter(),
    ]
    if extra_filters:
        filters.extend(extra_filters)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(_build_formatter(log_format))
    for filt in filters:
        console_handler.addFilter(filt)

    handlers: list[logging.Handler] = [console_handler]

    file_formatter = _build_formatter(log_format)
    file_handler = _build_rotating_file_handler(filters, file_formatter)
    if file_handler is not None:
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.captureWarnings(True)

    if invalid_choice is not None:
        logging.getLogger(__name__).warning(
            "Unsupported log format '%s' requested via %s; using '%s' instead.",
            invalid_choice,
            LOG_FORMAT_ENV,
            log_format,
        )

    for noisy_logger in ("google", "httpx", "urllib3", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    get_logger("whisper_flash_transcriber.logging", component="Logging").info(
        log_context(
            "Logging configured.",
            event="logging.configured",
            level=logging.getLevelName(level),
            run_id=_RUN_ID,
            session_started=_SESSION_START.isoformat(),
            log_dir=str(_LAST_LOG_DIRECTORY) if _LAST_LOG_DIRECTORY else None,
            log_file=str(_LAST_LOG_FILE) if _LAST_LOG_FILE else None,
            log_format=log_format,
        )
    )

    emit_startup_banner()


def get_run_id() -> str:
    """Return the identifier assigned to the current logging session."""

    return _RUN_ID


def get_log_format() -> str:
    """Return the active log output format."""

    return _ACTIVE_FORMAT


def get_log_paths() -> tuple[Path | None, Path | None]:
    """Return the directory and file currently used for file logging."""

    return _LAST_LOG_DIRECTORY, _LAST_LOG_FILE


def emit_startup_banner(
    *,
    logger: logging.Logger | ContextualLoggerAdapter | None = None,
    extra_details: Mapping[str, Any] | None = None,
) -> None:
    """Emit a structured log entry summarizing runtime diagnostics."""

    target = logger or get_logger("whisper_flash_transcriber.logging", component="Logging")
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "run_id": _RUN_ID,
        "session_started": _SESSION_START.isoformat(),
        "log_dir": str(_LAST_LOG_DIRECTORY) if _LAST_LOG_DIRECTORY else str(Path(os.getenv(LOG_DIR_ENV, "logs")).expanduser()),
        "log_file": str(_LAST_LOG_FILE) if _LAST_LOG_FILE else None,
        "log_format": _ACTIVE_FORMAT,
    }
    if extra_details:
        details.update(extra_details)

    target.info(
        log_context(
            "Runtime context captured for logging session.",
            event="logging.runtime_context",
            details=details,
        )
    )


def log_context(
    headline: str,
    /,
    *,
    event: str | None = None,
    details: Mapping[str, Any] | None = None,
    **fields: Any,
) -> StructuredMessage:
    """Build a :class:`StructuredMessage` with a friendly helper syntax."""

    return StructuredMessage(headline, event=event, details=details, **fields)


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


@contextmanager
def log_duration(
    logger: logging.Logger,
    headline: str,
    /,
    *,
    event: str | None = None,
    details: Mapping[str, Any] | None = None,
    level: int = logging.INFO,
    success_level: int | None = None,
    failure_level: int | None = None,
    log_start: bool = False,
    start_level: int | None = None,
) -> Iterable[dict[str, Any]]:
    """Context manager that logs the duration of an operation.

    The context yields a mutable ``dict`` that can be populated with additional
    metadata. Its content is merged with ``details`` when the operation
    completes or fails. On failure, ``exc_info`` is automatically attached to
    the emitted log record.
    """

    base_details = dict(details or {})
    if log_start:
        start_details = dict(base_details)
        start_details.setdefault("status", "started")
        logger.log(
            start_level or level,
            StructuredMessage(
                headline,
                event=event,
                details=start_details,
            ),
        )

    collected: dict[str, Any] = {}
    start_time = time.perf_counter()
    try:
        yield collected
    except Exception as exc:
        failure_details = dict(base_details)
        failure_details.update(collected)
        failure_details.setdefault("status", "failure")
        failure_details["duration_ms"] = int((time.perf_counter() - start_time) * 1000)
        failure_details.setdefault("error", repr(exc))
        logger.log(
            failure_level or logging.ERROR,
            StructuredMessage(
                headline,
                event=event,
                details=failure_details,
            ),
            exc_info=True,
        )
        raise
    else:
        success_details = dict(base_details)
        success_details.update(collected)
        success_details.setdefault("status", "success")
        success_details["duration_ms"] = int((time.perf_counter() - start_time) * 1000)
        logger.log(
            success_level or level,
            StructuredMessage(
                headline,
                event=event,
                details=success_details,
            ),
        )


def get_logger(
    name: str,
    *,
    component: str | None = None,
    default_event: str | None = None,
    **default_fields: Any,
) -> ContextualLoggerAdapter:
    """Return a logger adapter enriched with component metadata."""

    base_logger = logging.getLogger(name)
    return ContextualLoggerAdapter(
        base_logger,
        component=component,
        defaults=default_fields,
        default_event=default_event,
    )


__all__ = [
    "ContextualLoggerAdapter",
    "StructuredMessage",
    "current_correlation_id",
    "get_log_format",
    "get_log_paths",
    "get_logger",
    "log_duration",
    "log_context",
    "log_event",
    "log_operation",
    "scoped_correlation_id",
    "setup_logging",
]
