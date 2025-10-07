from __future__ import annotations

import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio_handler import AudioHandler
from .keyboard_hotkey_manager import KeyboardHotkeyManager
from .logging_utils import StructuredMessage, get_logger

LOGGER = get_logger("whisper_flash_transcriber.diagnostics", component="Bootstrap")


@dataclass
class DiagnosticResult:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None
    fatal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "fatal": self.fatal,
        }


@dataclass
class StartupDiagnosticsReport:
    checks: list[DiagnosticResult] = field(default_factory=list)

    def add(self, result: DiagnosticResult) -> None:
        self.checks.append(result)

    @property
    def has_errors(self) -> bool:
        return any(result.status == "error" for result in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(result.status == "warning" for result in self.checks)

    @property
    def has_fatal_errors(self) -> bool:
        return any(result.fatal for result in self.checks)

    def user_friendly_summary(self, *, include_success: bool = False) -> list[str]:
        lines: list[str] = []
        for result in self.checks:
            if not include_success and result.status == "ok":
                continue
            summary = f"{result.name}: {result.message}"
            if result.suggestion:
                summary = f"{summary}\n• Suggestion: {result.suggestion}"
            lines.append(summary)
        if not lines:
            lines.append("All diagnostics completed successfully.")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [result.to_dict() for result in self.checks],
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "has_fatal_errors": self.has_fatal_errors,
        }


def _log_result(result: DiagnosticResult) -> None:
    level = logging.INFO
    if result.status == "warning":
        level = logging.WARNING
    elif result.status == "error":
        level = logging.ERROR
    LOGGER.log(
        level,
        StructuredMessage(
            "Startup diagnostic completed.",
            event="diagnostics.result",
            check=result.name,
            status=result.status,
            fatal=result.fatal,
            suggestion=bool(result.suggestion),
            details=result.details or None,
        ),
    )


def _result_from_payload(name: str, payload: dict[str, Any], *, success_status: str = "ok") -> DiagnosticResult:
    ok = bool(payload.get("ok"))
    status = success_status if ok else "error"
    message = payload.get("message", "")
    suggestion = payload.get("suggestion")
    fatal = bool(payload.get("fatal"))
    details = {
        key: value for key, value in payload.items() if key not in {"ok", "message", "suggestion", "fatal"}
    }
    result = DiagnosticResult(
        name=name,
        status=status,
        message=message,
        suggestion=suggestion,
        fatal=fatal,
        details=details,
    )
    _log_result(result)
    return result


def _probe_audio_subsystem() -> DiagnosticResult:
    payload = AudioHandler.probe_default_device()
    return _result_from_payload("Audio Input", payload)


def _probe_hotkey_subsystem(hotkey_config_path: Path | str | None) -> DiagnosticResult:
    path_str = str(hotkey_config_path) if hotkey_config_path is not None else None
    try:
        manager = KeyboardHotkeyManager(config_file=path_str) if path_str else KeyboardHotkeyManager()
        payload = manager.dry_run_register()
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                "Hotkey diagnostics failed during manager instantiation.",
                event="diagnostics.hotkeys.initialization_failed",
                path=path_str,
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "Keyboard hotkey manager could not be initialized.",
            "details": {"error": str(exc), "path": path_str},
            "suggestion": "Verify keyboard library installation and permissions.",
            "fatal": True,
        }
    return _result_from_payload("Hotkey Permissions", payload)


def _probe_ctranslate2_environment() -> DiagnosticResult:
    try:
        ct2_spec = importlib.util.find_spec("ctranslate2")
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                "Failed to query ctranslate2 module availability.",
                event="diagnostics.ct2.spec_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "Unable to inspect the CTranslate2 runtime.",
            "details": {"error": str(exc)},
            "suggestion": "Ensure CTranslate2 is installed to enable model execution.",
            "fatal": False,
        }
        return _result_from_payload("CTranslate2", payload, success_status="warning")

    if ct2_spec is None:
        payload = {
            "ok": False,
            "message": "CTranslate2 is not installed; transcription cannot start.",
            "details": {},
            "suggestion": "Install the `ctranslate2` package (GPU-enabled wheels are recommended when available).",
            "fatal": True,
        }
        return _result_from_payload("CTranslate2", payload, success_status="error")

    try:
        ct2_module = importlib.import_module("ctranslate2")
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.error(
            StructuredMessage(
                "Failed to import ctranslate2 during diagnostics.",
                event="diagnostics.ct2.import_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "CTranslate2 could not be imported.",
            "details": {"error": str(exc)},
            "suggestion": "Reinstall CTranslate2 and verify wheel compatibility with the current interpreter.",
            "fatal": True,
        }
        return _result_from_payload("CTranslate2", payload, success_status="error")

    version = getattr(ct2_module, "__version__", "unknown")
    cuda_devices: int | None = None
    cuda_error: str | None = None
    compute_types: dict[str, list[str]] = {}

    get_device_count = getattr(ct2_module, "get_device_count", None)
    if callable(get_device_count):
        try:
            cuda_devices = int(get_device_count("cuda"))
        except Exception as exc:
            cuda_error = str(exc)
            LOGGER.debug("ctranslate2.get_device_count('cuda') failed.", exc_info=True)

    get_supported_types = getattr(ct2_module, "get_supported_compute_types", None)
    if callable(get_supported_types):
        try:
            compute_types["cuda"] = list(get_supported_types("cuda"))
        except Exception:
            LOGGER.debug("Unable to query CUDA compute types from ctranslate2.", exc_info=True)
        try:
            compute_types["cpu"] = list(get_supported_types("cpu"))
        except Exception:
            LOGGER.debug("Unable to query CPU compute types from ctranslate2.", exc_info=True)

    payload = {
        "ok": True,
        "message": "CTranslate2 runtime detected.",
        "details": {
            "version": version,
            "cuda_device_count": cuda_devices,
            "compute_types": compute_types,
            "cuda_error": cuda_error,
        },
        "suggestion": "Install the GPU-enabled wheels if CUDA acceleration is desired." if not cuda_devices else "",
        "fatal": False,
    }
    return _result_from_payload("CTranslate2", payload)


def run_startup_diagnostics(
    config_manager: Any,
    *,
    hotkey_config_path: Path | str | None = None,
) -> StartupDiagnosticsReport:
    """Run all startup diagnostics and persist the report in the config manager."""

    report = StartupDiagnosticsReport()
    report.add(_probe_audio_subsystem())
    report.add(_probe_hotkey_subsystem(hotkey_config_path))
    report.add(_probe_ctranslate2_environment())

    try:
        register = getattr(config_manager, "register_startup_diagnostics", None)
        if callable(register):
            register(report)
        else:
            setattr(config_manager, "startup_diagnostics_report", report)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning(
            StructuredMessage(
                "Failed to attach diagnostics report to config manager.",
                event="diagnostics.config_manager_attach_failed",
                error=str(exc),
            ),
            exc_info=True,
        )

    LOGGER.info(
        StructuredMessage(
            "Startup diagnostics completed.",
            event="diagnostics.completed",
            has_errors=report.has_errors,
            has_warnings=report.has_warnings,
            has_fatal_errors=report.has_fatal_errors,
        )
    )
    return report


def format_report_for_console(report: StartupDiagnosticsReport) -> str:
    """Return a human-readable representation of the diagnostics report."""

    lines: list[str] = ["=== Whisper Flash Transcriber Diagnostics ==="]
    for result in report.checks:
        status = result.status.upper()
        lines.append(f"[{status}] {result.name}: {result.message}")
        if result.suggestion:
            lines.append(f"    Suggestion: {result.suggestion}")
        if result.details:
            lines.append(f"    Details: {json.dumps(result.details, indent=2)}")
    lines.append("")
    lines.append("JSON report:")
    lines.append(json.dumps(report.to_dict(), indent=2))
    return "\n".join(lines)
