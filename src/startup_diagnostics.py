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


def _probe_ctranslate_environment() -> DiagnosticResult:
    try:
        ct_spec = importlib.util.find_spec("ctranslate2")
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                "Failed to query ctranslate2 module availability.",
                event="diagnostics.ctranslate2.spec_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "Unable to inspect CTranslate2 installation.",
            "details": {"error": str(exc)},
            "suggestion": "Ensure CTranslate2 is installed and accessible to the interpreter.",
            "fatal": False,
        }
        return _result_from_payload("CTranslate2", payload, success_status="warning")

    if ct_spec is None:
        payload = {
            "ok": False,
            "message": "CTranslate2 is not installed; transcription backend cannot initialize.",
            "details": {},
            "suggestion": "Install the 'ctranslate2' package to enable the faster-whisper backend.",
            "fatal": True,
        }
        return _result_from_payload("CTranslate2", payload, success_status="error")

    try:
        import ctranslate2  # type: ignore
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                "Failed to import ctranslate2 module during diagnostics.",
                event="diagnostics.ctranslate2.import_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "CTranslate2 could not be imported.",
            "details": {"error": str(exc)},
            "suggestion": "Reinstall CTranslate2 and verify compatibility with the current Python version.",
            "fatal": True,
        }
        return _result_from_payload("CTranslate2", payload, success_status="error")

    version = getattr(ctranslate2, "__version__", "unknown")

    def _supported(device: str) -> list[str]:
        try:
            return list(ctranslate2.get_supported_compute_types(device))
        except Exception as exc:  # pragma: no cover - diagnostics best-effort
            LOGGER.debug(
                "Failed to query compute types for device '%s': %s",
                device,
                exc,
                exc_info=True,
            )
            return []

    cpu_types = _supported("cpu")
    cuda_types = _supported("cuda")

    details = {
        "version": version,
        "cpu_compute_types": cpu_types,
        "cuda_compute_types": cuda_types,
    }

    if cuda_types:
        payload = {
            "ok": True,
            "message": (
                f"CTranslate2 {version} detected with CUDA support ({', '.join(cuda_types)})."
            ),
            "details": details,
        }
        return _result_from_payload("CTranslate2", payload, success_status="ok")

    payload = {
        "ok": True,
        "message": (
            f"CTranslate2 {version} available for CPU execution. Install GPU-enabled binaries "
            "to enable CUDA compute types if desired."
        ),
        "details": details,
        "suggestion": "Refer to the CTranslate2 documentation for GPU installation instructions.",
        "fatal": False,
    }
    return _result_from_payload("CTranslate2", payload, success_status="warning")


def run_startup_diagnostics(
    config_manager: Any,
    *,
    hotkey_config_path: Path | str | None = None,
) -> StartupDiagnosticsReport:
    """Run all startup diagnostics and persist the report in the config manager."""

    report = StartupDiagnosticsReport()
    report.add(_probe_audio_subsystem())
    report.add(_probe_hotkey_subsystem(hotkey_config_path))
    report.add(_probe_ctranslate_environment())

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
