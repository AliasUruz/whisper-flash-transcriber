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


def _probe_torch_environment() -> DiagnosticResult:
    try:
        torch_spec = importlib.util.find_spec("torch")
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                "Failed to query torch module availability.",
                event="diagnostics.torch.spec_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "Unable to inspect PyTorch installation.",
            "details": {"error": str(exc)},
            "suggestion": "Ensure PyTorch is installed and accessible to the interpreter.",
            "fatal": False,
        }
        return _result_from_payload("PyTorch", payload, success_status="warning")

    if torch_spec is None:
        payload = {
            "ok": False,
            "message": "PyTorch is not installed; GPU acceleration is unavailable.",
            "details": {},
            "suggestion": (
                "Install PyTorch with GPU support using the official wheel index if acceleration is required."
            ),
            "fatal": False,
        }
        return _result_from_payload("PyTorch", payload, success_status="warning")

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.error(
            StructuredMessage(
                "Failed to import torch module during diagnostics.",
                event="diagnostics.torch.import_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        payload = {
            "ok": False,
            "message": "PyTorch could not be imported.",
            "details": {"error": str(exc)},
            "suggestion": "Reinstall PyTorch and verify compatibility with the current Python version.",
            "fatal": False,
        }
        return _result_from_payload("PyTorch", payload, success_status="warning")

    version = getattr(torch, "__version__", "unknown")
    cuda_available = False
    cuda_devices: list[dict[str, Any]] = []
    cuda_error: str | None = None
    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_count = torch.cuda.device_count()
            for index in range(device_count):
                try:
                    name = torch.cuda.get_device_name(index)
                    props = torch.cuda.get_device_properties(index)
                    cuda_devices.append(
                        {
                            "index": index,
                            "name": name,
                            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )
                except Exception as gpu_exc:  # pragma: no cover - diagnostics best-effort
                    cuda_devices.append({"index": index, "error": str(gpu_exc)})
        else:
            device_count = 0
    except Exception as exc:  # pragma: no cover - diagnostics best-effort
        cuda_error = str(exc)
        device_count = 0
        LOGGER.warning(
            StructuredMessage(
                "CUDA diagnostics failed.",
                event="diagnostics.torch.cuda_failure",
                error=str(exc),
            ),
            exc_info=True,
        )

    details = {
        "version": version,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": cuda_devices,
    }
    if cuda_error:
        details["cuda_error"] = cuda_error

    if cuda_available and cuda_devices:
        message = f"PyTorch {version} detected with {len(cuda_devices)} CUDA device(s)."
        payload = {
            "ok": True,
            "message": message,
            "details": details,
            "suggestion": None,
            "fatal": False,
        }
        return _result_from_payload("PyTorch", payload, success_status="ok")

    if cuda_available and not cuda_devices:
        message = f"PyTorch {version} reports CUDA availability but device enumeration failed."
        payload = {
            "ok": False,
            "message": message,
            "details": details,
            "suggestion": "Check GPU drivers and ensure the user has permission to access the GPU device.",
            "fatal": False,
        }
        return _result_from_payload("PyTorch", payload, success_status="warning")

    message = f"PyTorch {version} loaded without CUDA support; CPU execution will be used."
    payload = {
        "ok": False,
        "message": message,
        "details": details,
        "suggestion": "Install a CUDA-enabled build of PyTorch if GPU acceleration is desired.",
        "fatal": False,
    }
    return _result_from_payload("PyTorch", payload, success_status="warning")


def run_startup_diagnostics(
    config_manager: Any,
    *,
    hotkey_config_path: Path | str | None = None,
) -> StartupDiagnosticsReport:
    """Run all startup diagnostics and persist the report in the config manager."""

    report = StartupDiagnosticsReport()
    report.add(_probe_audio_subsystem())
    report.add(_probe_hotkey_subsystem(hotkey_config_path))
    report.add(_probe_torch_environment())

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
