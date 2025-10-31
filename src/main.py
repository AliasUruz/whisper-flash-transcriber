import atexit
import argparse
import importlib.util
import json
import os
import shutil
import sys
import threading
import tkinter as tk
import tkinter.messagebox as messagebox
from pathlib import Path

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from typing import Mapping, cast

import src.config_manager as config_module
from src.model_manager import get_curated_entry, normalize_backend_label


ICON_PATH = os.path.join(PROJECT_ROOT, "icon.ico")
HOTKEY_CONFIG_PATH = Path(
    cast(str, getattr(config_module, "HOTKEY_CONFIG_FILE"))
).expanduser()
LEGACY_HOTKEY_CONFIG_PATHS = tuple(
    path.expanduser()
    for path in cast(
        tuple[Path, ...], getattr(config_module, "LEGACY_HOTKEY_LOCATIONS")
    )
)

from src.logging_utils import (
    StructuredMessage,
    emit_startup_banner,
    get_logger,
    install_exception_hooks,
    setup_logging,
)
LOGGER = get_logger("whisper_flash_transcriber.bootstrap", component="Bootstrap")


ENV_DEFAULTS = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "BITSANDBYTES_NOWELCOME": "1",
}


class HeadlessEventLoop:
    """Minimal event loop shim used when running without Tk."""

    def __init__(self, *, shutdown_event: threading.Event | None = None) -> None:
        self._shutdown_event = shutdown_event or threading.Event()
        self._timers: set[threading.Timer] = set()
        self._lock = threading.Lock()

    def after(self, delay_ms: int, callback, *args):  # type: ignore[override]
        delay_seconds = max(delay_ms, 0) / 1000

        def _invoke() -> None:
            try:
                callback(*args)
            finally:
                with self._lock:
                    self._timers.discard(timer)

        timer = threading.Timer(delay_seconds, _invoke)
        timer.daemon = True
        with self._lock:
            self._timers.add(timer)
        timer.start()
        return timer

    def after_cancel(self, timer: threading.Timer) -> None:
        timer.cancel()
        with self._lock:
            self._timers.discard(timer)

    def request_shutdown(self) -> None:
        self._shutdown_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._shutdown_event.wait(timeout)

    def close(self) -> None:
        self.request_shutdown()
        with self._lock:
            timers = tuple(self._timers)
            self._timers.clear()
        for timer in timers:
            timer.cancel()


def _patch_messagebox_for_headless() -> None:
    """Replace Tk messageboxes with log-based fallbacks for headless execution."""

    dialog_defaults: dict[str, object] = {
        "showinfo": "ok",
        "showwarning": "ok",
        "showerror": "ok",
        "askyesno": False,
        "askokcancel": False,
        "askretrycancel": False,
        "askyesnocancel": False,
        "askquestion": "no",
    }

    for name, default in dialog_defaults.items():
        original = getattr(messagebox, name, None)
        if original is None:
            continue

        def _make_stub(dialog_name: str, fallback: object):
            def _stub(*args, **kwargs):
                title = kwargs.get("title")
                message = kwargs.get("message")
                if args:
                    title = title or args[0]
                if len(args) > 1:
                    message = message or args[1]
                LOGGER.warning(
                    StructuredMessage(
                        "Suppressed Tkinter messagebox while running headless.",
                        event="headless.messagebox",
                        dialog=dialog_name,
                        title=title,
                        message=message,
                    )
                )
                return fallback

            return _stub

        setattr(messagebox, name, _make_stub(name, default))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper Flash Transcriber bootstrap entry point.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run startup diagnostics and exit without launching the UI.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without initializing the Tk UI or system tray icon.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip preflight checks and startup diagnostics (useful for troubleshooting).",
    )
    return parser.parse_args(argv)


def ensure_display_available() -> None:
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        LOGGER.warning(
            StructuredMessage(
                "Display server unavailable; aborting GUI bootstrap.",
                event="startup.display_check",
                platform=os.name,
                reason="missing_display_variable",
            )
        )
        sys.exit(0)


def configure_environment() -> None:
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", os.environ.get("TRANSFORMERS_VERBOSITY", "error"))
    LOGGER.debug(
        StructuredMessage(
            "Base environment variables applied.",
            event="startup.environment_applied",
            details={"defaults": tuple(sorted(ENV_DEFAULTS))},
        )
    )


def configure_cuda_logging() -> None:
    try:
        import ctranslate2  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        LOGGER.debug(
            StructuredMessage(
                "CTranslate2 module not available; skipping CUDA diagnostics.",
                event="startup.cuda_skipped",
                reason="ctranslate2_missing",
            )
        )
        return
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            StructuredMessage(
                "Failed to import CTranslate2 for CUDA diagnostics.",
                event="cuda.diagnostics_exception",
                error=str(exc),
            ),
            exc_info=True,
        )
        return

    try:
        has_cuda = bool(getattr(ctranslate2, "has_cuda", lambda: False)())
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            StructuredMessage(
                "Unable to query CUDA support via CTranslate2.",
                event="cuda.diagnostics_failed",
                error=str(exc),
            ),
            exc_info=True,
        )
        has_cuda = False

    gpu_count = 0
    if has_cuda and hasattr(ctranslate2, "get_device_count"):
        try:
            gpu_count = int(ctranslate2.get_device_count("cuda"))  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning(
                StructuredMessage(
                    "Unable to enumerate CUDA devices via CTranslate2.",
                    event="cuda.gpu_introspection_failed",
                    error=str(exc),
                ),
                exc_info=True,
            )
            gpu_count = 0

    if not has_cuda or gpu_count <= 0:
        cpu_fallback_message = StructuredMessage(
            "CUDA runtime not detected via CTranslate2; defaulting to CPU execution.",
            event="cuda.cpu_fallback",
            detector="ctranslate2",
            has_cuda=has_cuda,
            gpu_count=gpu_count,
        )
        LOGGER.info(cpu_fallback_message)
        return

    compute_types: tuple[str, ...] | None = None
    if hasattr(ctranslate2, "get_supported_compute_types"):
        try:
            supported = ctranslate2.get_supported_compute_types("cuda")  # type: ignore[arg-type]
            compute_types = tuple(supported)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug(
                StructuredMessage(
                    "Unable to determine supported CUDA compute types via CTranslate2.",
                    event="cuda.compute_type_probe_failed",
                    error=str(exc),
                ),
                exc_info=True,
            )

    LOGGER.info(
        StructuredMessage(
            "CUDA runtime detected via CTranslate2.",
            event="cuda.runtime_detected",
            detector="ctranslate2",
            gpu_count=gpu_count,
            compute_types=compute_types,
        )
    )

    try:
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
    except Exception:
        has_flash_attn = False
    flash_attn_probe_message = StructuredMessage(
        "FlashAttention 2 availability checked.",
        event="cuda.flash_attention_probe",
        available=has_flash_attn,
    )
    LOGGER.info(flash_attn_probe_message)


def patch_tk_variable_cleanup() -> None:
    original_del = tk.Variable.__del__

    def _safe_variable_del(self):
        tk_root = getattr(self, "_tk", None)
        if not tk_root:
            return
        try:
            if tk_root.getboolean(tk_root.call("info", "exists", self._name)):
                tk_root.globalunsetvar(self._name)
        except Exception:
            pass
        commands = getattr(self, "_tclCommands", None)
        if commands:
            for name in commands:
                try:
                    tk_root.deletecommand(name)
                except Exception:
                    pass
            self._tclCommands = None
        try:
            original_del(self)
        except Exception:
            pass

    tk.Variable.__del__ = _safe_variable_del


def _ensure_hotkey_payload(data: Mapping[str, object]) -> dict[str, object]:
    defaults: dict[str, object] = {
        "record_key": "f3",
        "agent_key": "f4",
        "record_mode": "toggle",
    }
    updated: dict[str, object] = dict(defaults)
    updated.update({k: v for k, v in data.items() if k in defaults})
    return updated


def _maybe_migrate_hotkey_config(target: Path, candidates: tuple[Path, ...]) -> None:
    if target.exists():
        return
    for candidate in candidates:
        try:
            if candidate.resolve() == target.resolve():
                return
        except Exception:
            if str(candidate) == str(target):
                return
        if not candidate.exists():
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.warning(
                StructuredMessage(
                    "Unable to prepare destination for hotkey configuration.",
                    event="startup.hotkey_config_migration_failed",
                    destination=str(target),
                    error=str(exc),
                )
            )
            return
        try:
            shutil.move(str(candidate), str(target))
        except Exception as exc:
            LOGGER.warning(
                StructuredMessage(
                    "Failed to migrate legacy hotkey configuration.",
                    event="startup.hotkey_config_migration_failed",
                    source=str(candidate),
                    destination=str(target),
                    error=str(exc),
                )
            )
            return
        LOGGER.info(
            StructuredMessage(
                "Hotkey configuration migrated to profile directory.",
                event="startup.hotkey_config_migrated",
                source=str(candidate),
                destination=str(target),
            )
        )
        return


def _ensure_json_file(
    path: Path,
    payload: Mapping[str, object],
    *,
    description: str,
    recover_on_error: bool = True,
) -> bool:
    """Persist ``payload`` when ``path`` is missing; return True if file was created."""

    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(dict(payload), indent=4), encoding="utf-8")
            created = True
        else:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
    except json.JSONDecodeError as exc:
        if recover_on_error:
            LOGGER.warning(
                StructuredMessage(
                    f"{description.capitalize()} file corrupted; restoring defaults.",
                    event="startup.artifact_recreated",
                    path=str(path),
                    error=str(exc),
                )
            )
            path.write_text(json.dumps(dict(payload), indent=4), encoding="utf-8")
            return True
        LOGGER.error(
            StructuredMessage(
                f"Failed to validate {description} file.",
                event="startup.artifact_validation_failed",
                path=str(path),
                error=str(exc),
            ),
            exc_info=True,
        )
        raise
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                f"Failed to validate {description} file.",
                event="startup.artifact_validation_failed",
                path=str(path),
                error=str(exc),
            ),
            exc_info=True,
        )
        raise
    return created


def _ensure_hotkey_config(path: Path) -> bool:
    """Validate the hotkey configuration file, recreating it when needed."""

    defaults: dict[str, object] = {"record_key": "f3", "agent_key": "f4", "record_mode": "toggle"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(defaults, indent=4), encoding="utf-8")
            return True

        with path.open("r", encoding="utf-8") as handle:
            try:
                current = json.load(handle)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                LOGGER.warning(
                    StructuredMessage(
                        "Hotkey configuration corrupted; restoring defaults.",
                        event="startup.hotkey_config_recreated",
                        path=str(path),
                        error=str(exc),
                    )
                )
                path.write_text(json.dumps(defaults, indent=4), encoding="utf-8")
                return True

        payload = _ensure_hotkey_payload(current if isinstance(current, dict) else {})
        if payload != current:
            path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
            return True
    except Exception as exc:
        LOGGER.error(
            StructuredMessage(
                "Failed to prepare hotkey configuration file.",
                event="startup.hotkey_config_failed",
                path=str(path),
                error=str(exc),
            ),
            exc_info=True,
        )
        raise

    return False


def _log_artifact_ready(description: str, path: Path, *, created: bool) -> None:
    exists = path.exists()
    size_bytes = 0
    if exists:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = 0

    LOGGER.info(
        StructuredMessage(
            f"{description} available.",
            event="startup.artifact_ready",
            path=str(path),
            created=created,
            exists=exists,
            size_bytes=size_bytes,
        )
    )


def run_startup_preflight(config_manager, *, hotkey_config_path: Path) -> None:
    """Ensure essential artifacts exist before continuing with the UI bootstrap."""

    LOGGER.info(
        StructuredMessage(
            "Running startup preflight checks.",
            event="startup.preflight.begin",
        )
    )

    persistence = config_manager.save_config()
    config_path = Path(persistence.config.path).resolve()
    config_created = persistence.config.created
    _log_artifact_ready("Configuration", config_path, created=config_created)
    if persistence.config.error:
        raise RuntimeError(
            f"Configuration persistence failed: {persistence.config.error}"
        )
    if not config_path.exists():
        raise RuntimeError(f"Configuration file missing after preflight: {config_path}")
    if not persistence.config.verified:
        raise RuntimeError(
            "Configuration file could not be verified after save operation."
        )

    secrets_path = Path(config_module.SECRETS_FILE).resolve()
    secrets_payload: dict[str, object] = {
        config_module.GEMINI_API_KEY_CONFIG_KEY: "",
        config_module.OPENROUTER_API_KEY_CONFIG_KEY: "",
    }
    secrets_created = _ensure_json_file(
        secrets_path,
        secrets_payload,
        description="secrets",
    )
    _log_artifact_ready("Secrets", secrets_path, created=secrets_created)
    if persistence.secrets.error:
        raise RuntimeError(
            f"Secrets persistence failed: {persistence.secrets.error}"
        )
    if not secrets_path.exists():
        raise RuntimeError(f"Secrets file missing after preflight: {secrets_path}")
    if not persistence.secrets.verified and persistence.secrets.wrote:
        raise RuntimeError(
            "Secrets file could not be verified after save operation."
        )

    _maybe_migrate_hotkey_config(hotkey_config_path, LEGACY_HOTKEY_CONFIG_PATHS)
    hotkey_created = _ensure_hotkey_config(hotkey_config_path)
    _log_artifact_ready("Hotkey configuration", hotkey_config_path, created=hotkey_created)
    if not hotkey_config_path.exists():
        raise RuntimeError(
            f"Hotkey configuration missing after preflight: {hotkey_config_path}"
        )

    bootstrap_report = config_manager.get_bootstrap_report()
    LOGGER.info(
        StructuredMessage(
            "Persistence bootstrap report ready.",
            event="startup.preflight.bootstrap_report",
            details={
                key: {
                    "path": value.get("path"),
                    "existed": value.get("existed"),
                    "created": value.get("created"),
                    "written": value.get("written"),
                    "verified": value.get("verified"),
                    "error": value.get("error"),
                }
                for key, value in bootstrap_report.items()
            },
        )
    )

    LOGGER.info(
        StructuredMessage(
            "Startup preflight checks completed.",
            event="startup.preflight.complete",
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    headless = bool(args.headless)
    skip_bootstrap = bool(getattr(args, "skip_bootstrap", False))
    setup_logging()
    install_exception_hooks(logger=LOGGER)
    LOGGER.debug(
        StructuredMessage(
            "Global exception hooks registered for runtime diagnostics.",
            event="bootstrap.exception_hooks",
        )
    )
    emit_startup_banner(
        extra_details={
            "icon_path": str(ICON_PATH),
            "hotkey_config": str(HOTKEY_CONFIG_PATH),
        }
    )
    LOGGER.info(
        StructuredMessage(
            "Whisper Flash Transcriber bootstrap sequence started.",
            event="bootstrap.start",
            python_version=sys.version.split()[0],
            working_directory=PROJECT_ROOT,
            diagnostics_only=bool(args.diagnostics),
            headless=headless,
        )
    )

    try:
        configure_environment()
        if not args.diagnostics and not headless:
            ensure_display_available()
        configure_cuda_logging()
        patch_tk_variable_cleanup()

        from src.config_manager import ConfigManager  # noqa: E402
        from src.core import AppCore  # noqa: E402
        from src.ui_manager import UIManager  # noqa: E402
        from src.installers.dependency_installer import (  # noqa: E402
            DependencyInstaller,
        )
        from src.startup_diagnostics import (  # noqa: E402
            format_report_for_console,
            run_startup_diagnostics,
        )

        LOGGER.info(
            StructuredMessage(
                "Initializing ConfigManager.",
                event="bootstrap.step.config_manager.init",
            )
        )
        config_manager = ConfigManager()
        LOGGER.info(
            StructuredMessage(
                "ConfigManager ready.",
                event="bootstrap.step.config_manager.ready",
            )
        )
        dependency_installer = DependencyInstaller(
            config_manager.get_python_packages_dir()
        )
        LOGGER.info(
            StructuredMessage(
                "DependencyInstaller created.",
                event="bootstrap.step.dependency_installer.init",
            )
        )
        dependency_installer.ensure_in_sys_path()
        for note in dependency_installer.environment_notes:
            LOGGER.info(
                StructuredMessage(
                    "Dependency environment advisory.",
                    event="startup.dependency_env_note",
                    note=note,
                )
            )
        LOGGER.info(
            StructuredMessage(
                "DependencyInstaller path ensured.",
                event="bootstrap.step.dependency_installer.ready",
            )
        )

        hf_cache_dir = config_manager.get_hf_cache_dir()
        try:
            if hf_cache_dir:
                Path(hf_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("HF_HOME", hf_cache_dir)
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
        except Exception as exc:
            LOGGER.warning(
                StructuredMessage(
                    "Failed to prepare Hugging Face cache directory.",
                    event="startup.hf_cache.prepare_failed",
                    path=hf_cache_dir,
                    error=str(exc),
                ),
                exc_info=True,
            )

        if args.diagnostics and skip_bootstrap:
            LOGGER.error(
                StructuredMessage(
                    "Cannot combine diagnostics mode with bootstrap skipping.",
                    event="bootstrap.invalid_args",
                )
            )
            return 2

        diagnostics_report = None
        if skip_bootstrap:
            LOGGER.warning(
                StructuredMessage(
                    "Bootstrap preflight and diagnostics skipped via command-line flag.",
                    event="bootstrap.skipped",
                )
            )
        else:
            run_startup_preflight(config_manager, hotkey_config_path=HOTKEY_CONFIG_PATH)
            diagnostics_report = run_startup_diagnostics(
                config_manager,
                hotkey_config_path=HOTKEY_CONFIG_PATH,
            )

            if args.diagnostics:
                print(format_report_for_console(diagnostics_report))
                exit_code = 1 if diagnostics_report.has_errors else 0
                LOGGER.info(
                    StructuredMessage(
                        "Diagnostics-only execution completed.",
                        event="bootstrap.diagnostics_only_complete",
                        exit_code=exit_code,
                    )
                )
                return exit_code

        main_tk_root: tk.Tk | None = None
        app_core_instance = None
        ui_manager_instance = None

        def on_exit_app_enhanced(*_):
            LOGGER.info(
                StructuredMessage(
                    "Exit requested from tray icon.",
                    event="ui.exit_requested",
                )
            )
            if app_core_instance:
                app_core_instance.shutdown()
            if ui_manager_instance and ui_manager_instance.tray_icon:
                ui_manager_instance.tray_icon.stop()
            if main_tk_root is not None:
                main_tk_root.after(0, main_tk_root.quit)

        atexit.register(
            lambda: LOGGER.info(
                StructuredMessage(
                    "Application terminated.",
                    event="shutdown.complete",
                )
            )
        )

        LOGGER.info(
            StructuredMessage(
                "Creating Tk root window.",
                event="bootstrap.step.tk.init",
            )
        )
        main_tk_root = tk.Tk()
        main_tk_root.withdraw()
        main_tk_root.title("Whisper Flash Transcriber")
        main_tk_root.protocol("WM_DELETE_WINDOW", on_exit_app_enhanced)
        LOGGER.info(
            StructuredMessage(
                "Tk root window ready.",
                event="bootstrap.step.tk.ready",
            )
        )

        icon_path = ICON_PATH
        if not os.path.exists(icon_path):
            LOGGER.warning(
                StructuredMessage(
                    "Main window icon not found on disk.",
                    event="ui.icon_missing",
                    path=icon_path,
                )
            )
        else:
            try:
                main_tk_root.iconbitmap(icon_path)
            except Exception:
                LOGGER.warning(
                    StructuredMessage(
                        "Failed to set main window icon.",
                        event="ui.icon_application_failed",
                        path=icon_path,
                    )
                )

        LOGGER.info(
            StructuredMessage(
                "Instantiating AppCore.",
                event="bootstrap.step.app_core.init",
            )
        )
        app_core_instance = AppCore(
            main_tk_root,
            config_manager=config_manager,
            hotkey_config_path=str(HOTKEY_CONFIG_PATH),
            startup_diagnostics=diagnostics_report,
            enable_onboarding=not headless,
        )
        LOGGER.info(
            StructuredMessage(
                "AppCore ready.",
                event="bootstrap.step.app_core.ready",
            )
        )
        ui_manager_instance = UIManager(
            main_tk_root,
            app_core_instance.config_manager,
            app_core_instance,
            model_manager=app_core_instance.model_manager,
            is_running_as_admin=app_core_instance.is_running_as_admin,
        )
        LOGGER.info(
            StructuredMessage(
                "UIManager ready.",
                event="bootstrap.step.ui_manager.ready",
            )
        )
        app_core_instance.ui_manager = ui_manager_instance
        ui_manager_instance.setup_tray_icon()
        app_core_instance.flush_pending_ui_notifications()
        ui_manager_instance.on_exit_app = on_exit_app_enhanced

        main_tk_root.after(0, app_core_instance.schedule_initial_onboarding)
        LOGGER.info(
            StructuredMessage(
                "Initial onboarding scheduled on Tkinter event loop.",
                event="bootstrap.onboarding_scheduled",
                onboarding_enabled=app_core_instance.onboarding_enabled,
            )
        )

        if diagnostics_report and diagnostics_report.has_fatal_errors:
            fatal_summary = "\n\n".join(
                diagnostics_report.user_friendly_summary(include_success=False)
            )

            def _show_diagnostics_failure() -> None:
                messagebox.showerror(
                    "Startup diagnostics",
                    fatal_summary,
                )

            main_tk_root.after(0, _show_diagnostics_failure)

        LOGGER.info(
            StructuredMessage(
                "Starting Tkinter mainloop.",
                event="ui.mainloop.start",
                thread=threading.current_thread().name,
            )
        )
        main_tk_root.mainloop()
        LOGGER.info(
            StructuredMessage(
                "Tkinter mainloop finished; application will exit.",
                event="ui.mainloop.stop",
            )
        )
        return 0
    except Exception as exc:
        LOGGER.critical(
            StructuredMessage(
                "Fatal error during application startup.",
                event="bootstrap.failure",
                error=str(exc),
            ),
            exc_info=True,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
