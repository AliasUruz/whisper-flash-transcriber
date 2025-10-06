import atexit
import importlib.util
import json
import os
import shutil
import sys
import threading
import tkinter as tk
from pathlib import Path

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import src.config_manager as config_module


ICON_PATH = os.path.join(PROJECT_ROOT, "icon.ico")
HOTKEY_CONFIG_PATH = Path(config_module.HOTKEY_CONFIG_FILE).expanduser()
LEGACY_HOTKEY_CONFIG_PATHS = tuple(
    Path(path).expanduser() for path in config_module.LEGACY_HOTKEY_LOCATIONS
)

from src.logging_utils import (
    StructuredMessage,
    emit_startup_banner,
    get_logger,
    setup_logging,
)


LOGGER = get_logger("whisper_flash_transcriber.bootstrap", component="Bootstrap")


ENV_DEFAULTS = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "BITSANDBYTES_NOWELCOME": "1",
}


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
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            LOGGER.debug(
                StructuredMessage(
                    "Torch module not available; skipping CUDA diagnostics.",
                    event="startup.cuda_skipped",
                    reason="torch_missing",
                )
            )
            return

        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                LOGGER.info(
                    StructuredMessage(
                        "cuDNN benchmark enabled for CUDA runtime.",
                        event="cuda.cudnn_benchmark_enabled",
                    )
                )
            except Exception as exc:
                LOGGER.warning(
                    StructuredMessage(
                        "Unable to enable cuDNN benchmark mode.",
                        event="cuda.cudnn_benchmark_failure",
                        error=str(exc),
                    )
                )

            try:
                num_gpus = torch.cuda.device_count()
                LOGGER.info(
                    StructuredMessage(
                        "CUDA runtime detected.",
                        event="cuda.runtime_detected",
                        cuda_version=torch.version.cuda,
                        gpu_count=num_gpus,
                    )
                )
                for idx in range(num_gpus):
                    try:
                        name = torch.cuda.get_device_name(idx)
                        props = torch.cuda.get_device_properties(idx)
                        total_mem_gb = props.total_memory / (1024 ** 3)
                        capability = f"{props.major}.{props.minor}"
                        LOGGER.info(
                            StructuredMessage(
                                "GPU device enumerated.",
                                event="cuda.gpu_discovered",
                                index=idx,
                                name=name,
                                vram_gb=f"{total_mem_gb:.2f}",
                                compute_capability=capability,
                            )
                        )
                    except Exception as gpu_exc:
                        LOGGER.warning(
                            StructuredMessage(
                                "Unable to query GPU device properties.",
                                event="cuda.gpu_introspection_failed",
                                index=idx,
                                error=str(gpu_exc),
                            )
                        )

                try:
                    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
                except Exception:
                    has_flash_attn = False
                LOGGER.info(
                    StructuredMessage(
                        "FlashAttention 2 availability checked.",
                        event="cuda.flash_attention_probe",
                        available=has_flash_attn,
                    )
                )
            except Exception as diag_exc:
                LOGGER.warning(
                    StructuredMessage(
                        "Unable to gather CUDA diagnostics.",
                        event="cuda.diagnostics_failed",
                        error=str(diag_exc),
                    )
                )
        else:
            LOGGER.info(
                StructuredMessage(
                    "CUDA runtime not detected; defaulting to CPU execution.",
                    event="cuda.cpu_fallback",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            StructuredMessage(
                "Skipping CUDA diagnostics due to unexpected error.",
                event="cuda.diagnostics_exception",
                error=str(exc),
            ),
            exc_info=True,
        )


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


def _ensure_hotkey_payload(data: dict[str, object]) -> dict[str, object]:
    defaults = {"record_key": "f3", "agent_key": "f4", "record_mode": "toggle"}
    updated = dict(defaults)
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
    payload: dict[str, object],
    *,
    description: str,
    recover_on_error: bool = True,
) -> bool:
    """Persist ``payload`` when ``path`` is missing; return True if file was created."""

    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
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
            path.write_text(json.dumps(payload, indent=4), encoding="utf-8")
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

    config_path = Path(config_manager.config_file).resolve()
    config_pre_exists = config_path.exists()
    config_manager.save_config()
    config_created = not config_pre_exists and config_path.exists()
    _log_artifact_ready("Configuration", config_path, created=config_created)
    if not config_path.exists():
        raise RuntimeError(f"Configuration file missing after preflight: {config_path}")

    secrets_path = Path(config_module.SECRETS_FILE).resolve()
    secrets_payload = {
        config_module.GEMINI_API_KEY_CONFIG_KEY: "",
        config_module.OPENROUTER_API_KEY_CONFIG_KEY: "",
    }
    secrets_created = _ensure_json_file(
        secrets_path,
        secrets_payload,
        description="secrets",
    )
    _log_artifact_ready("Secrets", secrets_path, created=secrets_created)
    if not secrets_path.exists():
        raise RuntimeError(f"Secrets file missing after preflight: {secrets_path}")

    _maybe_migrate_hotkey_config(hotkey_config_path, LEGACY_HOTKEY_CONFIG_PATHS)
    hotkey_created = _ensure_hotkey_config(hotkey_config_path)
    _log_artifact_ready("Hotkey configuration", hotkey_config_path, created=hotkey_created)
    if not hotkey_config_path.exists():
        raise RuntimeError(
            f"Hotkey configuration missing after preflight: {hotkey_config_path}"
        )

    LOGGER.info(
        StructuredMessage(
            "Startup preflight checks completed.",
            event="startup.preflight.complete",
        )
    )


def main() -> None:
    setup_logging()
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
        )
    )

    try:
        configure_environment()
        ensure_display_available()
        configure_cuda_logging()
        patch_tk_variable_cleanup()

        from src.config_manager import ConfigManager  # noqa: E402
        from src.core import AppCore  # noqa: E402
        from src.ui_manager import UIManager  # noqa: E402

        config_manager = ConfigManager()
        run_startup_preflight(config_manager, hotkey_config_path=HOTKEY_CONFIG_PATH)

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
            main_tk_root.after(0, main_tk_root.quit)

        atexit.register(
            lambda: LOGGER.info(
                StructuredMessage(
                    "Application terminated.",
                    event="shutdown.complete",
                )
            )
        )

        main_tk_root = tk.Tk()
        main_tk_root.withdraw()
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

        app_core_instance = AppCore(
            main_tk_root,
            config_manager=config_manager,
            hotkey_config_path=str(HOTKEY_CONFIG_PATH),
        )
        ui_manager_instance = UIManager(
            main_tk_root,
            app_core_instance.config_manager,
            app_core_instance,
            model_manager=app_core_instance.model_manager,
        )
        app_core_instance.ui_manager = ui_manager_instance
        ui_manager_instance.setup_tray_icon()
        app_core_instance.flush_pending_ui_notifications()
        ui_manager_instance.on_exit_app = on_exit_app_enhanced

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
    main()
