import atexit
import importlib
import os
import sys
import threading
import tkinter as tk
import threading

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


ICON_PATH = os.path.join(PROJECT_ROOT, "icon.ico")


from src.logging_utils import StructuredMessage, setup_logging


LOGGER = logging.getLogger("whisper_flash_transcriber.bootstrap")


ENV_DEFAULTS = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "BITSANDBYTES_NOWELCOME": "1",
}


from src.logging_utils import get_logger, log_context, setup_logging


LOGGER = get_logger("whisper_flash_transcriber.bootstrap", component="Bootstrap")


def ensure_display_available() -> None:
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        LOGGER.warning(
            log_context(
                "DISPLAY environment variable is not set; running in headless mode.",
                event="bootstrap.display_missing",
                action="abort_gui_startup",
                os_name=os.name,
            )
        )
        sys.exit(0)


def configure_environment() -> None:
    applied_defaults = 0
    for key, value in ENV_DEFAULTS.items():
        if key in os.environ:
            LOGGER.debug(
                log_context(
                    "Environment variable already defined; keeping existing value.",
                    event="bootstrap.env.skipped",
                    variable=key,
                    current_value=os.environ.get(key),
                )
            )
            continue

        os.environ[key] = value
        applied_defaults += 1
        LOGGER.debug(
            log_context(
                "Default environment variable applied.",
                event="bootstrap.env.default",
                variable=key,
                value=value,
            )
        )
    transformers_verbosity_before = os.environ.get("TRANSFORMERS_VERBOSITY")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", transformers_verbosity_before or "error")
    LOGGER.info(
        log_context(
            "Environment configuration completed.",
            event="bootstrap.env.ready",
            applied_defaults=applied_defaults,
            transformers_verbosity=os.environ.get("TRANSFORMERS_VERBOSITY"),
        )
    )


def configure_cuda_logging() -> None:
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            LOGGER.debug(
                log_context(
                    "PyTorch not available; skipping CUDA diagnostics.",
                    event="bootstrap.cuda.skip",
                    reason="torch_not_installed",
                )
            )
            return

        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                LOGGER.info(
                    log_context(
                        "cudnn.benchmark enabled for CUDA backend.",
                        event="bootstrap.cuda.cudnn_benchmark",
                        status="enabled",
                    )
                )
            except Exception as exc:
                LOGGER.warning(
                    log_context(
                        "Failed to enable cudnn.benchmark.",
                        event="bootstrap.cuda.cudnn_benchmark",
                        status="failed",
                        details=str(exc),
                    ),
                    exc_info=True,
                )

            try:
                num_gpus = torch.cuda.device_count()
                LOGGER.info(
                    log_context(
                        "CUDA runtime detected.",
                        event="bootstrap.cuda.detected",
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
                            log_context(
                                "GPU diagnostics collected.",
                                event="bootstrap.cuda.gpu_profile",
                                gpu_index=idx,
                                gpu_name=name,
                                total_vram_gb=round(total_mem_gb, 2),
                                compute_capability=capability,
                            )
                        )
                    except Exception as gpu_exc:
                        LOGGER.warning(
                            log_context(
                                "Unable to query GPU properties.",
                                event="bootstrap.cuda.gpu_profile",
                                gpu_index=idx,
                                status="failed",
                                details=str(gpu_exc),
                            ),
                            exc_info=True,
                        )

                try:
                    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
                except Exception:
                    has_flash_attn = False
                LOGGER.info(
                    log_context(
                        "FlashAttention 2 availability checked.",
                        event="bootstrap.cuda.flash_attn",
                        available=has_flash_attn,
                    )
                )
            except Exception as diag_exc:
                LOGGER.warning(
                    log_context(
                        "Failed to gather CUDA capabilities.",
                        event="bootstrap.cuda.diagnostics_failed",
                        details=str(diag_exc),
                    ),
                    exc_info=True,
                )
        else:
            LOGGER.info(
                log_context(
                    "CUDA runtime unavailable; defaulting to CPU execution.",
                    event="bootstrap.device_selection",
                    device="cpu",
                    reason="no_cuda_available",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning(
            log_context(
                "Skipping CUDA diagnostics due to unexpected error.",
                event="bootstrap.cuda.diagnostics_failed",
                details=str(exc),
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


def main() -> None:
    setup_logging()
    LOGGER.info(
        log_context(
            "Whisper Flash Transcriber bootstrap sequence started.",
            event="bootstrap.start",
            python_version=sys.version.split()[0],
            working_directory=PROJECT_ROOT,
        )
    )
    configure_environment()
    ensure_display_available()
    configure_cuda_logging()
    patch_tk_variable_cleanup()

    from src.core import AppCore  # noqa: E402
    from src.ui_manager import UIManager  # noqa: E402

    app_core_instance = None
    ui_manager_instance = None

    def on_exit_app_enhanced(*_):
        LOGGER.info(
            log_context(
                "Exit requested from tray icon.",
                event="bootstrap.exit_requested",
            )
        )
        if app_core_instance:
            app_core_instance.shutdown()
        if ui_manager_instance and ui_manager_instance.tray_icon:
            ui_manager_instance.tray_icon.stop()
        main_tk_root.after(0, main_tk_root.quit)

    atexit.register(
        lambda: LOGGER.info(
            log_context(
                "Application terminated.",
                event="bootstrap.terminated",
            )
        )
    )

    main_tk_root = tk.Tk()
    main_tk_root.withdraw()
    icon_path = ICON_PATH
    if not os.path.exists(icon_path):
        LOGGER.warning(
            log_context(
                "Failed to set main window icon: file not found.",
                event="bootstrap.icon",
                status="missing",
                path=icon_path,
            )
        )
    else:
        try:
            main_tk_root.iconbitmap(icon_path)
        except Exception:
            LOGGER.warning(
                log_context(
                    "Failed to apply main window icon; the file may be invalid.",
                    event="bootstrap.icon",
                    status="invalid",
                    path=icon_path,
                ),
                exc_info=True,
            )

    app_core_instance = AppCore(main_tk_root)
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
        log_context(
            "Starting Tkinter mainloop.",
            event="bootstrap.mainloop_start",
            thread=threading.current_thread().name,
        )
    )
    main_tk_root.mainloop()
    LOGGER.info(
        log_context(
            "Tkinter mainloop finished; application will exit.",
            event="bootstrap.mainloop_end",
        )
    )


if __name__ == "__main__":
    main()
