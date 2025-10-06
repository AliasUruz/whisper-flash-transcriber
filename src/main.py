import atexit
import importlib
import logging
import os
import sys
import tkinter as tk

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


ICON_PATH = os.path.join(PROJECT_ROOT, "icon.ico")


ENV_DEFAULTS = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "BITSANDBYTES_NOWELCOME": "1",
}


def ensure_display_available() -> None:
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        logging.warning(
            "DISPLAY environment variable missing; aborting GUI startup for test mode.",
            extra={"event": "display_validation", "status": "missing"},
        )
        sys.exit(0)


def configure_environment() -> None:
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", os.environ.get("TRANSFORMERS_VERBOSITY", "error"))
    logging.info(
        "Environment defaults applied.",
        extra={"event": "environment_config", "details": ",".join(sorted(ENV_DEFAULTS.keys()))},
    )


def configure_cuda_logging() -> None:
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            logging.debug(
                "torch not available; skipping CUDA diagnostics.",
                extra={"event": "cuda_diagnostics", "status": "torch_missing"},
            )
            return

        import torch  # type: ignore

        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                logging.info(
                    "cudnn.benchmark enabled (CUDA available).",
                    extra={"event": "cuda_optimization", "status": "enabled"},
                )
            except Exception as exc:
                logging.warning(
                    "Unable to enable cudnn.benchmark: %s",
                    exc,
                    extra={"event": "cuda_optimization", "status": "failed"},
                )

            try:
                num_gpus = torch.cuda.device_count()
                logging.info(
                    "CUDA runtime detected.",
                    extra={
                        "event": "cuda_detected",
                        "details": f"version={torch.version.cuda} gpu_count={num_gpus}",
                    },
                )
                for idx in range(num_gpus):
                    try:
                        name = torch.cuda.get_device_name(idx)
                        props = torch.cuda.get_device_properties(idx)
                        total_mem_gb = props.total_memory / (1024 ** 3)
                        capability = f"{props.major}.{props.minor}"
                        logging.info(
                            "GPU inventory report.",
                            extra={
                                "event": "cuda_gpu_profile",
                                "details": (
                                    f"index={idx} name={name} vram_gb={total_mem_gb:.2f} capability={capability}"
                                ),
                            },
                        )
                    except Exception as gpu_exc:
                        logging.warning(
                            "Unable to query GPU properties: %s",
                            gpu_exc,
                            extra={"event": "cuda_gpu_profile", "status": "failed", "details": f"index={idx}"},
                        )

                try:
                    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
                except Exception:
                    has_flash_attn = False
                logging.info(
                    "FlashAttention 2 availability assessed.",
                    extra={"event": "flash_attention_check", "status": str(has_flash_attn).lower()},
                )
            except Exception as diag_exc:
                logging.warning(
                    "Unable to gather CUDA capabilities: %s",
                    diag_exc,
                    extra={"event": "cuda_diagnostics", "status": "failed"},
                )
        else:
            logging.info(
                "CUDA runtime not available; defaulting to CPU execution.",
                extra={"event": "device_select", "status": "cpu"},
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        logging.warning(
            "Skipping CUDA diagnostics due to error: %s",
            exc,
            exc_info=True,
            extra={"event": "cuda_diagnostics", "status": "error"},
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
    from src.logging_utils import setup_logging

    setup_logging()
    configure_environment()
    ensure_display_available()
    configure_cuda_logging()
    patch_tk_variable_cleanup()

    from src.core import AppCore  # noqa: E402
    from src.ui_manager import UIManager  # noqa: E402

    app_core_instance = None
    ui_manager_instance = None

    def on_exit_app_enhanced(*_):
        logging.info(
            "Exit requested from tray icon.",
            extra={"event": "shutdown", "status": "requested"},
        )
        if app_core_instance:
            app_core_instance.shutdown()
        if ui_manager_instance and ui_manager_instance.tray_icon:
            ui_manager_instance.tray_icon.stop()
        main_tk_root.after(0, main_tk_root.quit)

    def _log_process_shutdown() -> None:
        logging.info(
            "Application terminated.",
            extra={"event": "shutdown", "status": "complete"},
        )

    atexit.register(_log_process_shutdown)

    main_tk_root = tk.Tk()
    main_tk_root.withdraw()
    icon_path = ICON_PATH
    if not os.path.exists(icon_path):
        logging.warning(
            "Failed to set main window icon. File not found: %s",
            icon_path,
            extra={"event": "icon_setup", "status": "missing"},
        )
    else:
        try:
            # Define o ícone da aplicação, que aparecerá na barra de tarefas
            main_tk_root.iconbitmap(icon_path)
        except Exception:
            logging.warning(
                "Failed to set main window icon. icon.ico may be missing or invalid at %s.",
                icon_path,
                extra={"event": "icon_setup", "status": "error"},
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

    logging.info(
        "Starting Tkinter mainloop on the main thread.",
        extra={"event": "tk_mainloop", "status": "starting"},
    )
    main_tk_root.mainloop()
    logging.info(
        "Tkinter mainloop finished. The application will exit.",
        extra={"event": "tk_mainloop", "status": "finished"},
    )


if __name__ == "__main__":
    main()
