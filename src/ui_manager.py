
import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import BooleanVar, filedialog, simpledialog  # Adicionado para diálogos
import logging
import threading
import time
import os
import sys
import subprocess
import webbrowser
from datetime import datetime
import pystray
from PIL import Image, ImageDraw
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

# Importar constantes de configuração
from .config_manager import (
    SETTINGS_WINDOW_GEOMETRY,
    SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI,
    GEMINI_AGENT_MODEL_CONFIG_KEY,
    GEMINI_AGENT_PROMPT_CONFIG_KEY,
    GEMINI_MODEL_CONFIG_KEY,
    GEMINI_MODEL_OPTIONS_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    OPENROUTER_API_KEY_CONFIG_KEY,
    OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    STORAGE_ROOT_DIR_CONFIG_KEY,
    RECORDINGS_DIR_CONFIG_KEY,
    ASR_COMPUTE_DEVICE_CONFIG_KEY,
    ASR_BACKEND_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
    ASR_CACHE_DIR_CONFIG_KEY,
    RECORDINGS_DIR_CONFIG_KEY,
    DEPS_INSTALL_DIR_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
    VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
    DEFAULT_CONFIG,
)
from . import state_manager as sm

from .utils.form_validation import safe_get_float, safe_get_int
from .utils.tooltip import Tooltip
from .logging_utils import get_log_directory
from .state_manager import StateEvent
from .utils.dependency_audit import DependencyAuditResult, DependencyIssue

# Importar get_available_devices_for_ui (pode ser movido para um utils ou ficar aqui)
# Por enquanto, vamos assumir que está disponível globalmente ou será movido para cá.
# Para este plano, vamos movê-lo para cá.
# import torch # Necessário para get_available_devices_for_ui - REMOVIDO

try:
    from .model_manager import (
        DownloadCancelledError as _DefaultDownloadCancelledError,
        ModelDownloadResult as _DefaultModelDownloadResult,
    )
except Exception:  # pragma: no cover - fallback se a exceção não existir
    @dataclass(frozen=True)
    class _DefaultModelDownloadResult:
        path: str = ""
        downloaded: bool = False

    class _DefaultDownloadCancelledError(Exception):
        """Fallback exception when model_manager is unavailable."""

        pass

# Importar gerenciador de modelos de ASR
try:
    from . import model_manager as _default_model_manager
except Exception:  # pragma: no cover - fallback caso o módulo não exista
    class _DummyModelManager:
        """Fallback com operações inertes quando o model_manager não está disponível."""

        DownloadCancelledError = _DefaultDownloadCancelledError

        @staticmethod
        def list_catalog():
            logging.warning("model_manager module not available.")
            return []

        @staticmethod
        def list_installed(*_args, **_kwargs):
            logging.warning("model_manager module not available.")
            return []

        @staticmethod
        def ensure_download(*_args, **_kwargs):
            logging.warning("model_manager module not available.")
            return _DefaultModelDownloadResult("", False)

        @staticmethod
        def get_model_download_size(*_args, **_kwargs):
            logging.warning("model_manager module not available.")
            return 0, 0

        @staticmethod
        def get_installed_size(*_args, **_kwargs):
            logging.warning("model_manager module not available.")
            return 0, 0

    _default_model_manager = _DummyModelManager()

def get_available_devices_for_ui():
    """Returns a list of devices for the settings interface."""
    devices = ["Auto-select (Recommended)"]
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                try:
                    name = torch.cuda.get_device_name(i)
                    total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    devices.append(f"GPU {i}: {name} ({total_mem_gb:.1f}GB)")
                except Exception as e:
                    devices.append(f"GPU {i}: Error getting name")
                    logging.error(f"Could not get GPU name {i}: {e}")
    except ImportError:
        logging.debug("torch not found, returning CPU-only devices.")
    devices.append("Force CPU")
    return devices


def _backend_display_value_global(value: str | None) -> str:
    """Normalize backend label for display and configuration."""
    normalized = (value or "").strip().lower()
    if normalized in {"ct2", "ctranslate2"}:
        return "ctranslate2"
    if normalized in {"faster whisper", "faster_whisper", "faster-whisper"}:
        return "ctranslate2"
    return normalized

class UIManager:
    def __init__(self, main_tk_root, config_manager, core_instance_ref, model_manager=None):
        self.main_tk_root = main_tk_root
        self.config_manager = config_manager
        self.core_instance_ref = core_instance_ref # Reference to the AppCore instance

        self.model_manager = model_manager if model_manager is not None else _default_model_manager
        self._download_cancelled_error = getattr(
            self.model_manager,
            "DownloadCancelledError",
            _DefaultDownloadCancelledError,
        )

        self.tray_icon = None
        self._pending_tray_tooltip = None
        self.settings_window_instance = None
        self.settings_thread_running = False
        self.settings_window_lock = threading.Lock()

        # Contexto ativo da janela de configurações
        self._settings_vars: Dict[str, Any] = {}
        self._settings_meta: Dict[str, Any] = {}
        self._pending_key_var_name: Optional[str] = None

        self.live_window = None
        self.live_textbox = None

        self._divergent_keys_logged: set[str] = set()

        self._download_window = None
        self._download_window_widgets: Dict[str, dict[str, Any]] = {}
        self._download_window_lock = threading.Lock()
        self._download_snapshot: list[dict[str, Any]] = []
        self._download_history: list[dict[str, Any]] = []

        self._last_operation_id: str | None = None


        # Assign methods to the instance
        self.show_live_transcription_window = self._show_live_transcription_window
        self.update_live_transcription = self._update_live_transcription
        self.close_live_transcription_window = self._close_live_transcription_window
        self.update_live_transcription_threadsafe = self.update_live_transcription_threadsafe

        # State mapping to icon colors (moved from global)
        self.ICON_COLORS = {
            "IDLE": ('green', 'white'),
            "LOADING_MODEL": ('gray', 'yellow'),
            "RECORDING": ('red', 'white'),
            "TRANSCRIBING": ('blue', 'white'),
            "ERROR_MODEL": ('black', 'red'),
            "ERROR_AUDIO": ('black', 'red'),
            "ERROR_TRANSCRIPTION": ('black', 'red'),
            "ERROR_SETTINGS": ('black', 'red'),
        }
        self.DEFAULT_ICON_COLOR = ('black', 'white')

        # Controle interno para atualizar a tooltip durante a gravação
        self.recording_timer_thread = None
        self.stop_recording_timer_event = threading.Event()

        # Controle interno para atualizar a tooltip durante a transcrição
        self.transcribing_timer_thread = None
        self.stop_transcribing_timer_event = threading.Event()

        # Contexto do último estado recebido (para tooltips dinâmicas)
        self._last_state_notification: Any = None
        self._state_context_suffix: str = ""

    # ------------------------------------------------------------------
    # Utilidades para gerenciamento da janela de configurações
    # ------------------------------------------------------------------
    def _get_core_state(self) -> str:
        """Retorna o estado atual exposto pelo ``StateManager`` do núcleo."""

        core = getattr(self, "core_instance_ref", None)
        if not core:
            return "IDLE"

        state_manager = getattr(core, "state_manager", None)
        if state_manager and hasattr(state_manager, "get_current_state"):
            try:
                state = state_manager.get_current_state()
            except Exception:
                logging.debug(
                    "UIManager: falha ao obter estado corrente via state_manager.",
                    exc_info=True,
                )
            else:
                if isinstance(state, str):
                    return state

        return "IDLE"

    def _set_settings_var(self, name: str, value: Any) -> None:
        self._settings_vars[name] = value

    def _get_settings_var(self, name: str) -> Any:
        return self._settings_vars.get(name)

    def _set_settings_meta(self, name: str, value: Any) -> None:
        self._settings_meta[name] = value

    def _get_settings_meta(self, name: str, default: Any = None) -> Any:
        return self._settings_meta.get(name, default)

    def _clear_settings_context(self) -> None:
        self._settings_vars.clear()
        self._settings_meta.clear()

    def _open_model_downloads_window(self) -> None:
        with self._download_window_lock:
            if self._download_window and self._download_window.winfo_exists():
                self._download_window.lift()
                self._download_window.focus_force()
                return
            window = ctk.CTkToplevel(self.main_tk_root)
            window.title("Model Downloads")
            window.geometry("620x520")
            window.protocol("WM_DELETE_WINDOW", self._close_model_downloads_window)
            outer = ctk.CTkFrame(window)
            outer.pack(fill="both", expand=True, padx=10, pady=10)
            active_container = ctk.CTkScrollableFrame(outer, label_text="Active Downloads")
            active_container.pack(fill="both", expand=True)
            history_container = ctk.CTkScrollableFrame(
                outer,
                label_text="Recent Download History",
                height=180,
            )
            history_container.pack(fill="both", expand=True, pady=(10, 0))
            self._download_window = window
            self._download_window_widgets.clear()
            self._download_window_widgets["_container"] = active_container
            self._download_window_widgets["_history_container"] = history_container

        snapshot = {}
        core = getattr(self, "core_instance_ref", None)
        if core is not None and hasattr(core, "get_model_downloads_snapshot"):
            try:
                snapshot = core.get_model_downloads_snapshot()
            except Exception:
                logging.debug("Failed to obtain download snapshot from core.", exc_info=True)
        tasks = snapshot.get("tasks", []) if isinstance(snapshot, dict) else []
        if tasks:
            self._download_snapshot = tasks
        self._refresh_download_window(self._download_snapshot)
        self._update_download_history_cache()
        self._refresh_download_history()

    def _close_model_downloads_window(self) -> None:
        with self._download_window_lock:
            if self._download_window and self._download_window.winfo_exists():
                self._download_window.destroy()
            self._download_window = None

    def _refresh_download_window(self, tasks: list[dict[str, Any]]) -> None:
        with self._download_window_lock:
            window = self._download_window
            container = self._download_window_widgets.get("_container")
        if not window or not window.winfo_exists() or container is None:
            return

        existing_ids = {
            key
            for key in self._download_window_widgets.keys()
            if key not in {"_container", "_history_container"}
        }
        new_ids: set[str] = set()

        for task in tasks:
            task_id = str(task.get("task_id") or "").strip()
            if not task_id:
                continue
            new_ids.add(task_id)
            self._ensure_download_row(container, task_id, task)

        stale_ids = existing_ids - new_ids
        for task_id in stale_ids:
            widgets = self._download_window_widgets.pop(task_id, None)
            if widgets:
                frame = widgets.get("frame")
                if frame is not None:
                    frame.destroy()

    def _update_download_history_cache(self) -> None:
        try:
            history = self.config_manager.get_model_download_history(limit=50)
        except Exception:
            logging.debug("Unable to obtain download history.", exc_info=True)
            history = []
        self._download_history = history

    def _refresh_download_history(self) -> None:
        with self._download_window_lock:
            history_container = self._download_window_widgets.get("_history_container")
            window = self._download_window
        if history_container is None or window is None or not window.winfo_exists():
            return

        for child in list(history_container.winfo_children()):
            child.destroy()

        if not self._download_history:
            empty_label = ctk.CTkLabel(history_container, text="No recent downloads recorded.")
            empty_label.pack(anchor="w", padx=6, pady=4)
            return

        for entry in reversed(self._download_history):
            status = str(entry.get("status", "unknown")).upper()
            model_id = entry.get("model_id", "?")
            backend = entry.get("backend", "?")
            timestamp = entry.get("timestamp", "")
            header = ctk.CTkLabel(
                history_container,
                text=f"{timestamp} — {model_id} [{backend}] — {status}",
                anchor="w",
            )
            header.pack(fill="x", padx=6, pady=(4, 0))
            details_parts: list[str] = []
            message = entry.get("message") or ""
            details = entry.get("details") or ""
            target_dir = entry.get("target_dir") or details
            bytes_downloaded = entry.get("bytes_downloaded")
            throughput = entry.get("throughput_bps")
            duration = entry.get("duration_seconds")
            if bytes_downloaded:
                details_parts.append(f"Size {self._format_bytes(int(bytes_downloaded))}")
            if throughput:
                details_parts.append(f"Throughput {self._format_throughput(float(throughput))}")
            if duration:
                details_parts.append(f"Duration {self._format_eta(float(duration))}")
            if target_dir:
                details_parts.append(f"Target {target_dir}")
            if message:
                details_parts.append(message)
            info = " | ".join(details_parts) if details_parts else ""
            if info:
                info_label = ctk.CTkLabel(history_container, text=info, anchor="w")
                info_label.pack(fill="x", padx=8, pady=(0, 4))

    def _ensure_download_row(self, container, task_id: str, task: dict[str, Any]) -> None:
        widgets = self._download_window_widgets.get(task_id)
        if widgets is None:
            row = ctk.CTkFrame(container)
            row.pack(fill="x", pady=6)
            title = ctk.CTkLabel(row, text="", anchor="w")
            title.pack(fill="x")
            progress = ctk.CTkProgressBar(row)
            progress.pack(fill="x", pady=(2, 2))
            info = ctk.CTkLabel(row, text="", anchor="w")
            info.pack(fill="x")
            target = ctk.CTkLabel(row, text="", anchor="w")
            target.pack(fill="x")
            btn_frame = ctk.CTkFrame(row)
            btn_frame.pack(fill="x", pady=(6, 0))
            pause_btn = ctk.CTkButton(
                btn_frame,
                text="Pause",
                width=90,
                command=lambda tid=task_id: self._on_download_action(tid, "pause"),
            )
            pause_btn.pack(side="left", padx=4)
            resume_btn = ctk.CTkButton(
                btn_frame,
                text="Resume",
                width=90,
                command=lambda tid=task_id: self._on_download_action(tid, "resume"),
            )
            resume_btn.pack(side="left", padx=4)
            cancel_btn = ctk.CTkButton(
                btn_frame,
                text="Cancel",
                width=90,
                command=lambda tid=task_id: self._on_download_action(tid, "cancel"),
            )
            cancel_btn.pack(side="left", padx=4)
            widgets = {
                "frame": row,
                "title": title,
                "progress": progress,
                "info": info,
                "target": target,
                "pause": pause_btn,
                "resume": resume_btn,
                "cancel": cancel_btn,
            }
            self._download_window_widgets[task_id] = widgets

        title_label: ctk.CTkLabel = widgets["title"]
        progress_bar: ctk.CTkProgressBar = widgets["progress"]
        info_label: ctk.CTkLabel = widgets["info"]
        target_label: ctk.CTkLabel = widgets["target"]
        pause_btn: ctk.CTkButton = widgets["pause"]
        resume_btn: ctk.CTkButton = widgets["resume"]
        cancel_btn: ctk.CTkButton = widgets["cancel"]

        model_id = task.get("model_id", "?")
        backend = task.get("backend", "?")
        status = str(task.get("status", "queued"))
        stage = task.get("stage", "")
        message = task.get("message") or ""
        bytes_done = int(task.get("bytes_done") or 0)
        bytes_total = int(task.get("bytes_total") or 0)
        percent = task.get("percent")
        target_dir = task.get("target_dir") or ""
        eta_seconds = task.get("eta_seconds")
        throughput = task.get("throughput_bps")

        title_label.configure(text=f"{model_id} [{backend}] — {status.upper()}")
        if percent is None or percent <= 0:
            progress_bar.set(0.0)
        else:
            progress_bar.set(min(100.0, float(percent)) / 100.0)

        progress_parts = [f"{self._format_bytes(bytes_done)}"]
        if bytes_total:
            progress_parts.append(f"of {self._format_bytes(bytes_total)}")
        if throughput:
            progress_parts.append(f"@ {self._format_throughput(throughput)}")
        if eta_seconds:
            progress_parts.append(f"ETA {self._format_eta(float(eta_seconds))}")
        if stage and stage not in status:
            progress_parts.append(f"stage={stage}")
        info_text = " | ".join(progress_parts)
        if message:
            info_text = f"{info_text} — {message}" if info_text else message
        info_label.configure(text=info_text)

        target_label.configure(text=f"Target: {target_dir}" if target_dir else "Target: <pending>")

        if status in {"completed", "skipped"}:
            pause_btn.configure(state="disabled")
            resume_btn.configure(state="disabled")
            cancel_btn.configure(state="disabled")
        elif status == "running":
            pause_btn.configure(state="normal")
            resume_btn.configure(state="disabled")
            cancel_btn.configure(state="normal")
        elif status == "paused":
            pause_btn.configure(state="disabled")
            resume_btn.configure(state="normal")
            cancel_btn.configure(state="normal")
        elif status in {"cancelled", "timed_out", "error"}:
            pause_btn.configure(state="disabled")
            resume_btn.configure(state="disabled")
            cancel_btn.configure(state="disabled")
        else:
            pause_btn.configure(state="normal")
            resume_btn.configure(state="disabled")
            cancel_btn.configure(state="normal")

    def _on_download_action(self, task_id: str, action: str) -> None:
        core = getattr(self, "core_instance_ref", None)
        if core is None:
            return
        try:
            if action == "pause":
                core.pause_download_task(task_id)
            elif action == "resume":
                core.resume_download_task(task_id)
            elif action == "cancel":
                core.cancel_download_task(task_id)
        except Exception as exc:
            messagebox.showerror("Model Downloads", f"Unable to {action} task: {exc}")

    def _handle_download_progress(self, context: dict[str, Any] | None) -> None:
        if not context:
            return
        details = context.get("details") if isinstance(context, dict) else None
        tasks = []
        if isinstance(details, dict):
            tasks = details.get("tasks") or []
        if tasks:
            self._download_snapshot = tasks
        should_refresh_history = False
        missing_history = not self._download_history and isinstance(details, dict)
        if isinstance(details, dict):
            status = str(details.get("status") or "").lower()
            if status in {"completed", "skipped", "cancelled", "timed_out", "error", "success", "timeout"}:
                should_refresh_history = True
        refresh_history_view = should_refresh_history or missing_history
        if refresh_history_view:
            self._update_download_history_cache()
        if self._download_window and self._download_window.winfo_exists():
            self._refresh_download_window(self._download_snapshot)
            if refresh_history_view:
                self._refresh_download_history()

    @staticmethod
    def _format_bytes(value: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        amount = float(max(0, int(value)))
        for unit in units:
            if amount < 1024 or unit == units[-1]:
                return f"{amount:.2f} {unit}"
            amount /= 1024
        return f"{amount:.2f} PB"

    @staticmethod
    def _format_throughput(value: float) -> str:
        if value <= 0:
            return "0 B/s"
        return f"{UIManager._format_bytes(int(value))}/s"

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds <= 0:
            return "0s"
        minutes, secs = divmod(int(seconds), 60)
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"
        self._pending_key_var_name = None

    def _handle_detected_key(self, key: str | None) -> None:
        target_var_name = self._pending_key_var_name
        self._pending_key_var_name = None
        if not target_var_name:
            return

        previous_value = self._get_settings_meta(f"{target_var_name}::previous_key")
        var = self._get_settings_var(target_var_name)

        normalized_key: str | None = None
        if isinstance(key, str) and key.strip():
            normalized_key = key.strip().upper()

        if var is not None:
            try:
                if normalized_key:
                    var.set(normalized_key)
                elif previous_value is not None:
                    var.set(previous_value)
                else:
                    var.set("PRESS KEY...")
            except Exception:
                logging.debug(
                    "UIManager: falha ao atualizar variável de tecla detectada.",
                    exc_info=True,
                )

        self._set_settings_meta(f"{target_var_name}::previous_key", None)

        core = getattr(self, "core_instance_ref", None)
        if core and hasattr(core, "set_key_detection_callback"):
            try:
                core.set_key_detection_callback(None)
            except Exception:
                logging.debug(
                    "UIManager: falha ao limpar callback de detecção de tecla.",
                    exc_info=True,
                )

        if not normalized_key:
            window = self._get_settings_var("window")
            if window is not None:
                try:
                    messagebox.showinfo(
                        "Hotkey Detection",
                        "Nenhuma tecla detectada dentro do intervalo informado.",
                        parent=window,
                    )
                except Exception:
                    logging.debug(
                        "UIManager: falha ao exibir alerta de detecção de tecla.",
                        exc_info=True,
                    )

    def _start_key_detection_for(self, var_name: str) -> None:
        key_var = self._get_settings_var(var_name)
        window = self._get_settings_var("window")
        if key_var is None or window is None:
            return

        try:
            current_value = key_var.get()
        except Exception:
            current_value = None

        detection_target = "agent" if var_name == "agent_key_var" else "record"
        core = getattr(self, "core_instance_ref", None)
        if core and hasattr(core, "prepare_key_detection"):
            try:
                core.prepare_key_detection(detection_target, current_value=current_value)
            except Exception:
                logging.error("Falha ao preparar contexto de detecção de tecla.", exc_info=True)

        try:
            previous_value = key_var.get()
        except Exception:
            previous_value = None
        self._set_settings_meta(f"{var_name}::previous_key", previous_value)
        try:
            key_var.set("PRESS KEY...")
        except Exception:
            self._set_settings_meta(f"{var_name}::previous_key", None)
            return
        try:
            window.update_idletasks()
        except Exception:
            pass
        self._pending_key_var_name = var_name
        core = getattr(self, "core_instance_ref", None)
        if not core or not hasattr(core, "set_key_detection_callback") or not hasattr(core, "start_key_detection_thread"):
            if key_var is not None and previous_value is not None:
                try:
                    key_var.set(previous_value)
                except Exception:
                    pass
            self._set_settings_meta(f"{var_name}::previous_key", None)
            self._pending_key_var_name = None
            logging.error("UIManager: núcleo indisponível para iniciar detecção de tecla.")
            return

        try:
            core.set_key_detection_callback(self._handle_detected_key)
            core.start_key_detection_thread(timeout=5.0)
        except Exception:
            logging.error(
                "UIManager: falha ao iniciar detecção de tecla.",
                exc_info=True,
            )
            if previous_value is not None:
                try:
                    key_var.set(previous_value)
                except Exception:
                    pass
            self._set_settings_meta(f"{var_name}::previous_key", None)
            self._pending_key_var_name = None

    def _safe_get_int(self, var, field_name: str, parent):
        """Wrapper em torno de :func:`safe_get_int` para manter logs centralizados."""

        return safe_get_int(var, field_name, parent)

    def _safe_get_float(self, var, field_name: str, parent):
        """Wrapper em torno de :func:`safe_get_float` para manter logs centralizados."""

        return safe_get_float(var, field_name, parent)

    def _open_directory_from_tray(self, target: Path) -> None:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logging.error("UIManager: falha ao garantir diretório %s: %s", target, exc, exc_info=True)
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)], close_fds=True)
            else:
                subprocess.Popen(["xdg-open", str(target)], close_fds=True)
        except Exception as exc:
            logging.error("UIManager: falha ao abrir %s: %s", target, exc, exc_info=True)

    def open_logs_directory(self) -> None:
        target = get_log_directory()
        if target is None:
            target = Path(os.getenv("WHISPER_LOG_DIR", "logs")).expanduser()
        self._open_directory_from_tray(target)

    def open_docs_directory(self) -> None:
        self._open_directory_from_tray(Path("docs"))

    def _resolve_initial_value(
        self,
        config_key: str,
        *,
        var_name: str | None = None,
        getter: Callable[[], Any] | None = None,
        default: Any | None = None,
        coerce: Callable[[Any], Any] | None = None,
        transform: Callable[[Any], Any] | None = None,
        allowed: Iterable[Any] | None = None,
        sensitive: bool = False,
    ) -> Any:
        """Resolve o valor inicial de um campo de configuração.

        A função consulta o ``ConfigManager`` (ou um ``getter`` explícito), aplica
        coerção, validação por conjunto permitido e qualquer transformação
        opcional. Em caso de erro, o valor retornado recai para o default definido
        no ``DEFAULT_CONFIG``.
        """

        label = var_name or config_key

        def _mask(value: Any) -> Any:
            if not sensitive:
                return value
            if isinstance(value, str):
                return "" if value == "" else "<hidden>"
            if value in (None, 0):
                return value
            return "<hidden>"

        fallback = default if default is not None else DEFAULT_CONFIG.get(config_key)

        try:
            value = getter() if getter is not None else self.config_manager.get(config_key)
        except Exception as exc:  # pragma: no cover - salvaguarda defensiva
            logging.warning(
                "UIManager: failed to read config '%s': %s. Using fallback.",
                label,
                exc,
                exc_info=True,
            )
            value = fallback

        if value is None and fallback is not None:
            value = fallback

        if coerce is not None:
            try:
                value = coerce(value)
            except Exception as exc:
                logging.warning(
                    "UIManager: invalid value for '%s': %r (%s). Using fallback %r.",
                    label,
                    _mask(value),
                    exc,
                    _mask(fallback),
                )
                value = fallback
                if value is not None:
                    try:
                        value = coerce(value)
                    except Exception:
                        pass

        if allowed is not None and value not in allowed:
            logging.warning(
                "UIManager: unexpected value for '%s': %r (allowed=%s). Using fallback %r.",
                label,
                _mask(value),
                allowed,
                _mask(fallback),
            )
            value = fallback
            if coerce is not None and value is not None:
                try:
                    value = coerce(value)
                except Exception:
                    pass

        if transform is not None:
            try:
                value = transform(value)
            except Exception as exc:
                logging.warning(
                    "UIManager: transform failed for '%s' with %r: %s. Using fallback.",
                    label,
                    _mask(value),
                    exc,
                )
                value = fallback
                if transform is not None and value is not None:
                    try:
                        value = transform(value)
                    except Exception:
                        pass

        if var_name:
            self._set_settings_meta(f"initial_{var_name}", value)

        return value

    def _update_text_correction_fields(self) -> None:
        enabled_var = self._get_settings_var("text_correction_enabled_var")
        if enabled_var is None:
            return
        try:
            enabled = bool(enabled_var.get())
        except Exception:
            enabled = False
        state = "normal" if enabled else "disabled"
        for widget_name in [
            "service_menu",
            "openrouter_key_entry",
            "openrouter_model_entry",
            "gemini_key_entry",
            "gemini_model_menu",
            "agent_model_menu",
            "gemini_prompt_correction_textbox",
            "agentico_prompt_textbox",
            "gemini_models_textbox",
        ]:
            widget = self._get_settings_var(widget_name)
            if widget is None:
                continue
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _on_service_menu_change(self, choice: str) -> None:
        mapping = self._get_settings_meta("service_display_map", {})
        service_var = self._get_settings_var("text_correction_service_var")
        label_var = self._get_settings_var("text_correction_service_label_var")
        if label_var is not None:
            try:
                label_var.set(choice)
            except Exception:
                pass
        if service_var is not None:
            try:
                service_var.set(mapping.get(choice, SERVICE_NONE))
            except Exception:
                pass
        self._update_text_correction_fields()

    def _update_chunk_length_state(self) -> None:
        mode_var = self._get_settings_var("chunk_length_mode_var")
        entry = self._get_settings_var("chunk_len_entry")
        if mode_var is None or entry is None:
            return
        try:
            mode_val = str(mode_var.get()).lower()
        except Exception:
            mode_val = "manual"
        state = "normal" if mode_val == "manual" else "disabled"
        try:
            entry.configure(state=state)
        except Exception:
            pass

    def _on_chunk_mode_change(self, _choice: str) -> None:
        self._update_chunk_length_state()

    def _derive_backend_from_model(self, model_ref: str) -> Optional[str]:
        display_to_id = self._get_settings_meta("display_to_id", {})
        catalog = self._get_settings_meta("catalog", [])
        model_id = display_to_id.get(model_ref, model_ref)
        entry = next((m for m in catalog if m.get("id") == model_id), None)
        if not entry:
            cache_var = self._get_settings_var("asr_cache_dir_var")
            cache_dir = cache_var.get() if cache_var is not None else ""
            try:
                installed = self.model_manager.list_installed(cache_dir)
            except OSError:
                installed = []
            entry = next((m for m in installed if m.get("id") == model_id), None)
        backend = entry.get("backend") if entry else None
        if not backend:
            return None
        normalized = _backend_display_value_global(backend)
        return normalized or None

    def _update_install_button_state(self) -> None:
        model_var = self._get_settings_var("asr_model_id_var")
        install_button = self._get_settings_var("install_button")
        quant_menu = self._get_settings_var("asr_ct2_menu")
        if model_var is None or install_button is None:
            return
        backend = self._derive_backend_from_model(model_var.get())
        try:
            install_button.configure(state="normal" if backend else "disabled")
        except Exception:
            pass
        if quant_menu is not None:
            try:
                quant_menu.configure(state="normal" if backend == "ctranslate2" else "disabled")
            except Exception:
                pass

    def _update_model_info(self, model_ref: str) -> None:
        display_to_id = self._get_settings_meta("display_to_id", {})
        cache_var = self._get_settings_var("asr_cache_dir_var")
        label = self._get_settings_var("model_size_label")
        if label is None:
            return
        model_id = display_to_id.get(model_ref, model_ref)
        try:
            d_bytes, d_files = self.model_manager.get_model_download_size(model_id)
            d_mb = d_bytes / (1024 * 1024)
            download_text = f"{d_mb:.1f} MB ({d_files} files)"
        except Exception:
            download_text = "?"

        installed_models = []
        if cache_var is not None:
            try:
                installed_models = self.model_manager.list_installed(cache_var.get())
            except OSError:
                messagebox.showerror(
                    "Settings",
                    "Unable to list installed ASR models. Please verify the cache directory.",
                )
        entry = next((m for m in installed_models if m.get("id") == model_id), None)
        if entry:
            i_bytes, i_files = self.model_manager.get_installed_size(entry.get("path"))
            i_mb = i_bytes / (1024 * 1024)
            installed_text = f"{i_mb:.1f} MB ({i_files} files)"
        else:
            installed_text = "-"
        try:
            label.configure(text=f"Download: {download_text} | Installed: {installed_text}")
        except Exception:
            pass

    def _on_backend_change(self, choice: str) -> None:
        backend_var = self._get_settings_var("asr_backend_var")
        if backend_var is not None:
            try:
                backend_var.set(choice)
            except Exception:
                pass
        self._update_install_button_state()
        model_var = self._get_settings_var("asr_model_id_var")
        if model_var is not None:
            self._update_model_info(model_var.get())

    def _on_model_change(self, choice_display: str) -> None:
        id_to_display = self._get_settings_meta("id_to_display", {})
        display_to_id = self._get_settings_meta("display_to_id", {})
        model_id = display_to_id.get(choice_display, choice_display)
        model_var = self._get_settings_var("asr_model_id_var")
        display_var = self._get_settings_var("asr_model_display_var")
        backend_menu = self._get_settings_var("asr_backend_menu")
        backend_var = self._get_settings_var("asr_backend_var")
        if model_var is not None:
            try:
                model_var.set(model_id)
            except Exception:
                pass
        if display_var is not None:
            try:
                display_var.set(id_to_display.get(model_id, model_id))
            except Exception:
                pass
        backend = self._derive_backend_from_model(model_id)
        if backend and backend_var is not None:
            try:
                backend_var.set(backend)
            except Exception:
                pass
            if backend_menu is not None:
                try:
                    backend_menu.configure(state="disabled")
                except Exception:
                    pass
        elif backend_menu is not None:
            try:
                backend_menu.configure(state="normal")
            except Exception:
                pass
        backend_value = None
        if backend_var is not None:
            try:
                backend_value = backend_var.get()
            except Exception:
                backend_value = None
        elif backend is not None:
            backend_value = backend

        current_model = self.config_manager.get_asr_model_id()
        current_backend = self.config_manager.get_asr_backend()
        config_changed = False

        if model_id != current_model:
            self.config_manager.set_asr_model_id(model_id)
            config_changed = True
        if backend_value is not None and backend_value != current_backend:
            self.config_manager.set_asr_backend(backend_value)
            config_changed = True

        if config_changed:
            self.config_manager.save_config()
        if backend_var is not None:
            self._on_backend_change(backend_var.get())
        self._update_model_info(model_id)

    def _install_selected_model(self) -> None:
        model_var = self._get_settings_var("asr_model_id_var")
        cache_var = self._get_settings_var("asr_cache_dir_var")
        if model_var is None or cache_var is None:
            return
        cache_dir = cache_var.get()
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"ASR cache directory is invalid:\n{exc}")
            return
        model_id = model_var.get()
        backend = self._derive_backend_from_model(model_id)
        if backend is None:
            messagebox.showerror("Model", "Unable to determine backend for selected model.")
            return
        try:
            size_bytes, file_count = self.model_manager.get_model_download_size(model_id)
            size_gb = size_bytes / (1024 ** 3)
            detail = f"approximately {size_gb:.2f} GB ({file_count} files)"
        except Exception:
            detail = "an unspecified size"
        if not messagebox.askyesno(
            "Model Download",
            f"Model '{model_id}' will download {detail}.\nContinue?",
        ):
            return
        try:
            compute_type_var = self._get_settings_var("asr_ct2_compute_type_var")
            quant = compute_type_var.get() if compute_type_var is not None else None
            download_result = self.core_instance_ref.download_model_with_ui(
                model_id=model_id,
                backend=backend,
                cache_dir=cache_dir,
                quant=quant if backend == "ctranslate2" else None,
                reason="settings",
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            messagebox.showerror("Model", f"Download failed: {exc}")
            return

        status = download_result.get("status")
        if status == "success":
            self._update_model_info(model_id)
            messagebox.showinfo("Model", "Download completed.")
        elif status == "cancelled":
            messagebox.showinfo("Model", "Download canceled.")
        elif status == "error":
            message = download_result.get("error") or "Unknown error"
            messagebox.showerror("Model", f"Download failed: {message}")

    def _reload_current_model(self) -> None:
        handler = getattr(self.core_instance_ref, "transcription_handler", None)
        if handler:
            handler.reload_asr()

    def _apply_settings_from_ui(self) -> None:
        settings_win = self._get_settings_var("window")
        if self._get_core_state() in ["RECORDING", "TRANSCRIBING", "LOADING_MODEL"]:
            messagebox.showwarning(
                "Apply Settings",
                "Cannot apply while recording/transcribing/loading model.",
                parent=settings_win,
            )
            return

        def _var(name: str):
            return self._get_settings_var(name)

        detected_key_var = _var("detected_key_var")
        agent_key_var = _var("agent_key_var")
        mode_var = _var("mode_var")
        auto_paste_var = _var("auto_paste_var")
        agent_model_var = _var("agent_model_var")
        hotkey_stability_var = _var("hotkey_stability_service_enabled_var")
        sound_enabled_var = _var("sound_enabled_var")
        sound_frequency_var = _var("sound_frequency_var")
        sound_duration_var = _var("sound_duration_var")
        sound_volume_var = _var("sound_volume_var")
        text_correction_enabled_var = _var("text_correction_enabled_var")
        text_correction_service_var = _var("text_correction_service_var")
        openrouter_api_key_var = _var("openrouter_api_key_var")
        openrouter_model_var = _var("openrouter_model_var")
        gemini_api_key_var = _var("gemini_api_key_var")
        gemini_model_var = _var("gemini_model_var")
        gemini_prompt_textbox = _var("gemini_prompt_correction_textbox")
        agentico_prompt_textbox = _var("agentico_prompt_textbox")
        gemini_models_textbox = _var("gemini_models_textbox")
        batch_size_var = _var("batch_size_var")
        min_transcription_duration_var = _var("min_transcription_duration_var")
        min_record_duration_var = _var("min_record_duration_var")
        use_vad_var = _var("use_vad_var")
        vad_threshold_var = _var("vad_threshold_var")
        vad_silence_duration_var = _var("vad_silence_duration_var")
        vad_pre_speech_padding_ms_var = _var("vad_pre_speech_padding_ms_var")
        vad_post_speech_padding_ms_var = _var("vad_post_speech_padding_ms_var")
        save_temp_recordings_var = _var("save_temp_recordings_var")
        display_transcripts_var = _var("display_transcripts_var")
        storage_root_dir_var = _var("storage_root_dir_var")
        recordings_dir_var = _var("recordings_dir_var")
        record_storage_mode_var = _var("record_storage_mode_var")
        max_memory_seconds_mode_var = _var("max_memory_seconds_mode_var")
        max_memory_seconds_var = _var("max_memory_seconds_var")
        chunk_length_mode_var = _var("chunk_length_mode_var")
        chunk_length_sec_var = _var("chunk_length_sec_var")
        launch_at_startup_var = _var("launch_at_startup_var")
        asr_backend_var = _var("asr_backend_var")
        asr_model_id_var = _var("asr_model_id_var")
        asr_compute_device_var = _var("asr_compute_device_var")
        asr_ct2_compute_type_var = _var("asr_ct2_compute_type_var")
        models_storage_dir_var = _var("models_storage_dir_var")
        deps_install_dir_var = _var("deps_install_dir_var")
        asr_cache_dir_var = _var("asr_cache_dir_var")
        recordings_dir_var = _var("recordings_dir_var")
        max_parallel_downloads_var = _var("max_parallel_downloads_var")

        if detected_key_var is None or mode_var is None or auto_paste_var is None:
            return

        record_key_value = detected_key_var.get()
        key_to_apply = record_key_value.lower() if record_key_value != "PRESS KEY..." else self.config_manager.get("record_key")
        agent_key_value = agent_key_var.get() if agent_key_var else self.config_manager.get("agent_key")
        agent_key_to_apply = agent_key_value.lower() if agent_key_value != "PRESS KEY..." else self.config_manager.get("agent_key")
        mode_to_apply_raw = mode_var.get() if mode_var else ""
        mode_to_apply = (mode_to_apply_raw or "").strip().lower()
        if mode_to_apply == "hold":
            mode_to_apply = "press"
        if mode_to_apply not in {"toggle", "press"}:
            mode_to_apply = DEFAULT_CONFIG.get("record_mode", "toggle")
        auto_paste_to_apply = bool(auto_paste_var.get())
        agent_model_to_apply = agent_model_var.get() if agent_model_var else self.config_manager.get("gemini_agent_model")
        hotkey_stability_to_apply = bool(hotkey_stability_var.get()) if hotkey_stability_var else True
        sound_enabled_to_apply = bool(sound_enabled_var.get()) if sound_enabled_var else True

        sound_freq_to_apply = self._safe_get_int(sound_frequency_var, "Frequência do Som", settings_win)
        if sound_freq_to_apply is None:
            return
        sound_duration_to_apply = self._safe_get_float(sound_duration_var, "Sound Duration", settings_win)
        if sound_duration_to_apply is None:
            return
        sound_volume_to_apply = self._safe_get_float(sound_volume_var, "Volume do Som", settings_win)
        if sound_volume_to_apply is None:
            return

        text_correction_enabled_to_apply = bool(text_correction_enabled_var.get()) if text_correction_enabled_var else False
        text_correction_service_to_apply = text_correction_service_var.get() if text_correction_service_var else SERVICE_NONE
        openrouter_api_key_to_apply = openrouter_api_key_var.get() if openrouter_api_key_var else ""
        openrouter_model_to_apply = openrouter_model_var.get() if openrouter_model_var else ""
        gemini_api_key_to_apply = gemini_api_key_var.get() if gemini_api_key_var else ""
        gemini_model_to_apply = gemini_model_var.get() if gemini_model_var else ""
        gemini_prompt_correction_to_apply = (
            gemini_prompt_textbox.get("1.0", "end-1c") if gemini_prompt_textbox else self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY)
        )
        agentico_prompt_to_apply = (
            agentico_prompt_textbox.get("1.0", "end-1c") if agentico_prompt_textbox else self.config_manager.get("prompt_agentico")
        )

        batch_size_to_apply = self._safe_get_int(batch_size_var, "Batch Size", settings_win)
        if batch_size_to_apply is None:
            return
        min_transcription_duration_to_apply = self._safe_get_float(min_transcription_duration_var, "Minimum Transcription Duration", settings_win)
        if min_transcription_duration_to_apply is None:
            return
        min_record_duration_to_apply = self._safe_get_float(min_record_duration_var, "Minimum Recording Duration", settings_win)
        if min_record_duration_to_apply is None:
            return

        use_vad_to_apply = bool(use_vad_var.get()) if use_vad_var else False
        vad_threshold_to_apply = self._safe_get_float(vad_threshold_var, "Limiar do VAD", settings_win)
        if vad_threshold_to_apply is None:
            return
        vad_silence_duration_to_apply = self._safe_get_float(vad_silence_duration_var, "Silence Duration", settings_win)
        if vad_silence_duration_to_apply is None:
            return

        pre_padding_raw = vad_pre_speech_padding_ms_var.get() if vad_pre_speech_padding_ms_var else ""
        if isinstance(pre_padding_raw, str) and not pre_padding_raw.strip():
            vad_pre_speech_padding_ms_to_apply = int(self.config_manager.get(
                VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
                DEFAULT_CONFIG.get(VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY, 150),
            ))
        else:
            vad_pre_speech_padding_ms_to_apply = self._safe_get_int(
                vad_pre_speech_padding_ms_var,
                "Pre-speech Padding",
                settings_win,
            )
            if vad_pre_speech_padding_ms_to_apply is None:
                return

        post_padding_raw = vad_post_speech_padding_ms_var.get() if vad_post_speech_padding_ms_var else ""
        if isinstance(post_padding_raw, str) and not post_padding_raw.strip():
            vad_post_speech_padding_ms_to_apply = int(self.config_manager.get(
                VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
                DEFAULT_CONFIG.get(VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY, 300),
            ))
        else:
            vad_post_speech_padding_ms_to_apply = self._safe_get_int(
                vad_post_speech_padding_ms_var,
                "Post-speech Padding",
                settings_win,
            )
            if vad_post_speech_padding_ms_to_apply is None:
                return

        save_temp_recordings_to_apply = bool(save_temp_recordings_var.get()) if save_temp_recordings_var else False
        display_transcripts_to_apply = bool(display_transcripts_var.get()) if display_transcripts_var else False
        record_storage_mode_to_apply = record_storage_mode_var.get() if record_storage_mode_var else "auto"
        max_memory_seconds_mode_to_apply = max_memory_seconds_mode_var.get() if max_memory_seconds_mode_var else "manual"
        max_memory_seconds_to_apply = self._safe_get_float(max_memory_seconds_var, "Max Memory Retention", settings_win)
        if max_memory_seconds_to_apply is None:
            return
        chunk_length_mode_to_apply = chunk_length_mode_var.get() if chunk_length_mode_var else "manual"
        chunk_length_sec_to_apply = self._safe_get_float(chunk_length_sec_var, "Chunk Length", settings_win)
        if chunk_length_sec_to_apply is None:
            return
        launch_at_startup_to_apply = bool(launch_at_startup_var.get()) if launch_at_startup_var else False
        asr_backend_to_apply = asr_backend_var.get() if asr_backend_var else self.config_manager.get_asr_backend()
        asr_model_id_to_apply = asr_model_id_var.get() if asr_model_id_var else self.config_manager.get_asr_model_id()
        asr_compute_device_to_apply = "auto"
        asr_ct2_compute_type_to_apply = asr_ct2_compute_type_var.get() if asr_ct2_compute_type_var else self.config_manager.get_asr_ct2_compute_type()
        storage_root_dir_to_apply_raw = storage_root_dir_var.get().strip() if storage_root_dir_var else self.config_manager.get_storage_root_dir()
        if not storage_root_dir_to_apply_raw:
            storage_root_dir_to_apply_raw = self.config_manager.get_storage_root_dir()
        try:
            storage_root_path = Path(storage_root_dir_to_apply_raw).expanduser()
            storage_root_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"Storage root directory is invalid:\n{exc}", parent=settings_win)
            return
        storage_root_dir_to_apply = str(storage_root_path)

        models_storage_dir_raw = models_storage_dir_var.get().strip() if models_storage_dir_var else ""
        if not models_storage_dir_raw:
            models_storage_path = storage_root_path
        else:
            try:
                models_storage_path = Path(models_storage_dir_raw).expanduser()
            except Exception as exc:
                messagebox.showerror("Invalid Path", f"Models storage directory is invalid:\n{exc}", parent=settings_win)
                return
        try:
            models_storage_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"Models storage directory is invalid:\n{exc}", parent=settings_win)
            return
        models_storage_dir_to_apply = str(models_storage_path)

        deps_install_dir_raw = deps_install_dir_var.get().strip() if deps_install_dir_var else ""
        if not deps_install_dir_raw:
            deps_install_dir_raw = self.config_manager.get_deps_install_dir()
        try:
            deps_install_path = Path(deps_install_dir_raw).expanduser()
        except Exception as exc:
            messagebox.showerror(
                "Invalid Path",
                f"Dependencies directory is invalid:\n{exc}",
                parent=settings_win,
            )
            return
        try:
            deps_install_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror(
                "Invalid Path",
                f"Dependencies directory is invalid:\n{exc}",
                parent=settings_win,
            )
            return
        deps_install_dir_to_apply = str(deps_install_path)

        recordings_dir_raw = recordings_dir_var.get().strip() if recordings_dir_var else ""
        if not recordings_dir_raw:
            recordings_path = storage_root_path / "recordings"
        else:
            try:
                recordings_path = Path(recordings_dir_raw).expanduser()
            except Exception as exc:
                messagebox.showerror("Invalid Path", f"Recordings directory is invalid:\n{exc}", parent=settings_win)
                return
        try:
            recordings_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"Recordings directory is invalid:\n{exc}", parent=settings_win)
            return
        recordings_dir_to_apply = str(recordings_path)

        models_storage_dir_raw = models_storage_dir_var.get().strip() if models_storage_dir_var else ""
        if not models_storage_dir_raw:
            models_storage_dir_raw = self.config_manager.get_models_storage_dir()
        try:
            models_storage_dir_path = Path(models_storage_dir_raw).expanduser()
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"Models storage directory is invalid:\n{exc}", parent=settings_win)
            return
        try:
            models_storage_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"Models storage directory is invalid:\n{exc}", parent=settings_win)
            return
        models_storage_dir_to_apply = str(models_storage_dir_path)

        asr_cache_dir_raw = asr_cache_dir_var.get().strip() if asr_cache_dir_var else ""
        if not asr_cache_dir_raw:
            asr_cache_path = storage_root_path / "asr"
        else:
            try:
                asr_cache_path = Path(asr_cache_dir_raw).expanduser()
            except Exception as exc:
                messagebox.showerror(
                    "Invalid Path",
                    f"ASR cache directory is invalid:\n{exc}",
                    parent=settings_win,
                )
                return
        try:
            asr_cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror(
                "Invalid Path",
                f"ASR cache directory is invalid:\n{exc}",
                parent=settings_win,
            )
            return
        asr_cache_dir_to_apply = str(asr_cache_path)

        selected_device_str = asr_compute_device_var.get() if asr_compute_device_var else "Auto-select (Recommended)"
        gpu_index_to_apply = -1
        if "Force CPU" in selected_device_str:
            asr_compute_device_to_apply = "cpu"
        elif selected_device_str.startswith("GPU"):
            asr_compute_device_to_apply = "cuda"
            try:
                gpu_index_to_apply = int(selected_device_str.split(":")[0].replace("GPU", "").strip())
            except (ValueError, IndexError):
                messagebox.showerror("Invalid Value", "Invalid GPU index.", parent=settings_win)
                return

        models_text = gemini_models_textbox.get("1.0", "end-1c") if gemini_models_textbox else ""
        new_models_list = [line.strip() for line in models_text.split("\n") if line.strip()]
        if not new_models_list:
            messagebox.showwarning("Invalid Value", "The model list cannot be empty. Please add at least one model.", parent=settings_win)
            return

        max_parallel_downloads_to_apply = self.config_manager.get_max_parallel_downloads()
        if max_parallel_downloads_var is not None:
            try:
                candidate = int(max_parallel_downloads_var.get())
            except (TypeError, ValueError):
                candidate = max_parallel_downloads_to_apply
            else:
                if candidate < 1:
                    candidate = 1
                elif candidate > 8:
                    candidate = 8
            max_parallel_downloads_to_apply = candidate
        self.config_manager.set_max_parallel_downloads(max_parallel_downloads_to_apply)
        core_ref = getattr(self, "core_instance_ref", None)
        if core_ref is not None and hasattr(core_ref, "model_download_controller"):
            try:
                core_ref.model_download_controller.update_parallel_limit(max_parallel_downloads_to_apply)
            except Exception:
                logging.debug(
                    "Unable to propagate max_parallel_downloads to controller.",
                    exc_info=True,
                )

        self.core_instance_ref.apply_settings_from_external(
            new_key=key_to_apply,
            new_mode=mode_to_apply,
            new_auto_paste=auto_paste_to_apply,
            new_sound_enabled=sound_enabled_to_apply,
            new_sound_frequency=sound_freq_to_apply,
            new_sound_duration=sound_duration_to_apply,
            new_sound_volume=sound_volume_to_apply,
            new_agent_key=agent_key_to_apply,
            new_text_correction_enabled=text_correction_enabled_to_apply,
            new_text_correction_service=text_correction_service_to_apply,
            new_openrouter_api_key=openrouter_api_key_to_apply,
            new_openrouter_model=openrouter_model_to_apply,
            new_gemini_api_key=gemini_api_key_to_apply,
            new_gemini_model=gemini_model_to_apply,
            new_gemini_prompt=gemini_prompt_correction_to_apply,
            prompt_agentico=agentico_prompt_to_apply,
            new_agent_model=agent_model_to_apply,
            new_gemini_model_options=new_models_list,
            new_batch_size=batch_size_to_apply,
            new_gpu_index=gpu_index_to_apply,
            new_hotkey_stability_service_enabled=hotkey_stability_to_apply,
            new_min_transcription_duration=min_transcription_duration_to_apply,
            new_min_record_duration=min_record_duration_to_apply,
            new_save_temp_recordings=save_temp_recordings_to_apply,
            new_record_storage_mode=record_storage_mode_to_apply,
            new_max_memory_seconds_mode=max_memory_seconds_mode_to_apply,
            new_max_memory_seconds=max_memory_seconds_to_apply,
            new_use_vad=use_vad_to_apply,
            new_vad_threshold=vad_threshold_to_apply,
            new_vad_silence_duration=vad_silence_duration_to_apply,
            new_vad_pre_speech_padding_ms=vad_pre_speech_padding_ms_to_apply,
            new_vad_post_speech_padding_ms=vad_post_speech_padding_ms_to_apply,
            new_display_transcripts_in_terminal=display_transcripts_to_apply,
            new_launch_at_startup=launch_at_startup_to_apply,
            new_chunk_length_mode=chunk_length_mode_to_apply,
            new_chunk_length_sec=float(chunk_length_sec_to_apply),
            new_asr_backend=asr_backend_to_apply,
            new_asr_model_id=asr_model_id_to_apply,
            new_asr_compute_device=asr_compute_device_to_apply,
            new_asr_ct2_compute_type=asr_ct2_compute_type_to_apply,
            new_models_storage_dir=models_storage_dir_to_apply,
            new_deps_install_dir=deps_install_dir_to_apply,
            new_asr_cache_dir=asr_cache_dir_to_apply,
            new_storage_root_dir=storage_root_dir_to_apply,
            new_recordings_dir=recordings_dir_to_apply,
        )
        self._close_settings_window()

    def _restore_default_settings(self) -> None:
        settings_win = self._get_settings_var("window")

        confirm_kwargs: Dict[str, Any] = {}
        if settings_win is not None:
            confirm_kwargs["parent"] = settings_win
        if not messagebox.askyesno(
            "Restore Defaults",
            "Are you sure you want to restore all settings to their default values?",
            **confirm_kwargs,
        ):
            return

        try:
            sanitized_defaults, changed_keys = self.config_manager.reset_to_defaults()
        except Exception as exc:  # pragma: no cover - falha defensiva
            logging.error("UIManager: erro ao resetar configurações para o padrão.", exc_info=True)
            messagebox.showerror(
                "Restore Defaults",
                "Falha ao restaurar as configurações padrão. Verifique os logs para detalhes.",
                **confirm_kwargs,
            )
            return

        if changed_keys:
            logging.info(
                "UIManager: chaves redefinidas para padrão: %s",
                ", ".join(sorted(changed_keys)),
            )

        payload = dict(sanitized_defaults)
        try:
            self.core_instance_ref.apply_settings_from_external(
                force_reload=True,
                forced_keys=set(payload.keys()) if payload else None,
                **payload,
            )
        except Exception:
            logging.error(
                "UIManager: erro ao reaplicar configuração após reset para padrões.",
                exc_info=True,
            )
            messagebox.showerror(
                "Restore Defaults",
                "As configurações foram redefinidas, mas não foi possível reconfigurar o aplicativo."
                " Reinicie a aplicação manualmente.",
                **confirm_kwargs,
            )
            return

        self._close_settings_window()

        messagebox.showinfo(
            "Defaults Restored",
            (
                "As configurações foram restauradas para os valores padrão.\n"
                "Reabra a janela de configurações para visualizar os valores atualizados."
            ),
            parent=self.main_tk_root,
        )

    def _reset_asr_settings(self) -> None:
        id_to_display = self._get_settings_meta("id_to_display", {})
        default_model_id = DEFAULT_CONFIG["asr_model_id"]
        default_display = id_to_display.get(default_model_id, default_model_id)

        def _set_var(name: str, value: Any) -> None:
            var = self._get_settings_var(name)
            if var is not None:
                try:
                    var.set(value)
                except Exception:
                    pass

        _set_var("asr_model_id_var", default_model_id)
        _set_var("asr_model_display_var", default_display)
        model_menu = self._get_settings_var("asr_model_menu")
        if model_menu is not None:
            try:
                model_menu.set(default_display)
            except Exception:
                pass

        _set_var("asr_backend_var", DEFAULT_CONFIG["asr_backend"])
        backend_menu = self._get_settings_var("asr_backend_menu")
        if backend_menu is not None:
            try:
                backend_menu.set(DEFAULT_CONFIG["asr_backend"])
            except Exception:
                pass

        _set_var("asr_ct2_compute_type_var", DEFAULT_CONFIG["asr_ct2_compute_type"])
        ct2_menu = self._get_settings_var("asr_ct2_menu")
        if ct2_menu is not None:
            try:
                ct2_menu.set(DEFAULT_CONFIG["asr_ct2_compute_type"])
            except Exception:
                pass

        _set_var("max_parallel_downloads_var", str(DEFAULT_CONFIG.get("max_parallel_downloads", 1)))

        _set_var("asr_cache_dir_var", DEFAULT_CONFIG["asr_cache_dir"])
        _set_var("storage_root_dir_var", DEFAULT_CONFIG[STORAGE_ROOT_DIR_CONFIG_KEY])
        _set_var("recordings_dir_var", DEFAULT_CONFIG[RECORDINGS_DIR_CONFIG_KEY])
        _set_var("python_packages_dir_var", DEFAULT_CONFIG[PYTHON_PACKAGES_DIR_CONFIG_KEY])
        _set_var("vad_models_dir_var", DEFAULT_CONFIG[VAD_MODELS_DIR_CONFIG_KEY])
        _set_var("hf_cache_dir_var", DEFAULT_CONFIG[HF_CACHE_DIR_CONFIG_KEY])

        self.config_manager.set_asr_model_id(default_model_id)
        self.config_manager.set_asr_backend(DEFAULT_CONFIG["asr_backend"])
        self.config_manager.set_asr_ct2_compute_type(DEFAULT_CONFIG["asr_ct2_compute_type"])
        self.config_manager.set_models_storage_dir(DEFAULT_CONFIG["models_storage_dir"])
        self.config_manager.set_asr_cache_dir(DEFAULT_CONFIG["asr_cache_dir"])
        self.config_manager.set_storage_root_dir(DEFAULT_CONFIG[STORAGE_ROOT_DIR_CONFIG_KEY])
        self.config_manager.set_recordings_dir(DEFAULT_CONFIG[RECORDINGS_DIR_CONFIG_KEY])
        self.config_manager.set_python_packages_dir(DEFAULT_CONFIG[PYTHON_PACKAGES_DIR_CONFIG_KEY])
        self.config_manager.set_vad_models_dir(DEFAULT_CONFIG[VAD_MODELS_DIR_CONFIG_KEY])
        self.config_manager.set_hf_cache_dir(DEFAULT_CONFIG[HF_CACHE_DIR_CONFIG_KEY])
        self.config_manager.save_config()

        backend_var = self._get_settings_var("asr_backend_var")
        backend_choice = backend_var.get() if backend_var is not None else DEFAULT_CONFIG["asr_backend"]
        self._on_backend_change(backend_choice)
        self._update_model_info(default_model_id)
        self._update_install_button_state()

    # Methods for the live transcription window
    def _show_live_transcription_window(self):
        # This functionality has been disabled at the user's request.
        pass
        # if self.live_window and self.live_window.winfo_exists(): return
        # self.live_window = ctk.CTkToplevel(self.main_tk_root)
        # self.live_window.overrideredirect(True)
        # self.live_window.geometry("400x150+50+50")
        # self.live_window.attributes("-alpha", 0.85)
        # self.live_window.attributes("-topmost", True)
        # self.live_textbox = ctk.CTkTextbox(self.live_window, wrap="word", activate_scrollbars=True)
        # self.live_textbox.pack(fill="both", expand=True)
        # self.live_textbox.insert("end", "Listening...")

    def _update_live_transcription(self, new_text):
        if self.live_textbox and self.live_window.winfo_exists():
            if self.live_textbox.get("1.0", "end-1c") == "Listening...":
                self.live_textbox.delete("1.0", "end")
            self.live_textbox.insert("end", new_text + " ")
            self.live_textbox.see("end")

    def _close_live_transcription_window(self):
        if self.live_window:
            self.live_window.destroy()
            self.live_window = None
            self.live_textbox = None
    
            # Assign methods to the instance
            self.show_live_transcription_window = self._show_live_transcription_window
            self.update_live_transcription = self._update_live_transcription
            self.close_live_transcription_window = self._close_live_transcription_window
            self.update_live_transcription_threadsafe = self.update_live_transcription_threadsafe

    # Thread-safe method to update live transcription
    def update_live_transcription_threadsafe(
        self,
        text,
        *,
        metadata=None,
        is_final: bool = False,
        **_: object,
    ):
        del metadata, is_final
        self.main_tk_root.after(0, lambda: self._update_live_transcription(text))

    def create_image(self, width, height, color1, color2=None):
        # Logic moved from global
        image = Image.new('RGB', (width, height), color1)
        if color2:
            dc = ImageDraw.Draw(image)
            dc.rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=color2)
        return image

    # ------------------------------------------------------------------
    # Painel de auditoria de dependências
    # ------------------------------------------------------------------
    def present_dependency_audit(
        self,
        result: DependencyAuditResult | None,
        *,
        error_message: str | None = None,
    ) -> None:
        """Exibe um painel não modal com o resultado da auditoria de dependências."""

        self._dependency_audit_presented = True
        window_exists = bool(
            self._dependency_audit_window and self._dependency_audit_window.winfo_exists()
        )

        if not window_exists:
            audit_window = ctk.CTkToplevel(self.main_tk_root)
            audit_window.title("Dependency Audit")
            audit_window.geometry("720x540")
            audit_window.resizable(True, True)
            try:
                audit_window.iconbitmap("icon.ico")
            except Exception:
                logging.debug("Dependency audit window icon not applied.", exc_info=True)
            audit_window.protocol("WM_DELETE_WINDOW", self._close_dependency_audit_window)

            header = ctk.CTkFrame(audit_window)
            header.pack(fill="x", padx=16, pady=(16, 12))
            self._dependency_audit_summary_label = ctk.CTkLabel(
                header,
                text="",
                font=("Segoe UI", 16, "bold"),
                justify="left",
                anchor="w",
                wraplength=660,
            )
            self._dependency_audit_summary_label.pack(fill="x")
            self._dependency_audit_timestamp_label = ctk.CTkLabel(
                header,
                text="",
                font=("Segoe UI", 12),
                justify="left",
                anchor="w",
            )
            self._dependency_audit_timestamp_label.pack(fill="x", pady=(6, 0))

            self._dependency_audit_content = ctk.CTkScrollableFrame(
                audit_window,
                height=360,
            )
            self._dependency_audit_content.pack(
                fill="both",
                expand=True,
                padx=16,
                pady=(0, 12),
            )

            button_frame = ctk.CTkFrame(audit_window)
            button_frame.pack(fill="x", padx=16, pady=(0, 16))
            self._dependency_audit_copy_all_btn = ctk.CTkButton(
                button_frame,
                text="Copiar todos os comandos",
                command=self._copy_all_dependency_audit_commands,
            )
            self._dependency_audit_copy_all_btn.pack(side="left")
            docs_button = ctk.CTkButton(
                button_frame,
                text="Abrir documentação",
                command=self._open_dependency_audit_docs,
            )
            docs_button.pack(side="right")

            self._dependency_audit_window = audit_window
        else:
            audit_window = self._dependency_audit_window
            try:
                audit_window.lift()
            except Exception:
                logging.debug("Failed to raise dependency audit window.", exc_info=True)

        summary_label = self._dependency_audit_summary_label
        timestamp_label = self._dependency_audit_timestamp_label
        container = self._dependency_audit_content
        if not container:
            return

        summary_text = ""
        timestamp_text = ""
        if result is not None:
            summary_text = result.summary_message()
            try:
                localized = result.generated_at.astimezone()
            except Exception:
                localized = result.generated_at
            timestamp_text = f"Gerado em {localized.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        elif error_message:
            summary_text = error_message
            timestamp_text = "Resultado indisponível no momento."
        else:
            summary_text = "Dependency audit result unavailable."
            timestamp_text = "Resultado indisponível no momento."

        if summary_label is not None:
            summary_label.configure(text=summary_text)
        if timestamp_label is not None:
            timestamp_label.configure(text=timestamp_text)

        for child in container.winfo_children():
            child.destroy()

        self._dependency_audit_commands = []
        has_issues = False

        if result is None:
            info = error_message or "Dependency audit result unavailable."
            ctk.CTkLabel(
                container,
                text=info,
                justify="left",
                anchor="w",
                wraplength=640,
            ).pack(anchor="w", padx=12, pady=(0, 8))
        else:
            sections = [
                (
                    "Dependências ausentes",
                    result.missing,
                    "Instale os pacotes listados para alinhar o ambiente ao manifesto.",
                ),
                (
                    "Versões fora da especificação",
                    result.version_mismatches,
                    "Atualize as bibliotecas abaixo para satisfazer os intervalos declarados.",
                ),
                (
                    "Divergências de hash",
                    result.hash_mismatches,
                    "Reinstale os pacotes com hashes divergentes para garantir integridade.",
                ),
            ]

            for title, issues, guidance in sections:
                if issues:
                    has_issues = True
                self._render_dependency_issue_section(
                    container,
                    title,
                    issues,
                    guidance,
                )

        if self._dependency_audit_copy_all_btn is not None:
            state = "normal" if (self._dependency_audit_commands and has_issues) else "disabled"
            self._dependency_audit_copy_all_btn.configure(state=state)

    def _render_dependency_issue_section(
        self,
        parent,
        title: str,
        issues: Iterable[DependencyIssue],
        guidance: str,
    ) -> None:
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=8, pady=(0, 12))

        ctk.CTkLabel(
            section,
            text=title,
            font=("Segoe UI", 15, "bold"),
            anchor="w",
            justify="left",
        ).pack(fill="x", pady=(8, 4), padx=8)

        if guidance:
            ctk.CTkLabel(
                section,
                text=guidance,
                wraplength=640,
                justify="left",
                anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 8))

        issues = list(issues)
        if not issues:
            ctk.CTkLabel(
                section,
                text="Nenhum item identificado.",
                justify="left",
                anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 8))
            return

        for issue in issues:
            card = ctk.CTkFrame(section, corner_radius=8)
            card.pack(fill="x", padx=12, pady=(0, 8))

            metadata_lines = [
                f"Requisito: {issue.requirement_string}",
                f"Origem: {Path(issue.requirement_file).name}:{issue.line_number}",
            ]
            if issue.specifier:
                metadata_lines.append(f"Política declarada: {issue.specifier}")
            if issue.marker:
                metadata_lines.append(f"Marker: {issue.marker}")
            if issue.installed_version:
                metadata_lines.append(f"Versão instalada: {issue.installed_version}")
            elif issue.category == "missing":
                metadata_lines.append("Versão instalada: <não localizada>")
            if issue.hashes:
                metadata_lines.append(f"Hashes esperados: {', '.join(issue.hashes)}")

            details = issue.details or {}
            expected = details.get("expected") if isinstance(details, dict) else None
            detected = details.get("detected") if isinstance(details, dict) else None
            if expected:
                metadata_lines.append(f"Hashes declarados: {expected}")
            if detected:
                metadata_lines.append(f"Hashes detectados: {detected}")

            ctk.CTkLabel(
                card,
                text="\n".join(metadata_lines),
                justify="left",
                anchor="w",
                wraplength=620,
            ).pack(fill="x", padx=12, pady=(8, 4))

            suggestion_text = issue.suggestion
            ctk.CTkLabel(
                card,
                text=suggestion_text,
                justify="left",
                anchor="w",
                wraplength=620,
                text_color=("#A0A0A0", "#A0A0A0"),
            ).pack(fill="x", padx=12, pady=(0, 8))

            action_row = ctk.CTkFrame(card, fg_color="transparent")
            action_row.pack(fill="x", padx=12, pady=(0, 12))
            ctk.CTkButton(
                action_row,
                text="Copiar comando",
                command=lambda cmd=suggestion_text: self._copy_to_clipboard(cmd),
            ).pack(side="left")

            self._dependency_audit_commands.append(suggestion_text)

    def _copy_all_dependency_audit_commands(self) -> None:
        commands = [cmd for cmd in self._dependency_audit_commands if cmd]
        if not commands:
            return
        unique_commands = list(dict.fromkeys(commands))
        payload = "\n".join(unique_commands)
        self._copy_to_clipboard(payload)

    def _copy_to_clipboard(self, text: str) -> None:
        if not text:
            return
        try:
            self.main_tk_root.clipboard_clear()
            self.main_tk_root.clipboard_append(text)
        except Exception as exc:
            logging.error("Failed to copy text to clipboard: %s", exc, exc_info=True)
            return
        self.show_status_tooltip("Comando copiado para a área de transferência.")

    def _open_dependency_audit_docs(self) -> None:
        doc_path = self._dependency_audit_docs_path
        try:
            resolved = doc_path.resolve(strict=True)
        except FileNotFoundError:
            logging.warning("Dependency audit documentation not found at %s", doc_path)
            self.show_status_tooltip("Arquivo de documentação não encontrado.")
            return
        webbrowser.open(resolved.as_uri())

    def _close_dependency_audit_window(self) -> None:
        window = self._dependency_audit_window
        if not window:
            return
        try:
            window.destroy()
        except Exception:
            logging.debug("Failed to destroy dependency audit window.", exc_info=True)
        finally:
            self._dependency_audit_window = None
            self._dependency_audit_summary_label = None
            self._dependency_audit_timestamp_label = None
            self._dependency_audit_content = None
            self._dependency_audit_copy_all_btn = None
            self._dependency_audit_commands = []

    def _build_vad_section(self, parent, use_vad_var, vad_threshold_var, vad_silence_duration_var, vad_pre_speech_padding_ms_var, vad_post_speech_padding_ms_var):
        vad_frame = ctk.CTkFrame(parent)
        vad_frame.pack(fill="x", pady=5)
        ctk.CTkCheckBox(vad_frame, text="Ativar Detecção de Voz (VAD)", variable=use_vad_var).pack(side="left", padx=5)
        Tooltip(vad_frame, "Grava apenas quando houver voz.")

        vad_options_frame = ctk.CTkFrame(parent)
        vad_options_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(vad_options_frame, text="Limiar do VAD:").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_threshold_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Sensibilidade da detecção de voz.")

        ctk.CTkLabel(vad_options_frame, text="Duração de silêncio (s):").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_silence_duration_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Duração de silêncio para encerrar a fala.")


        ctk.CTkLabel(vad_options_frame, text="Padding pós-voz (ms):").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_post_speech_padding_ms_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Milissegundos preservados após o fim da fala.")

    def _build_asr_section(
        self,
        *,
        settings_win,
        asr_frame,
        transcription_frame,
        available_devices,
        asr_backend_var,
        asr_model_id_var,
        asr_compute_device_var,
        asr_ct2_compute_type_var,
        models_storage_dir_var,
        deps_install_dir_var,
        storage_root_dir_var,
        recordings_dir_var,
        asr_cache_dir_var,
        ui_elements,
    ):
        """Construct the ASR configuration widgets.

        Returns a dictionary with UI references needed by the caller.
        """

        advanced_state = {'visible': False}
        advanced_specs = []
        toggle_button_ref = {'widget': None}

        def _register_advanced(widget, **pack_kwargs):
            advanced_specs.append((widget, pack_kwargs))
            if advanced_state['visible']:
                widget.pack(**pack_kwargs)

        def _set_advanced_visibility(show: bool) -> None:
            advanced_state['visible'] = show
            for widget, pack_kwargs in advanced_specs:
                try:
                    widget.pack_forget()
                except Exception:
                    continue
                if show:
                    widget.pack(**pack_kwargs)
            button = toggle_button_ref['widget']
            if button is not None:
                button.configure(text='Ocultar avançado' if show else 'Mostrar avançado')

        def _toggle_advanced() -> None:
            _set_advanced_visibility(not advanced_state['visible'])

        advanced_toggle = ctk.CTkButton(
            asr_frame,
            text='Mostrar avançado',
            command=_toggle_advanced,
        )
        advanced_toggle.pack(fill='x', pady=(0, 5))
        toggle_button_ref['widget'] = advanced_toggle

        previous_models_dir = {"value": models_storage_dir_var.get() if models_storage_dir_var else ""}
        previous_deps_dir = {"value": deps_install_dir_var.get() if deps_install_dir_var else ""}

        def _validate_directory(label: str, raw_value: str, *, show_dialog: bool = True) -> Path | None:
            sanitized = (raw_value or "").strip()
            if not sanitized:
                return None
            ok, message, resolved = self.config_manager.validate_directory_candidate(
                sanitized,
                label=label,
            )
            if show_dialog and message:
                dialog = messagebox.showinfo if ok else messagebox.showerror
                if settings_win is not None:
                    dialog(label, message, parent=settings_win)
                else:
                    dialog(label, message)
            if ok:
                logging.info(message)
                return resolved
            logging.warning(message)
            return None

        def _on_deps_focus_out() -> None:
            value = deps_install_dir_var.get().strip() if deps_install_dir_var else ""
            if not value:
                return
            resolved = _validate_directory("Dependencies", value, show_dialog=False)
            if resolved is None:
                deps_install_dir_var.set(previous_deps_dir.get("value", ""))
            else:
                deps_install_dir_var.set(str(resolved))

        def _browse_deps_dir() -> None:
            initial = deps_install_dir_var.get() if deps_install_dir_var else ""
            try:
                initial_dir = Path(initial).expanduser()
            except Exception:
                initial_dir = Path.home()
            selected = filedialog.askdirectory(initialdir=str(initial_dir))
            if selected:
                resolved = _validate_directory("Dependencies", selected)
                if resolved is not None:
                    deps_install_dir_var.set(str(resolved))
                else:
                    deps_install_dir_var.set(previous_deps_dir.get("value", ""))

        def _check_deps_dir() -> None:
            value = deps_install_dir_var.get().strip() if deps_install_dir_var else ""
            if not value:
                messagebox.showwarning(
                    "Dependencies",
                    "Selecione um diretório de dependências antes de validar.",
                    parent=settings_win,
                )
                return
            ok, message, _ = self.config_manager.validate_directory_candidate(
                value,
                label="Dependencies",
            )
            dialog = messagebox.showinfo if ok else messagebox.showerror
            if ok:
                logging.info(message)
            else:
                logging.warning(message)
            dialog("Dependencies", message, parent=settings_win)

        def _migrate_deps_dir() -> None:
            source = previous_deps_dir.get("value") or ""
            destination = deps_install_dir_var.get().strip() if deps_install_dir_var else ""
            if not source or not destination:
                messagebox.showwarning(
                    "Dependencies",
                    "Defina diretórios de origem e destino antes de migrar.",
                    parent=settings_win,
                )
                return
            if source == destination:
                messagebox.showinfo(
                    "Dependencies",
                    "Os diretórios de origem e destino são idênticos.",
                    parent=settings_win,
                )
                return
            confirm = messagebox.askyesno(
                "Dependencies",
                (
                    "Migrar os artefatos existentes de\n"
                    f"{source}\npara\n{destination}?"
                ),
                parent=settings_win,
            )
            if not confirm:
                return
            success = self.config_manager.migrate_directory(
                source,
                destination,
                label="Dependencies",
            )
            if success:
                previous_deps_dir["value"] = destination
                self.config_manager.apply_environment_overrides()
                messagebox.showinfo(
                    "Dependencies",
                    "Migração concluída com sucesso.",
                    parent=settings_win,
                )
            else:
                messagebox.showerror(
                    "Dependencies",
                    "A migração falhou. Verifique os logs para mais detalhes.",
                    parent=settings_win,
                )

        def _update_cache_dir_for_new_base(new_base: str) -> None:
            old_base = previous_models_dir.get("value") or ""
            previous_models_dir["value"] = new_base or ""
            if not new_base:
                return
            try:
                new_base_path = Path(new_base).expanduser()
            except Exception:
                return

            try:
                cache_current = Path(asr_cache_dir_var.get()).expanduser()
            except Exception:
                cache_current = None

            try:
                old_base_path = Path(old_base).expanduser() if old_base else None
            except Exception:
                old_base_path = None

            if cache_current is None or old_base_path is None:
                return

            try:
                relative = cache_current.relative_to(old_base_path)
            except ValueError:
                return

            asr_cache_dir_var.set(str(new_base_path / relative))

        def _browse_models_dir() -> None:
            initial = models_storage_dir_var.get() if models_storage_dir_var else ""
            try:
                initial_dir = Path(initial).expanduser()
            except Exception:
                initial_dir = Path.home()
            selected = filedialog.askdirectory(initialdir=str(initial_dir))
            if selected:
                models_storage_dir_var.set(selected)
                _update_cache_dir_for_new_base(selected)

        def _synchronize_cache_dir() -> None:
            base = models_storage_dir_var.get()
            if not base:
                return
            try:
                candidate = Path(base).expanduser() / "asr"
            except Exception:
                return
            asr_cache_dir_var.set(str(candidate))

        models_dir_frame = ctk.CTkFrame(asr_frame)
        models_dir_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(models_dir_frame, text="Diretório de Modelos:").pack(side="left", padx=(5, 10))
        models_dir_entry = ctk.CTkEntry(models_dir_frame, textvariable=models_storage_dir_var, width=240)
        models_dir_entry.pack(side="left", padx=5)
        Tooltip(models_dir_entry, "Diretório raiz usado para armazenar todos os modelos e artefatos pesados.")
        models_dir_entry.bind("<FocusOut>", lambda *_: _update_cache_dir_for_new_base(models_storage_dir_var.get()))

        browse_button = ctk.CTkButton(models_dir_frame, text="Selecionar...", command=_browse_models_dir)
        browse_button.pack(side="left", padx=5)
        Tooltip(browse_button, "Escolha o diretório raiz onde os modelos serão armazenados.")

        sync_button = ctk.CTkButton(models_dir_frame, text="Sincronizar Cache ASR", command=_synchronize_cache_dir)
        sync_button.pack(side="left", padx=5)
        Tooltip(sync_button, "Atualiza o diretório de cache ASR para ficar dentro do diretório de modelos.")

        deps_dir_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(deps_dir_frame, fill="x", pady=5)
        ctk.CTkLabel(deps_dir_frame, text="Diretório de Dependências:").pack(side="left", padx=(5, 10))
        deps_dir_entry = ctk.CTkEntry(deps_dir_frame, textvariable=deps_install_dir_var, width=240)
        deps_dir_entry.pack(side="left", padx=5)
        Tooltip(
            deps_dir_entry,
            "Local onde caches do Hugging Face e dependências auxiliares serão mantidos.",
        )
        deps_dir_entry.bind("<FocusOut>", lambda *_: _on_deps_focus_out())

        deps_browse_button = ctk.CTkButton(
            deps_dir_frame,
            text="Selecionar...",
            command=_browse_deps_dir,
        )
        deps_browse_button.pack(side="left", padx=5)
        Tooltip(deps_browse_button, "Escolha o diretório para armazenar dependências compartilhadas.")

        deps_validate_button = ctk.CTkButton(
            deps_dir_frame,
            text="Validar espaço",
            command=_check_deps_dir,
        )
        deps_validate_button.pack(side="left", padx=5)
        Tooltip(deps_validate_button, "Verifica permissões e espaço disponível no diretório selecionado.")

        deps_migrate_button = ctk.CTkButton(
            deps_dir_frame,
            text="Migrar ativos",
            command=_migrate_deps_dir,
        )
        deps_migrate_button.pack(side="left", padx=5)
        Tooltip(deps_migrate_button, "Move dependências existentes para o novo diretório.")

        asr_backend_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(asr_backend_frame, fill="x", pady=5)
        ctk.CTkLabel(asr_backend_frame, text="Backend ASR:").pack(side="left", padx=(5, 0))
        ctk.CTkButton(
            asr_backend_frame,
            text="?",
            width=20,
            command=lambda: messagebox.showinfo(
                "ASR Backend",
                "Seleciona o mecanismo de inferência utilizado na transcrição.",
            ),
        ).pack(side="left", padx=(0, 10))

        def _on_backend_change(choice: str) -> None:
            asr_backend_var.set(choice)
            _update_install_button_state()
            _update_model_info(asr_model_id_var.get())

        asr_backend_menu = ctk.CTkOptionMenu(
            asr_backend_frame,
            variable=asr_backend_var,
            values=["ctranslate2"],
            command=_on_backend_change,
        )
        asr_backend_menu.pack(side="left", padx=5)
        Tooltip(
            asr_backend_menu,
            "CTranslate2 runtime used for all curated models.",
        )

        quant_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(quant_frame, fill="x", pady=5)
        ctk.CTkLabel(quant_frame, text="Quantization:").pack(side="left", padx=(5, 0))
        ctk.CTkButton(
            quant_frame,
            text="?",
            width=20,
            command=lambda: messagebox.showinfo(
                "Quantization",
                "Reduces model precision for faster inference (float16/int8).",
            ),
        ).pack(side="left", padx=(0, 10))

        model_manager = self.model_manager
        runtime_catalog = self.config_manager.get_runtime_model_catalog() or []
        if not runtime_catalog:
            fallback_catalog = model_manager.list_catalog()
            runtime_catalog = []
            for entry in fallback_catalog:
                baseline = dict(entry)
                baseline.setdefault("hardware_status", "ok")
                baseline.setdefault("hardware_messages", [])
                baseline.setdefault("hardware_warnings", [])
                baseline.setdefault("hardware_blockers", [])
                runtime_catalog.append(baseline)
        runtime_by_id = {entry["id"]: entry for entry in runtime_catalog}
        ui_elements["catalog"] = runtime_catalog
        self._set_settings_meta("catalog", runtime_catalog)

        try:
            installed_models = model_manager.list_installed(asr_cache_dir_var.get())
        except OSError:
            messagebox.showerror(
                "Configuration",
                "Unable to access the model cache directory. Please review the path in Settings.",
            )
            installed_models = []
        except Exception as exc:  # pragma: no cover - defensive path
            logging.warning("Failed to list installed models: %s", exc)
            installed_models = []
        installed_by_id = {item.get("id"): item for item in installed_models if item.get("id")}

        id_to_display = {
            entry["id"]: entry.get("display_name", entry["id"])
            for entry in runtime_catalog
        }
        display_to_id = {display: model_id for model_id, display in id_to_display.items()}
        ui_elements["id_to_display"] = id_to_display
        ui_elements["display_to_id"] = display_to_id
        self._set_settings_meta("id_to_display", id_to_display)
        self._set_settings_meta("display_to_id", display_to_id)

        asr_model_display_var = ctk.StringVar(
            value=id_to_display.get(asr_model_id_var.get(), asr_model_id_var.get())
        )
        ui_elements["asr_model_display_var"] = asr_model_display_var
        self._set_settings_var("asr_model_display_var", asr_model_display_var)
        self._set_settings_var("asr_model_menu", None)

        recommendation_info = self.config_manager.get_runtime_recommendation() or {}
        recommended_model_id = recommendation_info.get("id")

        def _format_size(value: Any) -> str:
            try:
                amount = float(int(value))
            except (TypeError, ValueError):
                return "?"
            if amount <= 0:
                return "?"
            units = ["B", "KB", "MB", "GB", "TB"]
            for unit in units:
                if amount < 1024 or unit == units[-1]:
                    return f"{amount:.1f} {unit}"
                amount /= 1024
            return f"{amount:.1f} PB"

        def _format_duration(seconds: Any) -> str:
            try:
                total_seconds = float(seconds)
            except (TypeError, ValueError):
                return ""
            if total_seconds <= 0:
                return ""
            minutes = int(total_seconds // 60)
            secs = int(round(total_seconds % 60))
            if secs == 60:
                minutes += 1
                secs = 0
            if minutes:
                return f"~{minutes} min {secs:02d} s"
            return f"~{secs} s"

        def _compose_summary(entry: dict[str, Any]) -> str:
            parts: list[str] = []
            download_text = _format_size(entry.get("estimated_download_bytes"))
            duration_text = _format_duration(entry.get("estimated_download_seconds"))
            bandwidth = entry.get("estimated_download_reference_mbps")
            if download_text != "?":
                if duration_text:
                    if isinstance(bandwidth, (int, float)) and bandwidth:
                        parts.append(
                            f"Download {download_text} ({duration_text} @ {float(bandwidth):.0f} Mbps)"
                        )
                    else:
                        parts.append(f"Download {download_text} ({duration_text})")
                else:
                    parts.append(f"Download {download_text}")
            disk_text = _format_size(entry.get("estimated_disk_bytes"))
            if disk_text != "?":
                parts.append(f"Disco ~{disk_text}")
            cpu_rtf = entry.get("estimated_cpu_rtf")
            if isinstance(cpu_rtf, (int, float)) and cpu_rtf > 0:
                parts.append(f"CPU ≈ {cpu_rtf:.1f}× tempo real")
            gpu_rtf = entry.get("estimated_gpu_rtf")
            if isinstance(gpu_rtf, (int, float)) and gpu_rtf > 0:
                parts.append(f"GPU ≈ {gpu_rtf:.2f}× tempo real")
            backend_label = entry.get("backend")
            if backend_label:
                parts.append(f"Backend: {backend_label}")
            if entry.get("requires_gpu"):
                parts.append("Requer GPU")
            elif entry.get("preferred_device") == "gpu":
                parts.append("Melhor em GPU")
            parts.append("Instalado" if entry["id"] in installed_by_id else "Não instalado")
            summary = " • ".join(parts) if parts else ""
            description = entry.get("description")
            if description:
                summary = f"{summary}\n{description}" if summary else description
            return summary or " "

        def _hardware_status_text(entry: dict[str, Any]) -> tuple[str, str]:
            status = entry.get("hardware_status") or "ok"
            messages = entry.get("hardware_messages") or []
            if status == "blocked":
                text = "Incompatível: " + ("; ".join(messages) if messages else "requisitos não atendidos.")
                return text, "#d61f1f"
            if status == "warn":
                text = "Aviso: " + ("; ".join(messages) if messages else "desempenho pode ser limitado.")
                return text, "#d7a500"
            text = "Compatível com seu hardware atual."
            if messages:
                text += " " + " ".join(messages)
            return text, "#2a7f39"

        def _handle_model_choice(model_id: str) -> None:
            display_value = id_to_display.get(model_id, model_id)
            try:
                asr_model_display_var.set(display_value)
            except Exception:
                pass
            self._on_model_change(display_value)
            self._update_install_button_state()

        recommended_entries = [
            entry for entry in runtime_catalog if entry.get("ui_group") == "recommended"
        ]
        advanced_entries = [
            entry for entry in runtime_catalog if entry.get("ui_group") != "recommended"
        ]
        if not recommended_entries:
            recommended_entries = runtime_catalog
            advanced_entries = []

        recommended_section = ctk.CTkFrame(asr_frame, fg_color="transparent")
        recommended_section.pack(fill="x", pady=5)
        ctk.CTkLabel(
            recommended_section,
            text="Recomendados",
            font=ctk.CTkFont(weight="bold"),
        ).pack(anchor="w", padx=5, pady=(0, 4))

        def _create_model_option(entry: dict[str, Any], parent) -> None:
            container = ctk.CTkFrame(parent, fg_color="transparent")
            container.pack(fill="x", pady=2)
            header = ctk.CTkFrame(container, fg_color="transparent")
            header.pack(fill="x")
            radio = ctk.CTkRadioButton(
                header,
                text=id_to_display.get(entry["id"], entry["id"]),
                variable=asr_model_id_var,
                value=entry["id"],
                command=lambda eid=entry["id"]: _handle_model_choice(eid),
            )
            radio.pack(side="left", padx=(5, 0))
            if entry["id"] == recommended_model_id and entry.get("hardware_status") != "blocked":
                badge = ctk.CTkLabel(
                    header,
                    text="Sugerido",
                    text_color="#1f6aa5",
                )
                badge.pack(side="left", padx=(8, 0))
            summary_label = ctk.CTkLabel(
                container,
                text=_compose_summary(entry),
                justify="left",
                wraplength=520,
            )
            summary_label.pack(fill="x", padx=32, pady=(0, 2))
            status_text, status_color = _hardware_status_text(entry)
            status_label = ctk.CTkLabel(
                container,
                text=status_text,
                text_color=status_color,
                justify="left",
                wraplength=520,
            )
            status_label.pack(fill="x", padx=32, pady=(0, 4))

        for entry in recommended_entries:
            _create_model_option(entry, recommended_section)

        if advanced_entries:
            advanced_section = ctk.CTkFrame(asr_frame, fg_color="transparent")
            _register_advanced(advanced_section, fill="x", pady=5)
            ctk.CTkLabel(
                advanced_section,
                text="Avançados",
                font=ctk.CTkFont(weight="bold"),
            ).pack(anchor="w", padx=5, pady=(0, 4))
            for entry in advanced_entries:
                _create_model_option(entry, advanced_section)

        model_size_label = ctk.CTkLabel(
            asr_frame, text="Download: calculating... | Installed: -"
        )
        model_size_label.pack(fill="x", padx=5, pady=(2, 5))
        ui_elements["model_size_label"] = model_size_label

        def _derive_backend_from_model(model_id: str) -> str | None:
            entry = runtime_by_id.get(model_id)
            if entry is None:
                entry = installed_by_id.get(model_id)
            backend = _backend_display_value_global(entry.get("backend") if entry else None)
            return backend or None

        def _update_model_info(model_id: str) -> None:
            ui_elements["model_size_label"].configure(text="Download: calculating... | Installed: -")
            try:
                d_bytes, d_files = model_manager.get_model_download_size(model_id)
                d_mb = d_bytes / (1024 * 1024)
                download_text = f"{d_mb:.1f} MB ({d_files} files)"
            except Exception:
                download_text = "?"

            try:
                refreshed_installed = model_manager.list_installed(asr_cache_dir_var.get())
            except OSError:
                messagebox.showerror(
                    "Configuration",
                    "Unable to access the model cache directory. Please review the path in Settings.",
                )
                refreshed_installed = []
            entry = next((m for m in refreshed_installed if m.get("id") == model_id), None)
            if entry:
                i_bytes, i_files = model_manager.get_installed_size(entry.get("path"))
                i_mb = i_bytes / (1024 * 1024)
                installed_text = f"{i_mb:.1f} MB ({i_files} files)"
            else:
                installed_text = "-"

            ui_elements["model_size_label"].configure(
                text=f"Download: {download_text} | Installed: {installed_text}"
            )

        def _update_install_button_state() -> None:
            recommended_backend = _derive_backend_from_model(asr_model_id_var.get())
            install_button.configure(state="normal" if recommended_backend else "disabled")
            effective_backend = _backend_display_value_global(asr_backend_var.get()) or recommended_backend
            ui_elements["quant_menu"].configure(
                state="normal" if effective_backend == "ctranslate2" else "disabled"
            )

        _update_model_info(asr_model_id_var.get())

        def _reset_asr() -> None:
            default_model_id = DEFAULT_CONFIG["asr_model_id"]
            display_value = id_to_display.get(default_model_id, default_model_id)
            asr_model_id_var.set(default_model_id)
            asr_model_display_var.set(display_value)
            asr_backend_var.set(DEFAULT_CONFIG["asr_backend"])
            asr_backend_menu.set(DEFAULT_CONFIG["asr_backend"])
            asr_ct2_compute_type_var.set(DEFAULT_CONFIG["asr_ct2_compute_type"])
            asr_ct2_menu.set(DEFAULT_CONFIG["asr_ct2_compute_type"])
            asr_cache_dir_var.set(DEFAULT_CONFIG["asr_cache_dir"])
            storage_root_dir_var.set(DEFAULT_CONFIG[STORAGE_ROOT_DIR_CONFIG_KEY])
            recordings_dir_var.set(DEFAULT_CONFIG[RECORDINGS_DIR_CONFIG_KEY])
            _handle_model_choice(default_model_id)
            self.config_manager.set_asr_ct2_compute_type(DEFAULT_CONFIG["asr_ct2_compute_type"])
            self.config_manager.set_asr_cache_dir(DEFAULT_CONFIG["asr_cache_dir"])
            self.config_manager.set_storage_root_dir(DEFAULT_CONFIG[STORAGE_ROOT_DIR_CONFIG_KEY])
            self.config_manager.set_recordings_dir(DEFAULT_CONFIG[RECORDINGS_DIR_CONFIG_KEY])
            self.config_manager.set_python_packages_dir(DEFAULT_CONFIG[PYTHON_PACKAGES_DIR_CONFIG_KEY])
            self.config_manager.set_vad_models_dir(DEFAULT_CONFIG[VAD_MODELS_DIR_CONFIG_KEY])
            self.config_manager.set_hf_cache_dir(DEFAULT_CONFIG[HF_CACHE_DIR_CONFIG_KEY])
            self.config_manager.save_config()

        reset_asr_button = ctk.CTkButton(
            asr_frame, text="Reset ASR", command=_reset_asr
        )
        reset_asr_button.pack(fill="x", padx=5, pady=(0, 5))
        Tooltip(reset_asr_button, "Restore default ASR settings.")

        asr_device_frame = ctk.CTkFrame(asr_frame)
        asr_device_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(asr_device_frame, text="ASR Compute Device:").pack(side="left", padx=(5, 10))
        asr_device_menu = ctk.CTkOptionMenu(asr_device_frame, variable=asr_compute_device_var, values=available_devices)
        asr_device_menu.pack(side="left", padx=5)
        Tooltip(asr_device_menu, "Select compute device for ASR model.")

        asr_ct2_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(asr_ct2_frame, fill="x", pady=5)
        ctk.CTkLabel(asr_ct2_frame, text="CT2 Compute Type:").pack(side="left", padx=(5, 0))
        ctk.CTkButton(
            asr_ct2_frame,
            text="?",
            width=20,
            command=lambda: messagebox.showinfo(
                "CT2 Compute Type",
                "Numeric precision mode for the CTranslate2 backend.",
            ),
        ).pack(side="left", padx=(0, 10))
        asr_ct2_menu = ctk.CTkOptionMenu(
            asr_ct2_frame,
            variable=asr_ct2_compute_type_var,
            values=["default", "float16", "float32", "int8", "int8_float16", "int8_float32"],
        )
        asr_ct2_menu.pack(side="left", padx=5)
        Tooltip(asr_ct2_menu, "Compute type for CTranslate2 backend.")
        ui_elements["quant_menu"] = asr_ct2_menu

        asr_cache_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(asr_cache_frame, fill="x", pady=5)
        ctk.CTkLabel(
            asr_cache_frame,
            text="ASR Cache Directory:",
            width=200,
        ).pack(side="left", padx=(5, 10))
        asr_cache_entry = ctk.CTkEntry(asr_cache_frame, textvariable=asr_cache_dir_var, width=240)
        asr_cache_entry.pack(side="left", padx=5)
        Tooltip(asr_cache_entry, "Directory used to cache ASR models.")

        def _choose_asr_cache_dir():
            initial_dir = asr_cache_dir_var.get() if asr_cache_dir_var else ""
            directory = filedialog.askdirectory(
                title="Select ASR cache directory",
                initialdir=initial_dir or None,
            )
            if directory:
                asr_cache_dir_var.set(directory)

        browse_cache_button = ctk.CTkButton(
            asr_cache_frame,
            text="Browse...",
            width=90,
            command=_choose_asr_cache_dir,
        )
        browse_cache_button.pack(side="left", padx=5)
        Tooltip(browse_cache_button, "Open a folder chooser for the ASR cache directory.")

        def _install_model():
            cache_dir = asr_cache_dir_var.get()
            try:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Invalid Path", f"ASR cache directory is invalid:\n{e}")
                return

            model_id = asr_model_id_var.get()
            backend = _derive_backend_from_model(model_id)
            if backend is None:
                messagebox.showerror(
                    "Model", "Unable to determine backend for selected model.",
                )
                return

            try:
                size_bytes, file_count = model_manager.get_model_download_size(model_id)
                size_gb = size_bytes / (1024 ** 3)
                detail = f"approximately {size_gb:.2f} GB ({file_count} files)"
            except Exception:
                detail = "an unspecified size"

            if not messagebox.askyesno(
                "Model Download",
                f"Model '{model_id}' will download {detail}.\nContinue?",
            ):
                return

            try:
                download_result = self.core_instance_ref.download_model_with_ui(
                    model_id=model_id,
                    backend=backend,
                    cache_dir=cache_dir,
                    quant=asr_ct2_compute_type_var.get() if backend == "ctranslate2" else None,
                    reason="settings",
                )
            except Exception as e:  # pragma: no cover - defensive guard
                messagebox.showerror("Model", f"Download failed: {e}")
                return

            status = download_result.get("status")
            if status == "success":
                _update_model_info(model_id)
                messagebox.showinfo("Model", "Download completed.")
            elif status == "cancelled":
                messagebox.showinfo("Model", "Download canceled.")
            elif status == "error":
                message = download_result.get("error") or "Unknown error"
                messagebox.showerror("Model", f"Download failed: {message}")

        def _reload_model():
            handler = getattr(self.core_instance_ref, "transcription_handler", None)
            if handler:
                handler.reload_asr()

        install_button = ctk.CTkButton(
            asr_frame, text="Install/Update", command=_install_model
        )
        install_button.pack(pady=5)
        for idx, (widget, pack_kwargs) in enumerate(list(advanced_specs)):
            if "before" not in pack_kwargs:
                updated_kwargs = dict(pack_kwargs)
                updated_kwargs["before"] = install_button
                advanced_specs[idx] = (widget, updated_kwargs)
        reload_button = ctk.CTkButton(
            asr_frame, text="Reload Model", command=_reload_model
        )
        reload_button.pack(pady=5)

        asr_model_menu = None
        _update_model_info(asr_model_id_var.get())
        _update_install_button_state()
        _on_backend_change(asr_backend_var.get())
        should_show_advanced = any(
            [
                asr_backend_var.get() not in ("auto", ""),
                asr_ct2_compute_type_var.get() not in ("auto", "float16"),
            ]
        )
        _set_advanced_visibility(should_show_advanced)

        return {
            "advanced_toggle": advanced_toggle,
            "set_advanced_visibility": _set_advanced_visibility,
            "asr_backend_menu": asr_backend_menu,
            "asr_ct2_menu": asr_ct2_menu,
            "asr_model_menu": asr_model_menu,
            "asr_model_display_var": asr_model_display_var,
            "id_to_display": id_to_display,
            "on_backend_change": _on_backend_change,
            "update_model_info": _update_model_info,
            "update_install_button_state": _update_install_button_state,
        }

    def apply_settings_payload(self, payload: dict) -> None:
        """Apply settings in headless mode or via unit tests."""

        if not isinstance(payload, dict):  # Salvaguarda para chamadas incorretas
            raise TypeError("payload must be a dict of keyword arguments")
        if not self.core_instance_ref:
            raise RuntimeError("core_instance_ref is not configured")

        logging.info("UIManager: forwarding settings payload to AppCore (%d keys).", len(payload))
        self.core_instance_ref.apply_settings_from_external(**payload)

    # ------------------------------------------------------------------
    # Normalização de notificações de estado
    # ------------------------------------------------------------------
    def _coerce_state_notification(self, payload: Any) -> Any:
        """Tenta identificar ``StateNotification`` sem importar ``core``."""

        if payload is None:
            return None

        try:
            has_dataclass_fields = bool(getattr(payload, "__dataclass_fields__", None))
        except Exception:
            has_dataclass_fields = False

        required_attrs = ("state", "event")
        if has_dataclass_fields and all(hasattr(payload, attr) for attr in required_attrs):
            return payload

        try:
            cls = payload.__class__
            if cls.__name__ == "StateNotification" and all(hasattr(payload, attr) for attr in required_attrs):
                return payload
        except Exception:
            return None

        return None

    def _normalize_state_payload(self, payload: Any) -> tuple[Any, Any, dict[str, Any] | None]:
        """Extrai ``state``, payloads de aviso e metadados contextuais."""

        notification = self._coerce_state_notification(payload)
        if notification is not None:
            context = {
                "notification": notification,
                "state": getattr(notification, "state", None),
                "event": getattr(notification, "event", None),
                "details": getattr(notification, "details", None),
                "source": getattr(notification, "source", None),
                "previous_state": getattr(notification, "previous_state", None),
                "operation_id": getattr(notification, "operation_id", None),
            }
            return context["state"], None, context

        if isinstance(payload, dict):
            return payload.get("state"), payload.get("warning"), None

        return payload, None, None

    def _build_state_context_suffix(self, state: str, context: dict[str, Any] | None) -> str:
        """Gera sufixo textual com detalhes do estado para tooltips."""

        if not context:
            return ""

        parts: list[str] = []
        details = context.get("details")
        if details:
            parts.append(str(details))

        source = context.get("source")
        if source:
            parts.append(f"source={source}")

        event_obj = context.get("event")
        event_name = None
        if event_obj is not None:
            event_name = getattr(event_obj, "name", None)
            if not event_name:
                event_name = str(event_obj)
        if event_name:
            parts.append(f"event={event_name}")

        op_id = context.get("operation_id")
        if op_id and not parts:
            parts.append(f"op={op_id}")

        return f" — {' | '.join(parts)}" if parts else ""

    def _clamp_tray_tooltip(self, text: str) -> str:
        """Assegura que o texto da tooltip caiba no limite imposto pelo sistema."""

        max_len = 128
        if len(text) > max_len:
            truncated = text[: max_len - 1].rstrip()
            text = f"{truncated}…"
            logging.debug("Tray tooltip truncated to %d characters: %s", max_len, text)
        return text

    def _recording_tooltip_updater(self):
        """Atualiza a tooltip com a duração da gravação a cada segundo."""
        while not self.stop_recording_timer_event.is_set():
            start_time = getattr(self.core_instance_ref.audio_handler, "start_time", None)
            if start_time is None:
                break
            elapsed = time.time() - start_time
            tooltip = f"Whisper Recorder (RECORDING - {self._format_elapsed(elapsed)})"
            suffix = getattr(self, "_state_context_suffix", "")
            if suffix:
                tooltip = f"{tooltip}{suffix}"
            self.tray_icon.title = self._clamp_tray_tooltip(tooltip)
            time.sleep(1)

    def _format_elapsed(self, seconds: float) -> str:
        """Formata segundos em MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _transcribing_tooltip_updater(self):
        """Atualiza a tooltip com a duração da transcrição a cada segundo.

        Estende a tooltip com informações técnicas (device/dtype/attn/chunk/batch)
        quando disponíveis, atualizando a cada tick.
        """
        start_ts = time.time()
        while not self.stop_transcribing_timer_event.is_set():
            try:
                tech = ""
                th = getattr(self.core_instance_ref, "transcription_handler", None)
                if th is not None:
                    device_in_use = getattr(th, "device_in_use", None)
                    if device_in_use:
                        device = str(device_in_use)
                    else:
                        gpu_index = getattr(th, "gpu_index", -1)
                        device = f"cuda:{gpu_index}" if isinstance(gpu_index, int) and gpu_index >= 0 else "cpu"
                    compute_type = (
                        getattr(th, "compute_type_in_use", None)
                        or getattr(th, "asr_ct2_compute_type", None)
                        or "default"
                    )
                    try:
                        import importlib.util as _spec_util
                        attn_impl = "FA2" if _spec_util.find_spec("flash_attn") is not None else "SDPA"
                    except Exception:
                        attn_impl = "SDPA"
                    chunk = getattr(th, "chunk_length_sec", None)
                    chunk_display = chunk if chunk not in (None, "") else "auto"
                    # Se disponível no handler, podemos expor last_dynamic_batch_size; fallback em None
                    bs = getattr(th, "last_dynamic_batch_size", None) if hasattr(th, "last_dynamic_batch_size") else None
                    if bs in (None, ""):
                        bs = getattr(th, "batch_size", None) if hasattr(th, "batch_size") else None
                    bs_display = bs if bs not in (None, "") else "auto"
                    tech = (
                        f" [{device} ct2={compute_type} | {attn_impl} | chunk={chunk_display}s | batch={bs_display}]"
                    )
                elapsed = time.time() - start_ts
                tooltip = f"Whisper Recorder (TRANSCRIBING - {self._format_elapsed(elapsed)}){tech}"
                suffix = getattr(self, "_state_context_suffix", "")
                if suffix:
                    tooltip = f"{tooltip}{suffix}"
                if self.tray_icon:
                    self.tray_icon.title = self._clamp_tray_tooltip(tooltip)
            except Exception:
                # Em caso de falha, mantém somente o tempo
                elapsed = time.time() - start_ts
                if self.tray_icon:
                    tooltip = f"Whisper Recorder (TRANSCRIBING - {self._format_elapsed(elapsed)})"
                    suffix = getattr(self, "_state_context_suffix", "")
                    if suffix:
                        tooltip = f"{tooltip}{suffix}"
                    self.tray_icon.title = self._clamp_tray_tooltip(tooltip)
            time.sleep(1)

    def update_tray_icon(self, state):
        # Logic moved from global, ajustado para lidar com payloads estruturados
        normalized_state, warning_payload, context = self._normalize_state_payload(state)

        resolved_state = normalized_state if normalized_state is not None else "IDLE"
        state_str = str(resolved_state)

        event_obj = context.get("event") if context else None
        if event_obj == sm.StateEvent.MODEL_DOWNLOAD_PROGRESS:
            self._handle_download_progress(context)
        if context:
            self._last_state_notification = context.get("notification")
            self._last_operation_id = context.get("operation_id")
            self._state_context_suffix = self._build_state_context_suffix(state_str, context)
        else:
            self._last_state_notification = None
            self._last_operation_id = None
            self._state_context_suffix = ""

        suffix = self._state_context_suffix

        def _apply_suffix(text: str) -> str:
            return f"{text}{suffix}" if suffix else text

        if self.tray_icon:
            try:
                icon_image = Image.open("icon.png")
            except FileNotFoundError:
                logging.warning("icon.png not found, using fallback image for update.")
                color1, color2 = self.ICON_COLORS.get(state_str, self.DEFAULT_ICON_COLOR)
                icon_image = self.create_image(64, 64, color1, color2)
            self.tray_icon.icon = icon_image
            tooltip = f"Whisper Recorder ({state_str})"

            # Controle de threads de tooltip por estado
            if state_str == "RECORDING":
                # Parar contador de TRANSCRIBING se estiver ativo
                if self.transcribing_timer_thread and self.transcribing_timer_thread.is_alive():
                    self.stop_transcribing_timer_event.set()
                    self.transcribing_timer_thread.join(timeout=1)

                if not self.recording_timer_thread or not self.recording_timer_thread.is_alive():
                    self.stop_recording_timer_event.clear()
                    self.recording_timer_thread = threading.Thread(
                        target=self._recording_tooltip_updater,
                        daemon=True,
                        name="RecordingTooltipThread",
                    )
                    self.recording_timer_thread.start()
                start_time = getattr(self.core_instance_ref.audio_handler, "start_time", None)
                if start_time is not None:
                    elapsed = time.time() - start_time
                    tooltip = f"Whisper Recorder (RECORDING - {self._format_elapsed(elapsed)})"

            elif state_str == "TRANSCRIBING":
                # Parar contador de RECORDING se estiver ativo
                if self.recording_timer_thread and self.recording_timer_thread.is_alive():
                    self.stop_recording_timer_event.set()
                    self.recording_timer_thread.join(timeout=1)

                # Iniciar contador de TRANSCRIBING
                if not self.transcribing_timer_thread or not self.transcribing_timer_thread.is_alive():
                    self.stop_transcribing_timer_event.clear()
                    # Atualiza tooltip imediatamente ao entrar em TRANSCRIBING
                    # Inclui dados técnicos quando disponíveis
                    try:
                        # Esses atributos são expostos via core_instance_ref.transcription_handler
                        th = getattr(self.core_instance_ref, "transcription_handler", None)
                        if th is not None:
                            device_in_use = getattr(th, "device_in_use", None)
                            if device_in_use:
                                device = str(device_in_use)
                            else:
                                gpu_index = getattr(th, "gpu_index", -1)
                                device = (
                                    f"cuda:{gpu_index}"
                                    if isinstance(gpu_index, int) and gpu_index >= 0
                                    else "cpu"
                                )
                            # Determinar attn_impl conforme detecção feita no handler
                            try:
                                import importlib.util as _spec_util
                                attn_impl = "FA2" if _spec_util.find_spec("flash_attn") is not None else "SDPA"
                            except Exception:
                                attn_impl = "SDPA"
                            compute_type = (
                                getattr(th, "compute_type_in_use", None)
                                or getattr(th, "asr_ct2_compute_type", None)
                                or "default"
                            )
                            chunk = getattr(th, "chunk_length_sec", None)
                            chunk_display = chunk if chunk not in (None, "") else "auto"
                            bs = getattr(th, "last_dynamic_batch_size", None)
                            if bs in (None, ""):
                                bs = getattr(th, "batch_size", None) if hasattr(th, "batch_size") else None
                            bs_display = bs if bs not in (None, "") else "auto"
                            rich_tooltip = _apply_suffix(
                                f"Whisper Recorder (TRANSCRIBING) [{device} ct2={compute_type} | {attn_impl} | chunk={chunk_display}s | batch={bs_display}]"
                            )
                            self.tray_icon.title = self._clamp_tray_tooltip(rich_tooltip)
                        else:
                            fallback = _apply_suffix("Whisper Recorder (TRANSCRIBING - 00:00)")
                            self.tray_icon.title = self._clamp_tray_tooltip(fallback)
                    except Exception:
                        fallback = _apply_suffix("Whisper Recorder (TRANSCRIBING - 00:00)")
                        self.tray_icon.title = self._clamp_tray_tooltip(fallback)
                    self.transcribing_timer_thread = threading.Thread(
                        target=self._transcribing_tooltip_updater,
                        daemon=True,
                        name="TranscribingTooltipThread",
                    )
                    self.transcribing_timer_thread.start()

            else:
                # Parar ambos contadores em outros estados
                if self.recording_timer_thread and self.recording_timer_thread.is_alive():
                    self.stop_recording_timer_event.set()
                    self.recording_timer_thread.join(timeout=1)
                if self.transcribing_timer_thread and self.transcribing_timer_thread.is_alive():
                    self.stop_transcribing_timer_event.set()
                    self.transcribing_timer_thread.join(timeout=1)

                if state_str == "IDLE" and self.core_instance_ref:
                    tooltip += f" - Record: {self.core_instance_ref.record_key.upper()} - Agent: {self.core_instance_ref.agent_key.upper()}"
                elif state_str.startswith("ERROR") and self.core_instance_ref:
                    tooltip += " - Check Logs/Settings"

            # Ajusta tooltip final conforme estado atual para evitar mensagens inconsistentes
            if state_str == "TRANSCRIBING":
                # Garante que o texto base esteja correto mesmo antes do primeiro tick
                tooltip = "Whisper Recorder (TRANSCRIBING)"
            elif state_str == "RECORDING":
                tooltip = "Whisper Recorder (RECORDING)"
            elif state_str == "LOADING_MODEL":
                tooltip = "Whisper Recorder (LOADING_MODEL)"

            tooltip_with_context = _apply_suffix(tooltip)
            self.tray_icon.title = self._clamp_tray_tooltip(tooltip_with_context)
            self.tray_icon.update_menu()

            event_name = getattr(event_obj, "name", None) if event_obj is not None else None
            if event_name:
                logging.debug("Tray icon updated for state: %s (event=%s)", state_str, event_name)
            else:
                logging.debug("Tray icon updated for state: %s", state_str)

        if warning_payload:
            warning_level = str(warning_payload.get("level", "info")).lower()
            warning_message = warning_payload.get("message")
            if not warning_message:
                details = warning_payload.get("details", {}) or {}
                preferred = details.get("preferred")
                actual = details.get("actual")
                reason = details.get("reason")
                if preferred and actual and reason:
                    warning_message = f"Fallback de dispositivo: {preferred} → {actual} ({reason})."
                elif reason:
                    warning_message = str(reason)
            if warning_message:
                log_level = logging.WARNING if warning_level == "warning" else logging.INFO
                logging.log(log_level, "Aviso de estado recebido: %s", warning_message)
                try:
                    self.show_status_tooltip(warning_message)
                except Exception as tooltip_err:
                    logging.error(
                        "Failed to update tooltip with warning: %s",
                        tooltip_err,
                        exc_info=True,
                    )

    def run_settings_gui(self):
        # Logic moved from global, adjusted to use self.
        with self.settings_window_lock:
            if self.settings_thread_running:
                logging.info("Settings window is already running. Attempting to focus.")
                if self.settings_window_instance and self.settings_window_instance.winfo_exists():
                    self.settings_window_instance.lift()
                    self.settings_window_instance.focus_force()
                return

            self.settings_thread_running = True
            try:
                ctk.set_appearance_mode("dark")
                ctk.set_default_color_theme("blue")
                settings_win = ctk.CTkToplevel(self.main_tk_root)
                self.settings_window_instance = settings_win
                settings_win.title("Whisper Recorder Settings")
                try:
                    settings_win.iconbitmap("icon.ico")
                except Exception as e:
                    logging.warning(f"Failed to set settings window icon: {e}")
                settings_win.resizable(False, True)
                settings_win.attributes("-topmost", True)

                # Calculate Center Position
                settings_win.update_idletasks()
                window_width = int(SETTINGS_WINDOW_GEOMETRY.split('x')[0])
                window_height = int(SETTINGS_WINDOW_GEOMETRY.split('x')[1])
                screen_width = settings_win.winfo_screenwidth()
                screen_height = settings_win.winfo_screenheight()
                x_cordinate = int((screen_width / 2) - (window_width / 2))
                y_cordinate = int((screen_height / 2) - (window_height / 2))
                settings_win.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
            except Exception as e:
                logging.error(f"Failed to create Toplevel for settings: {e}", exc_info=True)
                self.settings_thread_running = False
                if self.settings_window_instance:
                    self.settings_window_instance.destroy()
                    self.settings_window_instance = None
                return

            finally:
                try:
                    button_frame = ctk.CTkFrame(settings_win)
                    button_frame.pack(side="bottom", fill="x", padx=10, pady=(20, 10))

                    apply_button = ctk.CTkButton(button_frame, text="Apply and Close", command=self._apply_settings_from_ui)
                    apply_button.pack(side="right", padx=5)
                    Tooltip(apply_button, "Salva todas as configurações e fecha.")
                    close_button = ctk.CTkButton(button_frame, text="Cancel", command=self._close_settings_window, fg_color="gray50")
                    close_button.pack(side="right", padx=5)
                    Tooltip(close_button, "Descarta as alterações e fecha.")

                    restore_button = ctk.CTkButton(button_frame, text="Restore Defaults", command=self._restore_default_settings)
                    restore_button.pack(side="left", padx=5)

                    force_reregister_button = ctk.CTkButton(
                        button_frame, text="Force Hotkey Re-registration", command=self.core_instance_ref.force_reregister_hotkeys
                    )
                    force_reregister_button.pack(side="left", padx=5)
                    Tooltip(force_reregister_button, "Re-registra todos os atalhos globais.")
                except Exception as btn_err:
                    logging.error(f"Failed to create action buttons: {btn_err}", exc_info=True)

                settings_win.protocol("WM_DELETE_WINDOW", self._close_settings_window)

                self._clear_settings_context()
                self._set_settings_var("window", settings_win)
                service_values_allowed = {SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI}
                self._set_settings_meta("service_values_allowed", service_values_allowed)

                # Variables (adjust to use self.config_manager.get)
                auto_paste_var = ctk.BooleanVar(value=self.config_manager.get("auto_paste"))
                mode_var = ctk.StringVar(value=self.config_manager.get("record_mode"))
                detected_key_var = ctk.StringVar(value=self.config_manager.get("record_key").upper())
                agent_key_var = ctk.StringVar(value=self.config_manager.get("agent_key").upper())
                agent_model_var = ctk.StringVar(value=self.config_manager.get("gemini_agent_model"))
                hotkey_stability_service_enabled_var = ctk.BooleanVar(value=self.config_manager.get("hotkey_stability_service_enabled")) # Nova variável unificada
                min_transcription_duration_var = ctk.DoubleVar(value=self.config_manager.get("min_transcription_duration")) # Nova variável
                min_record_duration_var = ctk.DoubleVar(value=self.config_manager.get("min_record_duration"))
                sound_enabled_var = ctk.BooleanVar(value=self.config_manager.get("sound_enabled"))
                sound_frequency_var = ctk.StringVar(value=str(self.config_manager.get("sound_frequency")))
                sound_duration_var = ctk.StringVar(value=str(self.config_manager.get("sound_duration")))
                sound_volume_var = ctk.DoubleVar(value=self.config_manager.get("sound_volume"))
                text_correction_enabled_var = ctk.BooleanVar(value=self.config_manager.get("text_correction_enabled"))
                text_correction_service_var = ctk.StringVar(value=self.config_manager.get("text_correction_service"))
                service_display_map = {
                    "None": SERVICE_NONE,
                    "OpenRouter": SERVICE_OPENROUTER,
                    "Gemini": SERVICE_GEMINI,
                }
                self._set_settings_meta("service_display_map", service_display_map)
                text_correction_service_label_var = ctk.StringVar(
                    value=next((label for label, val in service_display_map.items()
                                if val == text_correction_service_var.get()), "None")
                )

                mode_initial = self._resolve_initial_value(
                    "record_mode",
                    var_name="record_mode",
                    coerce=lambda v: "press" if str(v).lower() == "hold" else str(v).lower(),
                    allowed={"toggle", "press"},
                )
                mode_var = ctk.StringVar(value=mode_initial)

                record_key_value = self._resolve_initial_value(
                    "record_key",
                    var_name="record_key",
                    coerce=lambda v: str(v).lower(),
                )
                detected_key_var = ctk.StringVar(value=record_key_value.upper())

                agent_key_value = self._resolve_initial_value(
                    "agent_key",
                    var_name="agent_key",
                    coerce=lambda v: str(v).lower(),
                )
                agent_key_var = ctk.StringVar(value=agent_key_value.upper())

                agent_model_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        GEMINI_AGENT_MODEL_CONFIG_KEY,
                        var_name="gemini_agent_model",
                        coerce=str,
                    )
                )

                hotkey_stability_service_enabled_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        "hotkey_stability_service_enabled",
                        var_name="hotkey_stability_service_enabled",
                        coerce=bool,
                    )
                )

                min_transcription_duration_var = ctk.DoubleVar(
                    value=self._resolve_initial_value(
                        "min_transcription_duration",
                        var_name="min_transcription_duration",
                        coerce=float,
                    )
                )
                min_record_duration_var = ctk.DoubleVar(
                    value=self._resolve_initial_value(
                        "min_record_duration",
                        var_name="min_record_duration",
                        coerce=float,
                    )
                )

                sound_enabled_var = ctk.BooleanVar(
                    value=self._resolve_initial_value("sound_enabled", var_name="sound_enabled", coerce=bool)
                )

                sound_frequency_value = self._resolve_initial_value(
                    "sound_frequency",
                    var_name="sound_frequency",
                    coerce=lambda v: int(float(v)),
                )
                sound_frequency_var = ctk.StringVar(value=str(sound_frequency_value))

                sound_duration_value = self._resolve_initial_value(
                    "sound_duration",
                    var_name="sound_duration",
                    coerce=float,
                )
                sound_duration_var = ctk.StringVar(value=str(sound_duration_value))

                sound_volume_var = ctk.DoubleVar(
                    value=self._resolve_initial_value("sound_volume", var_name="sound_volume", coerce=float)
                )

                text_correction_enabled_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        "text_correction_enabled",
                        var_name="text_correction_enabled",
                        coerce=bool,
                    )
                )

                text_correction_service_value = self._resolve_initial_value(
                    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
                    var_name="text_correction_service",
                    coerce=lambda v: str(v).lower(),
                    allowed=service_values_allowed,
                )
                text_correction_service_var = ctk.StringVar(value=text_correction_service_value)
                text_correction_service_label_var = ctk.StringVar(
                    value=next(
                        (
                            label
                            for label, val in service_display_map.items()
                            if val == text_correction_service_value
                        ),
                        "None",
                    )
                )

                openrouter_api_key_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        OPENROUTER_API_KEY_CONFIG_KEY,
                        var_name="openrouter_api_key",
                        coerce=str,
                        sensitive=True,
                    )
                )
                openrouter_model_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        OPENROUTER_MODEL_CONFIG_KEY,
                        var_name="openrouter_model",
                        coerce=str,
                    )
                )

                gemini_api_key_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        GEMINI_API_KEY_CONFIG_KEY,
                        var_name="gemini_api_key",
                        coerce=str,
                        sensitive=True,
                    )
                )
                gemini_model_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        GEMINI_MODEL_CONFIG_KEY,
                        var_name="gemini_model",
                        coerce=str,
                    )
                )

                gemini_model_options = self._resolve_initial_value(
                    GEMINI_MODEL_OPTIONS_CONFIG_KEY,
                    var_name="gemini_model_options",
                    transform=lambda v, default=DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY]: [
                        str(item) for item in (v if isinstance(v, (list, tuple, set)) else default)
                    ],
                )
                if not gemini_model_options:
                    gemini_model_options = list(DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY])
                else:
                    gemini_model_options = list(dict.fromkeys(str(item) for item in gemini_model_options))

                current_gemini_model = gemini_model_var.get()
                if current_gemini_model and current_gemini_model not in gemini_model_options:
                    gemini_model_options.insert(0, current_gemini_model)

                gemini_prompt_initial = self._resolve_initial_value(
                    GEMINI_PROMPT_CONFIG_KEY,
                    var_name="gemini_prompt",
                    coerce=str,
                )
                agent_prompt_initial = self._resolve_initial_value(
                    GEMINI_AGENT_PROMPT_CONFIG_KEY,
                    var_name="prompt_agentico",
                    coerce=str,
                )

                batch_size_value = self._resolve_initial_value(
                    "batch_size",
                    var_name="batch_size",
                    coerce=lambda v: int(float(v)),
                )
                batch_size_var = ctk.StringVar(value=str(batch_size_value))

                use_vad_var = ctk.BooleanVar(
                    value=self._resolve_initial_value("use_vad", var_name="use_vad", coerce=bool)
                )
                launch_at_startup_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        "launch_at_startup",
                        var_name="launch_at_startup",
                        coerce=bool,
                    )
                )
                vad_threshold_var = ctk.DoubleVar(
                    value=self._resolve_initial_value("vad_threshold", var_name="vad_threshold", coerce=float)
                )
                vad_silence_duration_var = ctk.DoubleVar(
                    value=self._resolve_initial_value(
                        "vad_silence_duration",
                        var_name="vad_silence_duration",
                        coerce=float,
                    )
                )
                vad_pre_speech_padding_ms_var = ctk.IntVar(
                    value=self._resolve_initial_value(
                        "vad_pre_speech_padding_ms",
                        var_name="vad_pre_speech_padding_ms",
                        coerce=int,
                    )
                )
                vad_post_speech_padding_ms_var = ctk.IntVar(
                    value=self._resolve_initial_value(
                        "vad_post_speech_padding_ms",
                        var_name="vad_post_speech_padding_ms",
                        coerce=int,
                    )
                )

                save_temp_recordings_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        SAVE_TEMP_RECORDINGS_CONFIG_KEY,
                        var_name="save_temp_recordings",
                        coerce=bool,
                    )
                )
                recordings_dir_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        RECORDINGS_DIR_CONFIG_KEY,
                        var_name="recordings_dir",
                        getter=self.config_manager.get_recordings_dir,
                        coerce=str,
                    )
                )
                display_transcripts_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        DISPLAY_TRANSCRIPTS_KEY,
                        var_name="display_transcripts_in_terminal",
                        coerce=bool,
                    )
                )

                storage_root_dir_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        STORAGE_ROOT_DIR_CONFIG_KEY,
                        var_name="storage_root_dir",
                        coerce=str,
                    )
                )
                recordings_dir_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        RECORDINGS_DIR_CONFIG_KEY,
                        var_name="recordings_dir",
                        coerce=str,
                    )
                )

                record_storage_mode_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        "record_storage_mode",
                        var_name="record_storage_mode",
                        coerce=lambda v: str(v).lower(),
                        allowed={"auto", "memory", "disk"},
                    )
                )
                max_memory_seconds_mode_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        "max_memory_seconds_mode",
                        var_name="max_memory_seconds_mode",
                        coerce=lambda v: str(v).lower(),
                        allowed={"manual", "auto"},
                    )
                )
                max_memory_seconds_var = ctk.DoubleVar(
                    value=self._resolve_initial_value(
                        "max_memory_seconds",
                        var_name="max_memory_seconds",
                        coerce=float,
                    )
                )

                chunk_length_mode_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        "chunk_length_mode",
                        var_name="chunk_length_mode",
                        getter=self.config_manager.get_chunk_length_mode,
                        coerce=lambda v: str(v).lower(),
                        allowed={"auto", "manual"},
                    )
                )
                chunk_length_sec_var = ctk.DoubleVar(
                    value=self._resolve_initial_value(
                        "chunk_length_sec",
                        var_name="chunk_length_sec",
                        getter=self.config_manager.get_chunk_length_sec,
                        coerce=float,
                    )
                )

                backend_initial = self.config_manager.get_asr_backend()
                backend_display = _backend_display_value_global(backend_initial) or DEFAULT_CONFIG.get("asr_backend", "ctranslate2")
                asr_backend_var = ctk.StringVar(value=backend_display)
                asr_model_id_var = ctk.StringVar(value=self.config_manager.get_asr_model_id())
                # New: Chunk length controls
                chunk_length_mode_var = ctk.StringVar(value=self.config_manager.get_chunk_length_mode())
                chunk_length_sec_var = ctk.DoubleVar(value=self.config_manager.get_chunk_length_sec())
                # New: Torch compile switch variable
                asr_ct2_compute_type_var = ctk.StringVar(value=self.config_manager.get_asr_ct2_compute_type())
                models_storage_dir_var = ctk.StringVar(value=self.config_manager.get_models_storage_dir())
                deps_install_dir_var = ctk.StringVar(value=self.config_manager.get_deps_install_dir())
                asr_cache_dir_var = ctk.StringVar(value=self.config_manager.get_asr_cache_dir())
                python_packages_dir_var = ctk.StringVar(value=self.config_manager.get_python_packages_dir())
                vad_models_dir_var = ctk.StringVar(value=self.config_manager.get_vad_models_dir())
                hf_cache_dir_var = ctk.StringVar(value=self.config_manager.get_hf_cache_dir())

                for name, var in [
                    ("auto_paste_var", auto_paste_var),
                    ("mode_var", mode_var),
                    ("detected_key_var", detected_key_var),
                    ("agent_key_var", agent_key_var),
                    ("agent_model_var", agent_model_var),
                    ("hotkey_stability_service_enabled_var", hotkey_stability_service_enabled_var),
                    ("min_transcription_duration_var", min_transcription_duration_var),
                    ("min_record_duration_var", min_record_duration_var),
                    ("sound_enabled_var", sound_enabled_var),
                    ("sound_frequency_var", sound_frequency_var),
                    ("sound_duration_var", sound_duration_var),
                    ("sound_volume_var", sound_volume_var),
                    ("text_correction_enabled_var", text_correction_enabled_var),
                    ("text_correction_service_var", text_correction_service_var),
                    ("text_correction_service_label_var", text_correction_service_label_var),
                    ("openrouter_api_key_var", openrouter_api_key_var),
                    ("openrouter_model_var", openrouter_model_var),
                    ("gemini_api_key_var", gemini_api_key_var),
                    ("gemini_model_var", gemini_model_var),
                    ("batch_size_var", batch_size_var),
                    ("use_vad_var", use_vad_var),
                    ("launch_at_startup_var", launch_at_startup_var),
                    ("vad_threshold_var", vad_threshold_var),
                    ("vad_silence_duration_var", vad_silence_duration_var),
                    ("save_temp_recordings_var", save_temp_recordings_var),
                    ("display_transcripts_var", display_transcripts_var),
                    ("storage_root_dir_var", storage_root_dir_var),
                    ("recordings_dir_var", recordings_dir_var),
                    ("record_storage_mode_var", record_storage_mode_var),
                    ("max_memory_seconds_mode_var", max_memory_seconds_mode_var),
                    ("max_memory_seconds_var", max_memory_seconds_var),
                    ("chunk_length_mode_var", chunk_length_mode_var),
                    ("chunk_length_sec_var", chunk_length_sec_var),
                    ("asr_backend_var", asr_backend_var),
                    ("asr_model_id_var", asr_model_id_var),
                    ("asr_ct2_compute_type_var", asr_ct2_compute_type_var),
                    ("models_storage_dir_var", models_storage_dir_var),
                    ("deps_install_dir_var", deps_install_dir_var),
                    ("asr_cache_dir_var", asr_cache_dir_var),
                    ("python_packages_dir_var", python_packages_dir_var),
                    ("vad_models_dir_var", vad_models_dir_var),
                    ("hf_cache_dir_var", hf_cache_dir_var),
                    ("recordings_dir_var", recordings_dir_var),
                ]:
                    self._set_settings_var(name, var)

                # Compute device selection variable
                available_devices = get_available_devices_for_ui()
                self._set_settings_meta("available_devices", available_devices)
                current_device_selection = "Auto-select (Recommended)"
                asr_compute_device_value = self._resolve_initial_value(
                    ASR_COMPUTE_DEVICE_CONFIG_KEY,
                    var_name="asr_compute_device",
                    getter=self.config_manager.get_asr_compute_device,
                    coerce=lambda v: str(v).lower(),
                    allowed={"auto", "cpu", "cuda"},
                )
                gpu_index_value = self._resolve_initial_value(
                    GPU_INDEX_CONFIG_KEY,
                    var_name="gpu_index",
                    coerce=lambda v: int(float(v)),
                )
                if asr_compute_device_value == "cpu":
                    current_device_selection = "Force CPU"
                elif asr_compute_device_value == "cuda" and gpu_index_value >= 0:
                    for dev in available_devices:
                        if dev.startswith(f"GPU {gpu_index_value}"):
                            current_device_selection = dev
                            break
                asr_compute_device_var = ctk.StringVar(value=current_device_selection)
                self._set_settings_var("asr_compute_device_var", asr_compute_device_var)
                self._gpu_selection_var = asr_compute_device_var

                # Internal GUI functions (detect_key_task_internal, apply_settings, close_settings, etc.)
                # Will need to be adapted to call methods of self.core_instance_ref and self.config_manager
                # Example of adapted apply_settings:
                def apply_settings():
                    """Aplica as configurações respeitando as dependências do pipeline de ASR.

                    A sequência explicitamente segue ``backend → modelo → device → quantização``
                    porque o ``TranscriptionHandler`` depende do backend correto antes de
                    aceitar um modelo e, por consequência, o dispositivo informado precisa
                    estar sincronizado com as opções de quantização (ex.: ``ct2`` habilita
                    perfis diferentes). Toda a persistência é delegada para
                    ``AppCore.apply_settings_from_external``, que repassa os valores ao
                    ``ConfigManager`` e notifica os subsistemas correspondentes. Fluxo
                    validado com as frentes de UX e engenharia para preservar a coerência
                    entre a experiência da janela de ajustes e os requisitos técnicos de
                    carregamento de modelo.
                    """
                    logging.info("Apply settings clicked (in Tkinter thread).")
                    # State validations (moved to AppCore or handled via callbacks)
                    if self._get_core_state() in ["RECORDING", "TRANSCRIBING", "LOADING_MODEL"]:
                        messagebox.showwarning("Apply Settings", "Cannot apply while recording/transcribing/loading model.", parent=settings_win)
                        return

                    # Collect UI values
                    key_to_apply = detected_key_var.get().lower() if detected_key_var.get() != "PRESS KEY..." else self.config_manager.get("record_key")
                    mode_to_apply = mode_var.get()
                    auto_paste_to_apply = auto_paste_var.get() # Now unified
                    agent_key_to_apply = agent_key_var.get().lower() if agent_key_var.get() != "PRESS KEY..." else self.config_manager.get("agent_key")
                    model_to_apply = agent_model_var.get()
                    hotkey_stability_service_enabled_to_apply = hotkey_stability_service_enabled_var.get() # Coleta o valor da nova variável
                    sound_enabled_to_apply = sound_enabled_var.get()
                    sound_freq_to_apply = self._safe_get_int(sound_frequency_var, "Frequência do Som", settings_win)
                    if sound_freq_to_apply is None:
                        return
                    sound_duration_to_apply = self._safe_get_float(sound_duration_var, "Sound Duration", settings_win)
                    if sound_duration_to_apply is None:
                        return
                    sound_volume_to_apply = self._safe_get_float(sound_volume_var, "Volume do Som", settings_win)
                    if sound_volume_to_apply is None:
                        return
                    text_correction_enabled_to_apply = text_correction_enabled_var.get()
                    text_correction_service_to_apply = text_correction_service_var.get()
                    openrouter_api_key_to_apply = openrouter_api_key_var.get()
                    openrouter_model_to_apply = openrouter_model_var.get()
                    gemini_api_key_to_apply = gemini_api_key_var.get()
                    gemini_model_to_apply = gemini_model_var.get()
                    gemini_prompt_correction_to_apply = gemini_prompt_correction_textbox.get("1.0", "end-1c")
                    agentico_prompt_to_apply = agentico_prompt_textbox.get("1.0", "end-1c")
                    batch_size_to_apply = self._safe_get_int(batch_size_var, "Batch Size", settings_win)
                    if batch_size_to_apply is None:
                        return
                    min_transcription_duration_to_apply = self._safe_get_float(min_transcription_duration_var, "Minimum Transcription Duration", settings_win)
                    if min_transcription_duration_to_apply is None:
                        return
                    min_record_duration_to_apply = self._safe_get_float(min_record_duration_var, "Minimum Recording Duration", settings_win)
                    if min_record_duration_to_apply is None:
                        return
                    use_vad_to_apply = use_vad_var.get()
                    vad_threshold_to_apply = self._safe_get_float(vad_threshold_var, "Limiar do VAD", settings_win)
                    if vad_threshold_to_apply is None:
                        return
                    vad_silence_duration_to_apply = self._safe_get_float(vad_silence_duration_var, "Silence Duration", settings_win)
                    if vad_silence_duration_to_apply is None:
                        return
                    save_temp_recordings_to_apply = save_temp_recordings_var.get()
                    display_transcripts_to_apply = display_transcripts_var.get()
                    max_memory_seconds_mode_to_apply = max_memory_seconds_mode_var.get()
                    max_memory_seconds_to_apply = self._safe_get_float(max_memory_seconds_var, "Max Memory Retention", settings_win)
                    if max_memory_seconds_to_apply is None:
                        return
                    asr_backend_to_apply = asr_backend_var.get()
                    asr_model_id_to_apply = asr_model_id_var.get()
                    asr_compute_device_to_apply = "auto"
                    asr_ct2_compute_type_to_apply = asr_ct2_compute_type_var.get()
                    asr_cache_dir_to_apply = asr_cache_dir_var.get()
                    try:
                        Path(asr_cache_dir_to_apply).mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        messagebox.showerror("Invalid Path", f"ASR cache directory is invalid:\n{e}", parent=settings_win)
                        return

                    recordings_dir_to_apply = recordings_dir_var.get()
                    try:
                        Path(recordings_dir_to_apply).mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        messagebox.showerror(
                            "Invalid Path",
                            f"Recording directory is invalid:\n{exc}",
                            parent=settings_win,
                        )
                        return

                    python_packages_dir_to_apply = python_packages_dir_var.get()
                    try:
                        Path(python_packages_dir_to_apply).mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        messagebox.showerror(
                            "Invalid Path",
                            f"Python packages directory is invalid:\n{exc}",
                            parent=settings_win,
                        )
                        return

                    vad_models_dir_to_apply = vad_models_dir_var.get()
                    try:
                        Path(vad_models_dir_to_apply).mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        messagebox.showerror(
                            "Invalid Path",
                            f"VAD models directory is invalid:\n{exc}",
                            parent=settings_win,
                        )
                        return

                    hf_cache_dir_to_apply = hf_cache_dir_var.get()
                    try:
                        Path(hf_cache_dir_to_apply).mkdir(parents=True, exist_ok=True)
                    except Exception as exc:
                        messagebox.showerror(
                            "Invalid Path",
                            f"Hugging Face cache directory is invalid:\n{exc}",
                            parent=settings_win,
                        )
                        return

                    # Logic for converting UI to GPU index
                    selected_device_str = asr_compute_device_var.get()
                    gpu_index_to_apply = -1 # Default to "Auto-select"
                    if "Force CPU" in selected_device_str:
                        asr_compute_device_to_apply = "cpu"
                    elif selected_device_str.startswith("GPU"):
                        asr_compute_device_to_apply = "cuda"
                        try:
                            gpu_index_to_apply = int(selected_device_str.split(":")[0].replace("GPU", "").strip())
                        except (ValueError, IndexError):
                            messagebox.showerror("Invalid Value", "Invalid GPU index.", parent=settings_win)
                            return
                    asr_ct2_compute_type_to_apply = asr_ct2_compute_type_var.get()
                    asr_cache_dir_to_apply = asr_cache_dir_var.get()

                    models_text = gemini_models_textbox.get("1.0", "end-1c")
                    new_models_list = [line.strip() for line in models_text.split("\n") if line.strip()]
                    if not new_models_list:
                        messagebox.showwarning("Invalid Value", "The model list cannot be empty. Please add at least one model.", parent=settings_win)
                        return

                    # Call AppCore method to apply settings
                    self.apply_settings_payload(
                        {
                            "new_key": key_to_apply,
                            "new_mode": mode_to_apply,
                            "new_auto_paste": auto_paste_to_apply,
                            "new_sound_enabled": sound_enabled_to_apply,
                            "new_sound_frequency": sound_freq_to_apply,
                            "new_sound_duration": sound_duration_to_apply,
                            "new_sound_volume": sound_volume_to_apply,
                            "new_agent_key": agent_key_to_apply,
                            "new_text_correction_enabled": text_correction_enabled_to_apply,
                            "new_text_correction_service": text_correction_service_to_apply,
                            "new_openrouter_api_key": openrouter_api_key_to_apply,
                            "new_openrouter_model": openrouter_model_to_apply,
                            "new_gemini_api_key": gemini_api_key_to_apply,
                            "new_gemini_model": gemini_model_to_apply,
                            "new_gemini_prompt": gemini_prompt_correction_to_apply,
                            "prompt_agentico": agentico_prompt_to_apply,
                            "new_agent_model": model_to_apply,
                            "new_gemini_model_options": new_models_list,
                            "new_batch_size": batch_size_to_apply,
                            "new_gpu_index": gpu_index_to_apply,
                            "new_hotkey_stability_service_enabled": hotkey_stability_service_enabled_to_apply, # Nova configuração unificada
                            "new_min_transcription_duration": min_transcription_duration_to_apply,
                            "new_min_record_duration": min_record_duration_to_apply,
                            "new_save_temp_recordings": save_temp_recordings_to_apply,
                            "new_record_storage_mode": record_storage_mode_var.get(),
                            "new_max_memory_seconds_mode": max_memory_seconds_mode_to_apply,
                            "new_max_memory_seconds": max_memory_seconds_to_apply,
                            "new_use_vad": use_vad_to_apply,
                            "new_vad_threshold": vad_threshold_to_apply,
                            "new_vad_silence_duration": vad_silence_duration_to_apply,
                            "new_display_transcripts_in_terminal": display_transcripts_to_apply,
                            "new_launch_at_startup": launch_at_startup_var.get(),
                            # New chunk settings
                            "new_chunk_length_mode": chunk_length_mode_var.get(),
                            "new_chunk_length_sec": float(chunk_length_sec_var.get()),
                            "new_asr_backend": asr_backend_to_apply,
                            "new_asr_model_id": asr_model_id_to_apply,
                            "new_asr_compute_device": asr_compute_device_to_apply,
                            "new_asr_ct2_compute_type": asr_ct2_compute_type_to_apply,
                            "new_asr_cache_dir": asr_cache_dir_to_apply,
                            "new_storage_root_dir": storage_root_dir_var.get(),
                            "new_recordings_dir": recordings_dir_var.get(),
                            "new_python_packages_dir": python_packages_dir_to_apply,
                            "new_vad_models_dir": vad_models_dir_to_apply,
                            "new_hf_cache_dir": hf_cache_dir_to_apply,
                        }
                    )
                    self._close_settings_window() # Call class method

                def restore_defaults():
                    if not messagebox.askyesno(
                        "Restore Defaults",
                        "Are you sure you want to restore all settings to their default values?",
                        parent=settings_win,
                    ):
                        return

                    self.core_instance_ref.apply_settings_from_external(**DEFAULT_CONFIG)

                    auto_paste_var.set(DEFAULT_CONFIG["auto_paste"])
                    mode_var.set(DEFAULT_CONFIG["record_mode"])
                    detected_key_var.set(DEFAULT_CONFIG["record_key"].upper())
                    agent_key_var.set(DEFAULT_CONFIG["agent_key"].upper())
                    agent_model_var.set(DEFAULT_CONFIG[GEMINI_AGENT_MODEL_CONFIG_KEY])
                    hotkey_stability_service_enabled_var.set(DEFAULT_CONFIG["hotkey_stability_service_enabled"])
                    min_transcription_duration_var.set(DEFAULT_CONFIG["min_transcription_duration"])
                    min_record_duration_var.set(DEFAULT_CONFIG["min_record_duration"])
                    sound_enabled_var.set(DEFAULT_CONFIG["sound_enabled"])
                    sound_frequency_var.set(str(DEFAULT_CONFIG["sound_frequency"]))
                    sound_duration_var.set(str(DEFAULT_CONFIG["sound_duration"]))
                    sound_volume_var.set(DEFAULT_CONFIG["sound_volume"])
                    text_correction_enabled_var.set(DEFAULT_CONFIG["text_correction_enabled"])
                    text_correction_service_var.set(DEFAULT_CONFIG[TEXT_CORRECTION_SERVICE_CONFIG_KEY])
                    text_correction_service_label_var.set(
                        next(
                            (
                                label
                                for label, val in service_display_map.items()
                                if val == DEFAULT_CONFIG[TEXT_CORRECTION_SERVICE_CONFIG_KEY]
                            ),
                            "None",
                        )
                    )
                    # Sincroniza os campos de correção de texto caso os widgets existam
                    try:
                        service_menu  # Verifica se os widgets foram criados
                    except NameError:
                        pass
                    else:
                        service_menu.set(text_correction_service_label_var.get())
                        self._update_text_correction_fields()
                    openrouter_api_key_var.set(DEFAULT_CONFIG[OPENROUTER_API_KEY_CONFIG_KEY])
                    openrouter_model_var.set(DEFAULT_CONFIG[OPENROUTER_MODEL_CONFIG_KEY])
                    gemini_api_key_var.set(DEFAULT_CONFIG[GEMINI_API_KEY_CONFIG_KEY])
                    gemini_model_var.set(DEFAULT_CONFIG[GEMINI_MODEL_CONFIG_KEY])
                    agent_model_var.set(DEFAULT_CONFIG[GEMINI_AGENT_MODEL_CONFIG_KEY])
                    try:
                        gemini_model_menu
                    except NameError:
                        pass
                    else:
                        gemini_model_menu.configure(values=DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY])
                        gemini_model_menu.set(DEFAULT_CONFIG[GEMINI_MODEL_CONFIG_KEY])
                    try:
                        agent_model_menu
                    except NameError:
                        pass
                    else:
                        agent_model_menu.configure(values=DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY])
                        agent_model_menu.set(DEFAULT_CONFIG[GEMINI_AGENT_MODEL_CONFIG_KEY])
                    gemini_model_options[:] = list(DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY])
                    asr_model_id_var.set(DEFAULT_CONFIG[ASR_MODEL_ID_CONFIG_KEY])
                    asr_model_display_var = self._get_settings_var("asr_model_display_var")
                    id_to_display_map = self._get_settings_meta("id_to_display", {})
                    if asr_model_display_var is not None:
                        asr_model_display_var.set(
                            id_to_display_map.get(
                                DEFAULT_CONFIG[ASR_MODEL_ID_CONFIG_KEY],
                                DEFAULT_CONFIG[ASR_MODEL_ID_CONFIG_KEY],
                            )
                        )
                    gemini_prompt_correction_textbox.delete("1.0", "end")
                    gemini_prompt_correction_textbox.insert("1.0", DEFAULT_CONFIG[GEMINI_PROMPT_CONFIG_KEY])
                    agentico_prompt_textbox.delete("1.0", "end")
                    agentico_prompt_textbox.insert("1.0", DEFAULT_CONFIG[GEMINI_AGENT_PROMPT_CONFIG_KEY])
                    gemini_models_textbox.delete("1.0", "end")
                    gemini_models_textbox.insert(
                        "1.0",
                        "\n".join(DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY]),
                    )
                    batch_size_var.set(str(DEFAULT_CONFIG["batch_size"]))
                    asr_backend_var.set(DEFAULT_CONFIG[ASR_BACKEND_CONFIG_KEY])
                    asr_model_id_var.set(DEFAULT_CONFIG[ASR_MODEL_ID_CONFIG_KEY])
                    asr_model_display_var = self._get_settings_var("asr_model_display_var")
                    id_to_display_map = self._get_settings_meta("id_to_display", {})
                    if asr_model_display_var is not None:
                        asr_model_display_var.set(
                            id_to_display_map.get(
                                DEFAULT_CONFIG[ASR_MODEL_ID_CONFIG_KEY],
                                DEFAULT_CONFIG[ASR_MODEL_ID_CONFIG_KEY],
                            )
                        )

                    if DEFAULT_CONFIG[ASR_COMPUTE_DEVICE_CONFIG_KEY] == "cpu":
                        asr_compute_device_var.set("Force CPU")
                    elif (
                        DEFAULT_CONFIG[ASR_COMPUTE_DEVICE_CONFIG_KEY] == "cuda"
                        and DEFAULT_CONFIG[GPU_INDEX_CONFIG_KEY] >= 0
                    ):
                        found = "Auto-select (Recommended)"
                        for dev in available_devices:
                            if dev.startswith(f"GPU {DEFAULT_CONFIG[GPU_INDEX_CONFIG_KEY]}"):
                                found = dev
                                break
                        asr_compute_device_var.set(found)
                    else:
                        asr_compute_device_var.set("Auto-select (Recommended)")

                    use_vad_var.set(DEFAULT_CONFIG["use_vad"])
                    vad_threshold_var.set(DEFAULT_CONFIG["vad_threshold"])
                    vad_silence_duration_var.set(DEFAULT_CONFIG["vad_silence_duration"])
                    save_temp_recordings_var.set(DEFAULT_CONFIG[SAVE_TEMP_RECORDINGS_CONFIG_KEY])
                    storage_root_dir_var.set(DEFAULT_CONFIG[STORAGE_ROOT_DIR_CONFIG_KEY])
                    recordings_dir_var.set(DEFAULT_CONFIG[RECORDINGS_DIR_CONFIG_KEY])
                    display_transcripts_var.set(DEFAULT_CONFIG[DISPLAY_TRANSCRIPTS_KEY])
                    record_storage_mode_var.set(DEFAULT_CONFIG["record_storage_mode"])
                    max_memory_seconds_var.set(DEFAULT_CONFIG["max_memory_seconds"])
                    max_memory_seconds_mode_var.set(DEFAULT_CONFIG["max_memory_seconds_mode"])
                    launch_at_startup_var.set(DEFAULT_CONFIG["launch_at_startup"])
                    asr_backend_var.set(DEFAULT_CONFIG[ASR_BACKEND_CONFIG_KEY])
                    asr_compute_device_var.set(DEFAULT_CONFIG[ASR_COMPUTE_DEVICE_CONFIG_KEY])
                    asr_ct2_compute_type_var.set(DEFAULT_CONFIG[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY])
                    asr_cache_dir_var.set(DEFAULT_CONFIG[ASR_CACHE_DIR_CONFIG_KEY])

                    self.config_manager.save_config()

                scrollable_frame = ctk.CTkScrollableFrame(settings_win, fg_color="transparent")
                scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

                # The main_frame is now the scrollable_frame for all internal widgets
                main_frame = scrollable_frame

                # --- General Settings Section ---
                general_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
                general_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(general_frame, text="Configurações Gerais", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                # Record Hotkey
                key_frame = ctk.CTkFrame(general_frame)
                key_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(key_frame, text="Atalho de Gravação:").pack(side="left", padx=(5, 10))
                key_display = ctk.CTkLabel(key_frame, textvariable=detected_key_var, fg_color="gray20", corner_radius=5, width=120)
                key_display.pack(side="left", padx=5)
                Tooltip(key_display, "Atalho atual de gravação.")
                
                detect_key_button = ctk.CTkButton(
                    key_frame,
                    text="Detectar Tecla",
                    command=lambda: self._start_key_detection_for("detected_key_var"),
                )
                detect_key_button.pack(side="left", padx=5)
                Tooltip(detect_key_button, "Captura um novo atalho de gravação.")

                # Agent Hotkey (Moved here)
                agent_key_frame = ctk.CTkFrame(general_frame)
                agent_key_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(agent_key_frame, text="Atalho do Agente:").pack(side="left", padx=(5, 10))
                agent_key_display = ctk.CTkLabel(agent_key_frame, textvariable=agent_key_var, fg_color="gray20", corner_radius=5, width=120)
                agent_key_display.pack(side="left", padx=5)
                Tooltip(agent_key_display, "Atalho atual do modo agente.")
                detect_agent_key_button = ctk.CTkButton(
                    agent_key_frame,
                    text="Detectar Tecla",
                    command=lambda: self._start_key_detection_for("agent_key_var"),
                )
                detect_agent_key_button.pack(side="left", padx=5)
                Tooltip(detect_agent_key_button, "Captura um novo atalho do agente.")

                # Recording Mode
                mode_frame = ctk.CTkFrame(general_frame)
                mode_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(mode_frame, text="Modo de Gravação:").pack(side="left", padx=(5, 10))
                toggle_rb = ctk.CTkRadioButton(mode_frame, text="Alternar", variable=mode_var, value="toggle")
                toggle_rb.pack(side="left", padx=5)
                Tooltip(toggle_rb, "Pressione uma vez para iniciar ou parar.")
                hold_rb = ctk.CTkRadioButton(mode_frame, text="Segurar", variable=mode_var, value="press")
                hold_rb.pack(side="left", padx=5)
                Tooltip(hold_rb, "Grava enquanto a tecla estiver pressionada.")

                # Auto-Paste
                paste_frame = ctk.CTkFrame(general_frame)
                paste_frame.pack(fill="x", pady=5)
                paste_switch = ctk.CTkSwitch(paste_frame, text="Auto-colar", variable=auto_paste_var)
                paste_switch.pack(side="left", padx=5)
                Tooltip(paste_switch, "Cola automaticamente a transcrição.")

                # Hotkey Stability Service
                stability_service_frame = ctk.CTkFrame(general_frame)
                stability_service_frame.pack(fill="x", pady=5)
                stability_switch = ctk.CTkSwitch(stability_service_frame, text="Ativar Serviço de Estabilidade de Atalhos", variable=hotkey_stability_service_enabled_var)
                stability_switch.pack(side="left", padx=5)
                Tooltip(stability_switch, "Mantém os atalhos ativos mesmo sem foco.")

                startup_frame = ctk.CTkFrame(general_frame)
                startup_frame.pack(fill="x", pady=5)
                startup_switch = ctk.CTkSwitch(startup_frame, text="Iniciar com o Windows", variable=launch_at_startup_var)
                startup_switch.pack(side="left", padx=5)
                Tooltip(startup_switch, "Inicia automaticamente com o Windows.")

                # --- Storage Settings Section ---
                storage_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                storage_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(
                    storage_frame,
                    text="Armazenamento",
                    font=ctk.CTkFont(weight="bold"),
                ).pack(pady=(5, 10), anchor="w")

                storage_root_frame = ctk.CTkFrame(storage_frame)
                storage_root_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(storage_root_frame, text="Pasta raiz de dados:").pack(side="left", padx=(5, 10))
                storage_root_entry = ctk.CTkEntry(
                    storage_root_frame,
                    textvariable=storage_root_dir_var,
                    width=260,
                )
                storage_root_entry.pack(side="left", padx=5, fill="x", expand=True)
                Tooltip(
                    storage_root_entry,
                    "Diretório base usado para modelos e outros artefatos pesados.",
                )

                recordings_dir_frame = ctk.CTkFrame(storage_frame)
                recordings_dir_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(recordings_dir_frame, text="Diretório de gravações:").pack(side="left", padx=(5, 10))
                recordings_dir_entry = ctk.CTkEntry(
                    recordings_dir_frame,
                    textvariable=recordings_dir_var,
                    width=260,
                )
                recordings_dir_entry.pack(side="left", padx=5, fill="x", expand=True)
                Tooltip(
                    recordings_dir_entry,
                    "Local onde os arquivos WAV temporários e salvos serão armazenados.",
                )

                # --- Sound Settings Section ---
                sound_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                sound_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(sound_frame, text="Alertas Sonoros", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")
                
                sound_enabled_frame = ctk.CTkFrame(sound_frame)
                sound_enabled_frame.pack(fill="x", pady=5)
                sound_switch = ctk.CTkSwitch(sound_enabled_frame, text="Ativar Sons", variable=sound_enabled_var)
                sound_switch.pack(side="left", padx=5)
                Tooltip(sound_switch, "Reproduz um beep ao iniciar ou parar a gravação.")

                sound_details_frame = ctk.CTkFrame(sound_frame)
                sound_details_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(sound_details_frame, text="Frequência (Hz):").pack(side="left", padx=(5, 10))
                freq_entry = ctk.CTkEntry(sound_details_frame, textvariable=sound_frequency_var, width=60)
                freq_entry.pack(side="left", padx=5)
                Tooltip(freq_entry, "Frequência do beep em hertz.")
                ctk.CTkLabel(sound_details_frame, text="Duração (s):").pack(side="left", padx=(5, 10))
                duration_entry = ctk.CTkEntry(sound_details_frame, textvariable=sound_duration_var, width=60)
                duration_entry.pack(side="left", padx=5)
                Tooltip(duration_entry, "Duração do beep em segundos.")
                ctk.CTkLabel(sound_details_frame, text="Volume:").pack(side="left", padx=(5, 10))
                volume_slider = ctk.CTkSlider(sound_details_frame, from_=0.0, to=1.0, variable=sound_volume_var)
                volume_slider.pack(side="left", padx=5, fill="x", expand=True)
                Tooltip(volume_slider, "Volume do beep.")

                # --- Text Correction (AI Services) Section ---
                ai_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                ai_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(ai_frame, text="Correção de Texto (Serviços de IA)", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                text_correction_frame = ctk.CTkFrame(ai_frame)
                text_correction_frame.pack(fill="x", pady=5)
                correction_switch = ctk.CTkSwitch(
                    text_correction_frame,
                    text="Ativar Correção de Texto",
                    variable=text_correction_enabled_var,
                    command=self._update_text_correction_fields,
                )
                correction_switch.pack(side="left", padx=5)
                Tooltip(correction_switch, "Usa um serviço de IA para refinar o texto.")

                service_frame = ctk.CTkFrame(ai_frame)
                service_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(service_frame, text="Serviço:").pack(side="left", padx=(5, 10))
                service_menu = ctk.CTkOptionMenu(
                    service_frame,
                    variable=text_correction_service_label_var,
                    values=list(service_display_map.keys()),
                    command=self._on_service_menu_change,
                )
                service_menu.pack(side="left", padx=5)
                self._set_settings_var("service_menu", service_menu)
                Tooltip(service_menu, "Selecione o serviço de correção de texto.")
                service_menu.set(text_correction_service_label_var.get())

                # --- OpenRouter Settings ---
                openrouter_frame = ctk.CTkFrame(ai_frame)
                openrouter_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(openrouter_frame, text="Chave OpenRouter:").pack(side="left", padx=(5, 10))
                openrouter_key_entry = ctk.CTkEntry(openrouter_frame, textvariable=openrouter_api_key_var, show="*", width=250)
                openrouter_key_entry.pack(side="left", padx=5)
                self._set_settings_var("openrouter_key_entry", openrouter_key_entry)
                Tooltip(openrouter_key_entry, "Chave da API OpenRouter.")
                ctk.CTkLabel(openrouter_frame, text="Modelo OpenRouter:").pack(side="left", padx=(5, 10))
                openrouter_model_entry = ctk.CTkEntry(openrouter_frame, textvariable=openrouter_model_var, width=200)
                openrouter_model_entry.pack(side="left", padx=5)
                self._set_settings_var("openrouter_model_entry", openrouter_model_entry)
                Tooltip(openrouter_model_entry, "Modelo utilizado no OpenRouter.")

                # --- Gemini Settings ---
                gemini_frame = ctk.CTkFrame(ai_frame)
                gemini_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_frame, text="Chave Gemini:").pack(side="left", padx=(5, 10))
                gemini_key_entry = ctk.CTkEntry(gemini_frame, textvariable=gemini_api_key_var, show="*", width=250)
                gemini_key_entry.pack(side="left", padx=5)
                self._set_settings_var("gemini_key_entry", gemini_key_entry)
                Tooltip(gemini_key_entry, "Chave da API Gemini.")
                ctk.CTkLabel(gemini_frame, text="Modelo Gemini:").pack(side="left", padx=(5, 10))
                gemini_model_menu = ctk.CTkOptionMenu(gemini_frame, variable=gemini_model_var, values=gemini_model_options)
                gemini_model_menu.pack(side="left", padx=5)
                self._set_settings_var("gemini_model_menu", gemini_model_menu)
                Tooltip(gemini_model_menu, "Modelo utilizado nas requisições Gemini.")

                ctk.CTkLabel(gemini_frame, text="Modelo do Agente:").pack(side="left", padx=(5, 10))
                agent_model_menu = ctk.CTkOptionMenu(
                    gemini_frame,
                    variable=agent_model_var,
                    values=gemini_model_options,
                )
                agent_model_menu.pack(side="left", padx=5)
                self._set_settings_var("agent_model_menu", agent_model_menu)
                Tooltip(agent_model_menu, "Modelo dedicado às ações do modo agente.")

                # --- Gemini Prompt ---
                gemini_prompt_frame = ctk.CTkFrame(ai_frame)
                gemini_prompt_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_prompt_frame, text="Prompt de Correção (Gemini):").pack(anchor="w", pady=(5,0))
                gemini_prompt_correction_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=100, wrap="word")
                gemini_prompt_correction_textbox.pack(fill="x", expand=True, pady=5)
                gemini_prompt_correction_textbox.insert("1.0", gemini_prompt_initial)
                self._set_settings_var("gemini_prompt_correction_textbox", gemini_prompt_correction_textbox)
                Tooltip(gemini_prompt_correction_textbox, "Prompt usado para refinar o texto.")

                ctk.CTkLabel(gemini_prompt_frame, text="Prompt do Modo Agente:").pack(anchor="w", pady=(5,0))
                agentico_prompt_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                agentico_prompt_textbox.pack(fill="x", expand=True, pady=5)
                agentico_prompt_textbox.insert("1.0", agent_prompt_initial)
                self._set_settings_var("agentico_prompt_textbox", agentico_prompt_textbox)
                Tooltip(agentico_prompt_textbox, "Prompt executado no modo agente.")

                ctk.CTkLabel(gemini_prompt_frame, text="Modelos Gemini (um por linha):").pack(anchor="w", pady=(5,0))
                gemini_models_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                gemini_models_textbox.pack(fill="x", expand=True, pady=5)
                gemini_models_textbox.insert("1.0", "\n".join(gemini_model_options))
                self._set_settings_var("gemini_models_textbox", gemini_models_textbox)
                Tooltip(gemini_models_textbox, "Lista de modelos para tentativa, um por linha.")

                transcription_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                transcription_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(
                    transcription_frame,
                    text="Transcrição",
                    font=ctk.CTkFont(weight="bold"),
                ).pack(pady=(5, 10), anchor="w")

                asr_frame = ctk.CTkFrame(transcription_frame, fg_color="transparent")
                asr_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(
                    asr_frame,
                    text="Modelo ASR",
                    font=ctk.CTkFont(weight="bold"),
                ).pack(anchor="w", pady=(5, 10))
                self._set_settings_var("transcription_frame", transcription_frame)
                self._set_settings_var("asr_frame", asr_frame)

        self._build_asr_section(
            settings_win=settings_win,
            asr_frame=asr_frame,
            transcription_frame=transcription_frame,
            available_devices=available_devices,
            asr_backend_var=asr_backend_var,
            asr_model_id_var=asr_model_id_var,
            asr_compute_device_var=asr_compute_device_var,
            asr_ct2_compute_type_var=asr_ct2_compute_type_var,
            models_storage_dir_var=models_storage_dir_var,
            deps_install_dir_var=deps_install_dir_var,
            storage_root_dir_var=storage_root_dir_var,
            recordings_dir_var=recordings_dir_var,
            asr_cache_dir_var=asr_cache_dir_var,
            ui_elements={},
        )

        # New: Chunk Length Mode
        chunk_mode_frame = ctk.CTkFrame(transcription_frame)
        chunk_mode_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(chunk_mode_frame, text="Modo do Tamanho do Bloco:").pack(side="left", padx=(5, 10))
        chunk_mode_menu = ctk.CTkOptionMenu(
            chunk_mode_frame,
            variable=chunk_length_mode_var,
            values=["auto", "manual"],
            command=self._on_chunk_mode_change,
        )
        chunk_mode_menu.pack(side="left", padx=5)
        Tooltip(chunk_mode_menu, "Define como o tamanho do bloco é calculado.")

        # New: Chunk Length (sec)
        chunk_len_frame = ctk.CTkFrame(transcription_frame)
        chunk_len_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(chunk_len_frame, text="Duração do Bloco (s):").pack(side="left", padx=(5, 10))
        chunk_len_entry = ctk.CTkEntry(chunk_len_frame, textvariable=chunk_length_sec_var, width=80)
        chunk_len_entry.pack(side="left", padx=5)
        self._set_settings_var("chunk_len_entry", chunk_len_entry)
        Tooltip(chunk_len_entry, "Duração fixa do bloco quando em modo manual.")

        # New: Ignore Transcriptions Shorter Than
        min_transcription_duration_frame = ctk.CTkFrame(transcription_frame)
        min_transcription_duration_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            min_transcription_duration_frame,
            text="Ignorar transcrições menores que (s):",
        ).pack(side="left", padx=(5, 10))
        min_transcription_duration_entry = ctk.CTkEntry(
            min_transcription_duration_frame,
            textvariable=min_transcription_duration_var,
            width=80,
        )
        min_transcription_duration_entry.pack(side="left", padx=5)
        Tooltip(min_transcription_duration_entry, "Descarta segmentos menores que isso.")

        min_record_duration_frame = ctk.CTkFrame(transcription_frame)
        min_record_duration_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            min_record_duration_frame,
            text="Duração mínima da gravação (s):",
        ).pack(side="left", padx=(5, 10))
        min_record_duration_entry = ctk.CTkEntry(
            min_record_duration_frame,
            textvariable=min_record_duration_var,
            width=80,
        )
        min_record_duration_entry.pack(side="left", padx=5)
        Tooltip(min_record_duration_entry, "Descarta gravações menores que isso.")

        self._build_vad_section(
            settings_win,
            use_vad_var,
            vad_threshold_var,
            vad_silence_duration_var,
            vad_pre_speech_padding_ms_var,
            vad_post_speech_padding_ms_var,
        )

        self._update_text_correction_fields()

    def show_status_tooltip(self, message: str) -> None:
        if not message:
            return
        if self.tray_icon:
            self.tray_icon.title = self._clamp_tray_tooltip(message)
            logging.debug("UIManager: tooltip updated to: %s", message)
        else:
            self._pending_tray_tooltip = message
            logging.debug("UIManager: pending tooltip queued: %s", message)

    def setup_tray_icon(self):
        # Logic moved from global, adjusted to use self.
        initial_state = self._get_core_state()
        try:
            initial_image = Image.open("icon.png")
        except FileNotFoundError:
            logging.warning("icon.png not found, using fallback image.")
            color1, color2 = self.ICON_COLORS.get(initial_state, self.DEFAULT_ICON_COLOR)
            initial_image = self.create_image(64, 64, color1, color2)
        initial_tooltip = self._pending_tray_tooltip or f"Whisper Recorder ({initial_state})"

        self.tray_icon = pystray.Icon(
            "whisper_recorder",
            initial_image,
            initial_tooltip,
            menu=pystray.Menu(lambda: self.create_dynamic_menu())
        )
        if self._pending_tray_tooltip:
            self.tray_icon.title = self._clamp_tray_tooltip(self._pending_tray_tooltip)
        # Set update callback in core_instance
        self.core_instance_ref.set_state_update_callback(self.update_tray_icon)
        self.core_instance_ref.set_segment_callback(self.update_live_transcription_threadsafe) # Connect segment callback

        # pystray's run() blocks the main thread. Since the application uses
        # Tkinter's mainloop, run the tray icon in detached mode so both loops
        # can coexist without relying on an extra thread that might terminate
        # prematurely.
        self.tray_icon.run_detached()
    
    def _close_settings_window(self):
        """Closes the settings window and resets the flag."""
        with self.settings_window_lock:
            if self.settings_window_instance:
                self.settings_window_instance.destroy()
                self.settings_window_instance = None
            self.settings_thread_running = False
            self._clear_settings_context()

    def create_dynamic_menu(self):
        # Logic moved from global, adjusted to use self.core_instance_ref
        # ...
        current_state = self._get_core_state()
        is_recording = current_state == "RECORDING"

        menu_items = [
            pystray.MenuItem(
                '\u23f9\ufe0f Parar Gravação' if is_recording else '\u25b6\ufe0f Iniciar Gravação',
                lambda icon, item: self.core_instance_ref.toggle_recording(),
                default=True,
                enabled=lambda item: self._get_core_state() in ['RECORDING', 'IDLE']
            ),
            pystray.MenuItem(
                '\U0001f4dd Correção de Texto',
                lambda icon, item: self.toggle_text_correction_from_tray(),
                checked=lambda item: bool(
                    self.config_manager.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY, False)
                ),
            ),
            pystray.MenuItem(
                '\u2699\ufe0f Configurações',
                lambda icon, item: self.main_tk_root.after(0, self.run_settings_gui),
                enabled=lambda item: self._get_core_state() not in ['LOADING_MODEL', 'RECORDING']
            ),
            pystray.MenuItem(
                '\U0001f9ed Assistente Inicial',
                lambda icon, item: self.main_tk_root.after(
                    0, lambda: self.core_instance_ref.launch_first_run_wizard(force=True)
                ),
                enabled=lambda item: self._get_core_state() not in ['LOADING_MODEL', 'RECORDING']
            ),
            pystray.MenuItem(
                'Modelo Gemini',
                pystray.Menu(
                    *[
                        pystray.MenuItem(
                            model,
                            lambda icon, item: self.core_instance_ref.apply_settings_from_external(new_gemini_model=item.text),
                            radio=True,
                            checked=lambda item: self.config_manager.get('gemini_model') == item.text
                        ) for model in self.config_manager.get(GEMINI_MODEL_OPTIONS_CONFIG_KEY, [])
                    ]
                )
            ),
            pystray.MenuItem(
                '\U0001f4e6 Tamanho do Lote',
                pystray.Menu(
                    pystray.MenuItem(
                        'Automático (VRAM)',
                        lambda icon, item: self.core_instance_ref.update_setting('batch_size_mode', 'auto'),
                        radio=True,
                        checked=lambda item: self.config_manager.get('batch_size_mode') == 'auto'
                    ),
                    pystray.MenuItem(
                        'Manual',
                        lambda icon, item: self.core_instance_ref.update_setting('batch_size_mode', 'manual'),
                        radio=True,
                        checked=lambda item: self.config_manager.get('batch_size_mode') == 'manual'
                    ),
                    pystray.MenuItem(
                        'Definir Batch Size Manual...',
                        lambda icon, item: self.main_tk_root.after(0, self._prompt_for_manual_batch_size)
                    )
                )
            ),
            pystray.MenuItem(
                '\U0001f4c4 Abrir Logs',
                lambda icon, item: self.main_tk_root.after(0, self.open_logs_directory)
            ),
            pystray.MenuItem(
                '\U0001f4da Abrir Documentação',
                lambda icon, item: self.main_tk_root.after(0, self.open_docs_directory)
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('\u274c Sair', self.on_exit_app)
        ]
        return tuple(menu_items)

    def on_exit_app(self, *_):
        # Logic moved from global, adjusted to use self.
        logging.info("Exit requested from tray icon.")
        if self.core_instance_ref:
            self.core_instance_ref.shutdown()
        if self.tray_icon:
            self.tray_icon.stop()
        self.main_tk_root.quit()

    def toggle_text_correction_from_tray(self):
        """Alterna a correção de texto diretamente pelo menu da bandeja."""
        current_value = bool(
            self.config_manager.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY, False)
        )
        new_value = not current_value

        core = getattr(self, "core_instance_ref", None)
        if core is None:
            logging.error("Tray menu requested text correction toggle, but AppCore reference is unavailable.")
            self.show_status_tooltip("Não foi possível acessar o núcleo do aplicativo.")
            return

        try:
            core.update_setting(
                TEXT_CORRECTION_ENABLED_CONFIG_KEY,
                new_value,
            )
        except Exception as exc:  # pragma: no cover - defensive logging only
            logging.error(
                "Failed to toggle text correction from tray menu: %s",
                exc,
                exc_info=True,
            )
            self.show_status_tooltip("Falha ao atualizar a correção de texto.")
            return

        status_message = (
            "Correção de texto ativada."
            if new_value
            else "Correção de texto desativada."
        )
        self.show_status_tooltip(status_message)

    def _prompt_for_manual_batch_size(self):
        """Prompts the user for a manual batch size and applies it."""
        current_manual_batch_size = self.config_manager.get("manual_batch_size", 8)
        new_batch_size_str = simpledialog.askstring(
            "Definir Batch Size Manual",
            f"Insira o novo Batch Size manual (atual: {current_manual_batch_size}):",
            parent=self.settings_window_instance # Usar a janela de configurações como pai se estiver aberta
        )
        if new_batch_size_str:
            try:
                new_batch_size = int(new_batch_size_str)
                if new_batch_size > 0:
                    self.core_instance_ref.update_setting(
                        'batch_size_mode', 'manual'
                    )
                    self.core_instance_ref.update_setting(
                        'manual_batch_size', new_batch_size
                    )
                    logging.info(f"Batch Size manual definido para: {new_batch_size}")
                else:
                    messagebox.showerror("Input Error", "Batch size must be a positive integer.", parent=self.settings_window_instance)
            except ValueError:
                messagebox.showerror("Input Error", "Invalid entry. Please provide an integer.", parent=self.settings_window_instance)



