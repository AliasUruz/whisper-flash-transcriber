import logging
import threading
import time
import os
import sys
from collections.abc import Callable
from typing import Iterable
from threading import RLock
from pathlib import Path
import atexit
try:
    import pyautogui  # Ainda necessário para _do_paste
except ImportError as exc:
    raise SystemExit(
        "Erro: a biblioteca 'pyautogui' não está instalada. "
        "Execute 'pip install -r requirements.txt' antes de executar o aplicativo."
    ) from exc
import pyperclip # Ainda necessário para _handle_transcription_result
from tkinter import messagebox # Adicionado para messagebox no _on_model_load_failed

# Importar os novos módulos
from . import state_manager as sm
from .config_manager import (
    ConfigManager,
    REREGISTER_INTERVAL_SECONDS,
    HOTKEY_HEALTH_CHECK_INTERVAL,
    DISPLAY_TRANSCRIPTS_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    MIN_RECORDING_DURATION_CONFIG_KEY,
    USE_VAD_CONFIG_KEY,
    VAD_THRESHOLD_CONFIG_KEY,
    VAD_SILENCE_DURATION_CONFIG_KEY,
    RECORD_STORAGE_MODE_CONFIG_KEY,
    RECORD_STORAGE_LIMIT_CONFIG_KEY,
    LAUNCH_AT_STARTUP_CONFIG_KEY,
    CLEAR_GPU_CACHE_CONFIG_KEY,
    ASR_BACKEND_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
    ASR_COMPUTE_DEVICE_CONFIG_KEY,
    ASR_DTYPE_CONFIG_KEY,
    ASR_CT2_CPU_THREADS_CONFIG_KEY,
    ASR_CACHE_DIR_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    OPENROUTER_TIMEOUT_CONFIG_KEY,
    VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
    VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
    AUTO_PASTE_MODIFIER_CONFIG_KEY,
)
from .audio_handler import AudioHandler
from .action_orchestrator import ActionOrchestrator
from .transcription_handler import TranscriptionHandler
from .keyboard_hotkey_manager import KeyboardHotkeyManager # Assumindo que está na raiz
from .gemini_api import GeminiAPI # Adicionado para correção de texto
from . import model_manager as model_manager_module


LOGGER = logging.getLogger('whisper_flash_transcriber.core')
MODEL_LOGGER = logging.getLogger('whisper_recorder.model')



StateUpdateCallback = Callable[[sm.StateNotification], None]


class AppCore:
    def __init__(self, main_tk_root):
        self.main_tk_root = main_tk_root # Referência para a raiz Tkinter

        # --- Locks ---
        self.hotkey_lock = RLock()
        self.recording_lock = RLock()
        self.transcription_lock = RLock()
        self.state_lock = RLock()
        self.keyboard_lock = RLock()
        self.agent_mode_lock = RLock() # Adicionado para o modo agente
        self.model_prompt_lock = RLock()
        self._key_detection_thread: threading.Thread | None = None
        self._key_detection_target: str | None = None
        self._key_detection_previous_value: str | None = None

        # --- Callbacks para UI (definidos externamente pelo UIManager) ---
        self.state_update_callback: StateUpdateCallback | None = None
        self.on_segment_transcribed = None # Callback para UI ao vivo

        # --- Módulos ---
        self.config_manager = ConfigManager()
        self.state_manager = sm.StateManager(sm.STATE_LOADING_MODEL, main_tk_root)
        self._ui_manager = None  # Será setado externamente pelo main.py
        self._pending_tray_tooltips: list[str] = []

        self.state_manager = sm.StateManager(sm.STATE_LOADING_MODEL, main_tk_root)

        self.full_transcription = ""

        self.action_orchestrator = ActionOrchestrator(
            state_manager=self.state_manager,
            config_manager=self.config_manager,
            clipboard_module=pyperclip,
            paste_callback=self._do_paste,
            log_status_callback=self._log_status,
            tk_root=self.main_tk_root,
            close_ui_callback=self._close_live_transcription_window,
            fallback_text_provider=lambda: self.full_transcription.strip(),
            reset_transcription_buffer=self._reset_full_transcription,
            delete_temp_audio_callback=self._delete_temp_audio_file,
        )

        self.model_manager = model_manager_module
        self._download_cancelled_error = getattr(
            self.model_manager,
            "DownloadCancelledError",
            Exception,
        )

        # Sincronizar modelos ASR já presentes no disco no início da aplicação
        try:
            self._refresh_installed_models("__init__", raise_errors=True)
        except OSError:
            messagebox.showerror(
                "Configuração",
                "Diretório de cache inválido. Verifique as configurações.",
            )
        except Exception as e:
            LOGGER.warning(
                "AppCore[__init__]: falha ao sincronizar modelos instalados: %r",
                e,
                exc_info=True,
            )

        self.audio_handler = AudioHandler(
            config_manager=self.config_manager,
            state_manager=self.state_manager,
            on_audio_segment_ready_callback=self.action_orchestrator.on_audio_segment_ready,
        )
        self.gemini_api = GeminiAPI(self.config_manager)
        self.transcription_handler = TranscriptionHandler(
            config_manager=self.config_manager,
            gemini_api_client=self.gemini_api,
            on_model_ready_callback=self._on_model_loaded,
            on_model_error_callback=self._on_model_load_failed,
            on_transcription_result_callback=self.action_orchestrator.handle_transcription_result,
            on_agent_result_callback=self.action_orchestrator.handle_agent_result,
            on_segment_transcribed_callback=self._on_segment_transcribed_for_ui,
            is_state_transcribing_fn=self.is_state_transcribing,
        )
        self.transcription_handler.core_instance_ref = self
        # Expõe referência do núcleo ao handler
        self.action_orchestrator.bind_transcription_handler(self.transcription_handler)

        self._ui_manager = None # Será setado externamente pelo main.py
        # --- Estado da Aplicação ---
        self.shutting_down = False
        self.full_transcription = "" # Acumula transcrição completa
        self.model_prompt_active = False

    @property
    def ui_manager(self):
        return self._ui_manager

    @ui_manager.setter
    def ui_manager(self, ui_manager_instance):
        self._ui_manager = ui_manager_instance
        if ui_manager_instance:
            self.state_manager.subscribe(ui_manager_instance.update_tray_icon)

        # --- Hotkey Manager ---
        self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
        self.ahk_running = False
        self.last_key_press_time = 0.0
        self.reregister_timer_thread = None
        self.stop_reregister_event = threading.Event()
        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()
        self.key_detection_callback = None # Callback para atualizar a UI com a tecla detectada
        self._key_detection_thread: threading.Thread | None = None

        # Carregar configurações iniciais
        self._apply_initial_config_to_core_attributes()

        self._active_model_download_event: threading.Event | None = None
        self.model_download_timeout = self._resolve_model_download_timeout()

        try:
            cache_dir = self.config_manager.get("asr_cache_dir")
            model_id = self.asr_model_id
            backend = self.asr_backend
            ct2_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)

            cache_root = Path(cache_dir)
            storage_backend = model_manager_module.backend_storage_name(backend)
            model_path = cache_root / storage_backend / model_id

            start_loading = True
            if not (model_path.is_dir() and any(model_path.iterdir())):
                legacy_path = cache_root / str(backend) / model_id if backend else None
                if legacy_path and legacy_path.is_dir() and any(legacy_path.iterdir()):
                    MODEL_LOGGER.info(
                        "Found legacy model directory at %s; using it for backend %s.",
                        legacy_path,
                        backend or storage_backend,
                    )
                else:
                    MODEL_LOGGER.warning("ASR model not found locally; waiting for user confirmation before downloading.")
                    self.state_manager.set_state(sm.STATE_ERROR_MODEL)
                    self._prompt_model_install(model_id, backend, cache_dir, ct2_type)
                    start_loading = False

            if start_loading:
                self._start_model_loading_with_synced_config()
        except OSError:
            messagebox.showerror("Erro", "Diretório de cache inválido.")
            self.state_manager.set_state(
                sm.StateEvent.MODEL_CACHE_INVALID,
                details=f"Invalid cache directory reported during init: {cache_dir}",
                source="init",
            )


        self._cleanup_old_audio_files_on_startup()
        atexit.register(self.shutdown)

    def _resolve_model_download_timeout(self) -> float | None:
        """Return configured timeout (seconds) for ASR downloads."""
        raw_timeout = self.config_manager.get("asr_download_timeout_seconds", 0)
        try:
            timeout_value = float(raw_timeout)
        except (TypeError, ValueError):
            LOGGER.debug("Invalid ASR download timeout %r. Ignoring.", raw_timeout)
            return None
        return timeout_value if timeout_value > 0 else None

    def _prompt_model_install(self, model_id, backend, cache_dir, ct2_type):
        """Agenda um prompt para download do modelo na thread principal."""

        def _ask_user():
            with self.model_prompt_lock:
                if self.model_prompt_active:
                    LOGGER.info(
                        "Model install prompt suppressed because another prompt is already active."
                    )
                    return

                last_decision = self.config_manager.get_last_model_prompt_decision()
                if (
                    last_decision.get("model_id") == model_id
                    and last_decision.get("decision") == "defer"
                    and (time.time() - last_decision.get("timestamp", 0)) < 86400
                ):
                    LOGGER.info(
                        "Model install prompt suppressed because user recently deferred it."
                    )
                    return

                self.model_prompt_active = True

            try:
                try:
                    size_bytes, file_count = self.model_manager.get_model_download_size(model_id)
                    size_gb = size_bytes / (1024 ** 3)
                    download_msg = f"Download of approximately {size_gb:.2f} GB ({file_count} files)."
                except Exception as size_error:
                    MODEL_LOGGER.debug(f"Could not fetch download size for {model_id}: {size_error}")
                    download_msg = "Download size unavailable."
                prompt_text = (
                    f"Model '{model_id}' is not installed.\n{download_msg}\nDownload now?"
                )
                if messagebox.askyesno("Model Download", prompt_text):
                    self.config_manager.record_model_prompt_decision("accept", model_id, backend)
                    cancel_event = threading.Event()
                    self._active_model_download_event = cancel_event
                    self._start_model_download(
                        model_id,
                        backend,
                        cache_dir,
                        ct2_type,
                        timeout=self.model_download_timeout,
                        cancel_event=cancel_event,
                    )
                else:
                    self.config_manager.record_model_prompt_decision("defer", model_id, backend)
                    MODEL_LOGGER.info("User declined model download prompt.")
                    messagebox.showinfo(
                        "Model",
                        "No model installed. You can install one later in the settings.",
                    )
                    self.state_manager.set_state(
                        sm.StateEvent.MODEL_DOWNLOAD_DECLINED,
                        details=f"User declined download for '{model_id}'",
                        source="model_prompt",
                    )
            except Exception as prompt_error:
                MODEL_LOGGER.error(f"Failed to display model download prompt: {prompt_error}", exc_info=True)
                self.state_manager.set_state(sm.STATE_ERROR_MODEL)
            finally:
                self.model_prompt_active = False

        self.main_tk_root.after(0, _ask_user)

    def _record_download_status(
        self,
        status: str,
        model_id: str,
        backend: str,
        *,
        message: str = "",
        details: str = "",
    ) -> None:
        """Persist structured information about download attempts."""

        config_manager = getattr(self, "config_manager", None)
        if not config_manager:
            return
        try:
            config_manager.record_model_download_status(
                status=status,
                model_id=model_id,
                backend=backend,
                message=message,
                details=details,
            )
        except Exception:  # pragma: no cover - persistence best effort
            LOGGER.debug(
                "Failed to persist download status for model %s (status=%s).",
                model_id,
                status,
                exc_info=True,
            )

    def download_model_and_reload(self, model_id, backend, cache_dir, quant):
        """
        Handles the full model download and subsequent reload process.
        This method is designed to be called from a background thread started by the UI.
        It raises exceptions back to the caller thread.
        """
        cancel_event = threading.Event()
        self._active_model_download_event = cancel_event
        
        try:
            self.state_manager.set_state(sm.STATE_LOADING_MODEL)
            self._record_download_status(
                "in_progress",
                model_id,
                backend,
                message="Model download started.",
            )

            ensure_kwargs = {
                "quant": quant,
                "timeout": self.model_download_timeout,
                "cancel_event": cancel_event,
            }

            result = self.model_manager.ensure_download(
                model_id,
                backend,
                cache_dir,
                **ensure_kwargs,
            )

            downloaded = bool(getattr(result, "downloaded", True))
            result_path = getattr(result, "path", "")
            status = "success" if downloaded else "skipped"
            message = (
                "Model download completed."
                if downloaded
                else "Model already present; download skipped."
            )
            self._record_download_status(
                status,
                model_id,
                backend,
                message=message,
                details=result_path,
            )

            # If download was not cancelled and did not raise an error, reload
            self._refresh_installed_models("post_download", raise_errors=False)
            self.transcription_handler.reload_asr()

        except self._download_cancelled_error as e:
            LOGGER.info(f"Download was cancelled for model {model_id}: {e}")
            self._record_download_status(
                "cancelled",
                model_id,
                backend,
                message=str(e) or "Model download cancelled.",
            )
            raise # Re-raise for the UI thread to handle
        except Exception as e:
            LOGGER.error(f"An error occurred during model download/reload for {model_id}: {e}", exc_info=True)
            self.state_manager.set_state(sm.STATE_ERROR_MODEL)
            self._record_download_status(
                "error",
                model_id,
                backend,
                message="Model download failed.",
                details=str(e),
            )
            raise # Re-raise for the UI thread to handle
        finally:
            self._active_model_download_event = None

    def _start_model_download(
        self,
        model_id,
        backend,
        cache_dir,
        ct2_type,
        *,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ):
        """Inicia o download do modelo em uma thread separada."""

        if cancel_event is not None:
            cancel_event.clear()

        def _download():
            try:
                self.state_manager.set_state(sm.STATE_LOADING_MODEL)
                self._record_download_status(
                    "in_progress",
                    model_id,
                    backend,
                    message="Model download started.",
                )
                ensure_kwargs = {
                    "quant": ct2_type,
                    "timeout": timeout,
                    "cancel_event": cancel_event,
                }
                result = self.model_manager.ensure_download(
                    model_id,
                    backend,
                    cache_dir,
                    **ensure_kwargs,
                )
                downloaded = bool(getattr(result, "downloaded", True))
                result_path = getattr(result, "path", "")
                status = "success" if downloaded else "skipped"
                message = (
                    "Model download completed."
                    if downloaded
                    else "Model already present; download skipped."
                )
                self._record_download_status(
                    status,
                    model_id,
                    backend,
                    message=message,
                    details=result_path,
                )
            except self._download_cancelled_error as cancel_exc:
                by_user = bool(getattr(cancel_exc, "by_user", False))
                timed_out = bool(getattr(cancel_exc, "timed_out", False))
                reason = str(cancel_exc).strip()
                if not reason:
                    if timed_out:
                        reason = "Model download timed out."
                    elif by_user:
                        reason = "Model download cancelled by user."
                    else:
                        reason = "Model download cancelled."
                self._record_download_status(
                    "cancelled",
                    model_id,
                    backend,
                    message=reason,
                )
                context_flags = []
                if by_user:
                    context_flags.append("by_user=True")
                if timed_out:
                    context_flags.append("timed_out=True")
                context_suffix = f" ({', '.join(context_flags)})" if context_flags else ""
                MODEL_LOGGER.info(
                    "Model download cancelled%s: backend=%s model_id=%s reason=%s",
                    context_suffix,
                    backend,
                    model_id,
                    reason,
                )
                self.state_manager.set_state(
                    sm.StateEvent.MODEL_DOWNLOAD_CANCELLED,
                    details=f"Download for '{model_id}' cancelled: {reason}",
                    source="model_download",
                )
                self.main_tk_root.after(
                    0,
                    lambda msg=reason: messagebox.showinfo("Model", msg),
                )
                return
            except OSError:
                MODEL_LOGGER.error("Invalid cache directory during model download.", exc_info=True)
                self._record_download_status(
                    "error",
                    model_id,
                    backend,
                    message="Invalid cache directory during model download.",
                )
                self.state_manager.set_state(sm.STATE_ERROR_MODEL)
                self.main_tk_root.after(0, lambda: messagebox.showerror("Model", "Diretório de cache inválido. Verifique as configurações."))
            except Exception as exc:
                MODEL_LOGGER.error(f"Model download failed: {exc}", exc_info=True)
                self._record_download_status(
                    "error",
                    model_id,
                    backend,
                    message="Model download failed.",
                    details=str(exc),
                )
                self.state_manager.set_state(sm.STATE_ERROR_MODEL)
                self.main_tk_root.after(0, lambda exc=exc: messagebox.showerror("Model", f"Download failed: {exc}"))  # noqa: F821
            else:
                MODEL_LOGGER.info("Model download completed successfully.")
                self.main_tk_root.after(0, self.transcription_handler.start_model_loading)
            finally:
                active_event = getattr(self, "_active_model_download_event", None)
                if active_event is cancel_event:
                    self._active_model_download_event = None
        threading.Thread(target=_download, daemon=True, name="ModelDownloadThread").start()

    def _start_model_loading_with_synced_config(self):
        """Start model loading after asserting the ConfigManager linkage.

        A sincronização explícita garante que ``TranscriptionHandler`` use a
        mesma instância de ``ConfigManager`` do núcleo antes de delegar o
        carregamento do modelo. Este passo deve ser verificado manualmente em
        cenários de recarga de modelo após alterações de configuração.
        """
        handler = getattr(self, "transcription_handler", None)
        if handler is None:
            LOGGER.error("Cannot start model loading: transcription handler missing.")
            self.state_manager.set_state(sm.STATE_ERROR_MODEL)
            return

        handler.config_manager = self.config_manager

        try:
            assert handler.config_manager is self.config_manager
        except AssertionError:
            LOGGER.error(
                "ConfigManager mismatch detected before model loading; aborting to avoid stale settings."
            )
            self.state_manager.set_state(sm.STATE_ERROR_SETTINGS)
            return

        LOGGER.debug(
            "ConfigManager synchronized before model loading (id=%s).",
            id(self.config_manager),
        )

        handler.start_model_loading()

    def cancel_model_download(self) -> None:
        """Solicita o cancelamento do download de modelo em andamento."""
        event = getattr(self, "_active_model_download_event", None)
        if event is not None:
            event.set()

    def _apply_initial_config_to_core_attributes(self):
        # Mover a atribuição de self.record_key, self.record_mode, etc.
        # para cá, usando self.config_manager.get()
        self.record_key = self.config_manager.get("record_key")
        self.record_mode = self.config_manager.get("record_mode")
        self.auto_paste = self.config_manager.get("auto_paste")
        self.agent_key = self.config_manager.get("agent_key")
        self.hotkey_stability_service_enabled = self.config_manager.get("hotkey_stability_service_enabled") # Nova configuração unificada
        self.keyboard_library = self.config_manager.get("keyboard_library")
        self.min_record_duration = self.config_manager.get("min_record_duration")
        self.display_transcripts_in_terminal = self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY)
        self.asr_backend = self.config_manager.get("asr_backend")
        self.asr_model_id = self.config_manager.get("asr_model_id")
        ct2_compute_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        self.asr_ct2_compute_type = ct2_compute_type
        self.ct2_quantization = ct2_compute_type
        # ... e outras configurações que AppCore precisa diretamente

    def _sync_installed_models(self):
        """Atualiza o ConfigManager com os modelos ASR instalados."""
        try:
            self._refresh_installed_models("_sync_installed_models", raise_errors=False)
        except Exception as e:  # pragma: no cover - salvaguarda
            LOGGER.warning(
                "AppCore[_sync_installed_models]: falha ao sincronizar modelos instalados: %r",
                e,
                exc_info=True,
            )

    def _refresh_installed_models(self, context: str, *, raise_errors: bool) -> None:
        cache_dir_value = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)
        cache_dir_raw = str(cache_dir_value).strip() if cache_dir_value is not None else ""
        thread_name = threading.current_thread().name
        resolved_path = Path(cache_dir_raw).expanduser() if cache_dir_raw else None

        LOGGER.debug(
            "AppCore[%s]: preparando list_installed (cache_dir='%s', resolvido='%s', thread='%s')",
            context,
            cache_dir_raw or "<não configurado>",
            resolved_path if resolved_path else "<sem caminho>",
            thread_name,
        )

        if not cache_dir_raw:
            self._handle_missing_cache_dir(None, context, "not_configured")
            self.config_manager.set_asr_installed_models([])
            return

        if resolved_path is None or not resolved_path.exists():
            self._handle_missing_cache_dir(resolved_path, context, "missing")
            self.config_manager.set_asr_installed_models([])
            return

        try:
            installed = self.model_manager.list_installed(resolved_path)
        except Exception as exc:  # pragma: no cover - defensivo
            LOGGER.warning(
                "AppCore[%s]: list_installed falhou em '%s': %r",
                context,
                resolved_path,
                exc,
                exc_info=True,
            )
            self.config_manager.set_asr_installed_models([])
            if raise_errors:
                raise
            return

        LOGGER.info(
            "AppCore[%s]: list_installed retornou %d modelo(s) a partir de '%s' (thread='%s').",
            context,
            len(installed),
            resolved_path,
            thread_name,
        )
        self.config_manager.set_asr_installed_models(installed)

    def _handle_missing_cache_dir(self, cache_dir, context: str, reason: str) -> None:
        cache_repr = str(cache_dir) if cache_dir else "<não configurado>"
        LOGGER.warning(
            "AppCore[%s]: diretório de cache de ASR indisponível (%s, motivo=%s).",
            context,
            cache_repr,
            reason,
        )
        if reason == "not_configured":
            event = sm.StateEvent.MODEL_CACHE_NOT_CONFIGURED
            detail = "ASR cache directory not configured"
        elif reason == "missing":
            event = sm.StateEvent.MODEL_CACHE_MISSING
            detail = f"ASR cache directory missing: {cache_repr}"
        else:
            event = sm.StateEvent.MODEL_CACHE_INVALID
            detail = f"ASR cache directory issue ({reason}): {cache_repr}"
        self.state_manager.set_state(event, details=detail, source=f"cache_dir::{context}")
        tooltip = (
            "Whisper Recorder - configure o diretório de modelos ASR"
            if reason == "not_configured"
            else f"Whisper Recorder - verifique o cache de modelos ASR ({cache_repr})"
        )
        self._queue_tooltip_update(tooltip)

    def _queue_tooltip_update(self, message: str) -> None:
        if not message:
            return
        ui_manager = getattr(self, "ui_manager", None)
        tray_icon = getattr(ui_manager, "tray_icon", None) if ui_manager else None
        if tray_icon and hasattr(ui_manager, "show_status_tooltip"):
            self.main_tk_root.after(0, lambda: ui_manager.show_status_tooltip(message))
        else:
            if message not in self._pending_tray_tooltips:
                self._pending_tray_tooltips.append(message)
                LOGGER.debug("AppCore: tooltip pendente armazenada: %s", message)

    def report_runtime_notice(self, message: str, *, level: int = logging.WARNING) -> None:
        """Publica um aviso em log e encaminha mensagem para a UI."""
        if not message:
            return
        LOGGER.log(level, message)
        self._queue_tooltip_update(message)

    def flush_pending_ui_notifications(self) -> None:
        if not self._pending_tray_tooltips:
            return
        ui_manager = getattr(self, "ui_manager", None)
        tray_icon = getattr(ui_manager, "tray_icon", None) if ui_manager else None
        if not tray_icon or not hasattr(ui_manager, "show_status_tooltip"):
            LOGGER.debug(
                "AppCore: UI ainda não está pronta; %d tooltip(s) continuam pendentes.",
                len(self._pending_tray_tooltips),
            )
            return

        pending = list(self._pending_tray_tooltips)
        self._pending_tray_tooltips.clear()
        for message in pending:
            self.main_tk_root.after(0, lambda msg=message: ui_manager.show_status_tooltip(msg))

    # --- Callbacks de Módulos ---
    def set_state_update_callback(self, callback: StateUpdateCallback | None) -> None:
        """Registra um callback para receber ``sm.StateNotification`` estruturado."""

        if callback is not None and not callable(callback):
            raise TypeError("state_update_callback must be callable or None")
        self.state_manager.subscribe(callback)

    def set_segment_callback(self, callback):
        self.on_segment_transcribed = callback

    def set_key_detection_callback(self, callback):
        """Define o callback para atualizar a UI com a tecla detectada."""
        self.key_detection_callback = callback

    def prepare_key_detection(self, target: str, *, current_value: str | None = None) -> None:
        """Configura o contexto utilizado pela captura assíncrona de hotkeys."""

        normalized = (target or "").strip().lower()
        if normalized in {"record", "record_key", "detected_key_var"}:
            normalized = "record"
        elif normalized in {"agent", "agent_key", "agent_key_var"}:
            normalized = "agent"
        else:
            LOGGER.debug(
                "Key detection requested for unknown target '%s'. Falling back to record hotkey.",
                target,
            )
            normalized = "record"

        with self.keyboard_lock:
            self._key_detection_target = normalized
            if current_value and current_value.strip() and current_value.strip().upper() != "PRESS KEY...":
                self._key_detection_previous_value = current_value.strip()
            else:
                self._key_detection_previous_value = None

    def _resolve_key_detection_fallback(self) -> str:
        """Resolve o valor usado quando nenhuma tecla é capturada."""

        with self.keyboard_lock:
            target = self._key_detection_target or "record"
            fallback = self._key_detection_previous_value

        if fallback:
            return fallback
        if target == "agent":
            return str(getattr(self, "agent_key", ""))
        return str(getattr(self, "record_key", ""))

    def _on_model_loaded(self):
        """Callback do TranscriptionHandler quando o modelo é carregado com sucesso."""
        LOGGER.info("AppCore: Model loaded successfully.")
        self.state_manager.set_state(
            sm.StateEvent.MODEL_READY,
            details="Transcription model loaded",
            source="transcription_handler",
        )
        self._start_autohotkey()

    def notify_model_loading_started(self):
        """Expõe atualização explícita de estado para carregamento de modelo."""
        self.state_manager.set_state(sm.STATE_LOADING_MODEL)
        
        # Iniciar serviços de estabilidade de hotkey se habilitados
        if self.hotkey_stability_service_enabled:
            # Iniciar thread de re-registro periódico
            if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                self.stop_reregister_event.clear()
                self.reregister_timer_thread = threading.Thread(
                    target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister"
                )
                self.reregister_timer_thread.start()
                LOGGER.info("Periodic hotkey re-registration thread started.")
            
            # Iniciar thread de verificação de saúde
            if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                self.stop_health_check_event.clear()
                self.health_check_thread = threading.Thread(
                    target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread"
                )
                self.health_check_thread.start()
                LOGGER.info("Hotkey health monitoring thread launched.")
        else:
            LOGGER.info("Hotkey stability services are disabled by configuration.")

    def _on_model_load_failed(self, error_msg):
        """Callback do TranscriptionHandler quando o modelo falha ao carregar."""
        LOGGER.error(f"AppCore: Falha ao carregar o modelo: {error_msg}")
        self.state_manager.set_state(
            sm.StateEvent.MODEL_LOADING_FAILED,
            details=error_msg,
            source="transcription_handler",
        )
        self._log_status(f"Erro: Falha ao carregar o modelo. {error_msg}", error=True)
        # Exibir messagebox via UI Manager se disponível
        if self.ui_manager:
            if error_msg == "Diretório de cache inválido.":
                self.main_tk_root.after(0, lambda: messagebox.showerror("Erro", "Diretório de cache inválido."))
            else:
                error_title = "Erro de Carregamento do Modelo"
                error_message = (
                    f"Falha ao carregar o modelo Whisper:\n{error_msg}\n\n"
                    "Por favor, verifique sua conexão com a internet, "
                    "o nome do modelo nas configurações ou a memória da sua GPU."
                )
                self.main_tk_root.after(
                    0,
                    lambda: messagebox.showerror(error_title, error_message),
                )

    def _on_segment_transcribed_for_ui(self, text):
        """Callback para enviar texto de segmento para a UI ao vivo."""
        if self.on_segment_transcribed:
            self.on_segment_transcribed(text)
        self.full_transcription += text + " " # Acumula a transcrição completa

    def _reset_full_transcription(self) -> None:
        self.full_transcription = ""

    def _close_live_transcription_window(self) -> None:
        ui_manager = getattr(self, "ui_manager", None)
        if ui_manager:
            try:
                ui_manager.close_live_transcription_window()
            except Exception:  # pragma: no cover - apenas log defensivo
                LOGGER.debug("Failed to close live transcription window.", exc_info=True)

    def _do_paste(self):
        # Lógica movida de WhisperCore._do_paste
        try:
            hotkey_sequence = self._resolve_paste_hotkey_sequence()
            pyautogui.hotkey(*hotkey_sequence)
            LOGGER.info("Text pasted.")
            self._log_status("Text pasted.")
        except Exception as e:
            LOGGER.error(f"Erro ao colar: {e}")
            self._log_status("Erro ao colar.", error=True)

    def _resolve_paste_hotkey_sequence(self) -> tuple[str, ...]:
        """Resolve the hotkey combination used for auto-paste."""

        raw_modifier = self.config_manager.get(AUTO_PASTE_MODIFIER_CONFIG_KEY, "auto")
        modifiers = self._normalize_paste_modifiers(raw_modifier)

        if not modifiers or modifiers == ("auto",):
            default_modifier = "command" if sys.platform == "darwin" else "ctrl"
            return (default_modifier, "v")

        if "v" in modifiers:
            return tuple(modifiers)

        return modifiers + ("v",)

    def _normalize_paste_modifiers(self, raw_value: object) -> tuple[str, ...]:
        """Normalize modifiers from configuration or defaults."""

        if raw_value is None:
            return ("auto",)

        mapping = {
            "ctrl": "ctrl",
            "control": "ctrl",
            "command": "command",
            "cmd": "command",
            "option": "alt",
            "alt": "alt",
            "win": "win",
            "windows": "win",
            "meta": "command" if sys.platform == "darwin" else "win",
            "super": "command" if sys.platform == "darwin" else "win",
        }

        if isinstance(raw_value, str):
            value = raw_value.strip()
            if not value:
                return ("auto",)
            segments = [segment for segment in value.replace("+", " ").split() if segment]
        elif isinstance(raw_value, (list, tuple, set)):
            segments = []
            for item in raw_value:
                if item is None:
                    continue
                segment = str(item).strip()
                if segment:
                    segments.extend(segment.replace("+", " ").split())
        else:
            segment = str(raw_value).strip()
            if not segment:
                return ("auto",)
            segments = segment.replace("+", " ").split()

        if not segments:
            return ("auto",)

        normalized: list[str] = []
        for segment in segments:
            lowered = segment.lower()
            mapped = mapping.get(lowered, lowered)
            if mapped:
                normalized.append(mapped)

        if not normalized:
            return ("auto",)

        return tuple(dict.fromkeys(normalized))

    def start_key_detection_thread(self, *, timeout: float | None = None) -> None:
        """Inicia detecção assíncrona de uma tecla pressionada para a UI."""

        manager = getattr(self, "ahk_manager", None)
        if manager is None:
            LOGGER.error(
                "AppCore: KeyboardHotkeyManager indisponível; não é possível detectar teclas."
            )
            return

        callback = getattr(self, "key_detection_callback", None)
        if callback is None:
            LOGGER.debug(
                "AppCore: detecção de tecla solicitada sem callback registrado; ignorando pedido."
            )
            return

        try:
            timeout_value = float(timeout) if timeout is not None else 5.0
        except (TypeError, ValueError):
            timeout_value = 5.0
            LOGGER.debug(
                "AppCore: timeout de detecção inválido (%s); utilizando padrão de %.1fs.",
                timeout,
                timeout_value,
            )

        def _deliver_result(result: str | None) -> None:
            if result is None:
                result = self._resolve_key_detection_fallback()
            else:
                result = str(result)

            try:
                callback(result)
            except Exception:  # pragma: no cover - UI callback defensivo
                LOGGER.error("Callback de detecção de tecla gerou exceção.", exc_info=True)

        def _schedule_delivery(result: str | None) -> None:
            def _invoke() -> None:
                _deliver_result(result)

            if self.main_tk_root:
                try:
                    self.main_tk_root.after(0, _invoke)
                except Exception:  # pragma: no cover - entrega síncrona como fallback
                    LOGGER.debug(
                        "Falha ao agendar callback de detecção no loop Tk; executando diretamente.",
                        exc_info=True,
                    )
                    _invoke()
            else:
                _invoke()

        def _detect_key() -> None:
            try:
                detected_key = manager.detect_key(timeout=timeout_value)
            except Exception as exc:
                LOGGER.error(
                    "AppCore: erro durante detecção de tecla: %s",
                    exc,
                    exc_info=True,
                )
                detected_key = None

            if not detected_key:
                LOGGER.info(
                    "Nenhuma tecla detectada dentro do timeout configurado; restaurando valor anterior."
                )

            _schedule_delivery(detected_key)

            with self.keyboard_lock:
                self._key_detection_thread = None
                self._key_detection_target = None
                self._key_detection_previous_value = None

        with self.keyboard_lock:
            if self._key_detection_thread and self._key_detection_thread.is_alive():
                LOGGER.debug(
                    "Thread de detecção de tecla já está em execução; ignorando novo pedido."
                )
                return

            self._key_detection_thread = threading.Thread(
                target=_detect_key,
                name="KeyDetectionThread",
                daemon=True,
            )
            self._key_detection_thread.start()

    def is_state_transcribing(self) -> bool:
        """Indica se o estado atual é TRANSCRIBING."""
        return self.state_manager.is_transcribing()

    def _log_status(self, text, error=False):
        if error:
            LOGGER.error(text)
        else:
            LOGGER.info(text)

    # --- Hotkey Logic (movida de WhisperCore) ---
    def _start_autohotkey(self):
        with self.hotkey_lock:
            if self.ahk_running:
                return True
            self.ahk_manager.update_config(
                record_key=self.record_key, agent_key=self.agent_key, record_mode=self.record_mode
            )
            self.ahk_manager.set_callbacks(
                toggle=self.toggle_recording, start=self.start_recording,
                stop=self.stop_recording_if_needed, agent=self.start_agent_command
            )
            success = self.ahk_manager.start()
            if success:
                self.ahk_running = True
                self._log_status(f"Hotkey registered: {self.record_key.upper()} (mode: {self.record_mode})")
            else:
                self.state_manager.set_state(
                    sm.StateEvent.SETTINGS_HOTKEY_START_FAILED,
                    details="KeyboardHotkeyManager.start returned False",
                    source="hotkeys",
                )
                self._log_status("Erro: Falha ao iniciar KeyboardHotkeyManager.", error=True)
            return success

    def register_hotkeys(self):
        self._cleanup_hotkeys()
        time.sleep(0.2)
        if not self.record_key:
            self.state_manager.set_state(
                sm.StateEvent.SETTINGS_MISSING_RECORD_KEY,
                details="Record hotkey not configured",
                source="hotkeys",
            )
            self._log_status("Error: No record key set!", error=True)
            return False
        success = self._start_autohotkey()
        if success:
            self._log_status(f"Global hotkey registered: {self.record_key.upper()} (mode: {self.record_mode})")
            if self.state_manager.get_current_state() not in [sm.STATE_RECORDING, sm.STATE_LOADING_MODEL]:
                self.state_manager.set_state(
                    sm.StateEvent.SETTINGS_RECOVERED,
                    details="Hotkeys registered successfully",
                    source="hotkeys",
                )
        else:
            self.state_manager.set_state(
                sm.StateEvent.SETTINGS_HOTKEY_START_FAILED,
                details="Hotkey registration failed",
                source="hotkeys",
            )
            self._log_status("Error: Hotkey registration failed.", error=True)
        return success

    def _cleanup_hotkeys(self):
        with self.keyboard_lock:
            try:
                if self.ahk_running:
                    self.ahk_manager.stop()
                    self.ahk_running = False
                    time.sleep(0.2)
            except Exception as e:
                LOGGER.error(f"Error stopping KeyboardHotkeyManager: {e}")

    def _reload_keyboard_and_suppress(self):
        with self.keyboard_lock:
            max_attempts = 3
            attempt = 0
            last_error = None
            self._cleanup_hotkeys()
            time.sleep(0.3)
            while attempt < max_attempts:
                attempt += 1
                try:
                    if self.ahk_running:
                        self.ahk_manager.stop()
                        self.ahk_running = False
                        time.sleep(0.2)
                    self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
                    LOGGER.info("KeyboardHotkeyManager reload completed successfully.")
                    break
                except Exception as e:
                    last_error = e
                    LOGGER.error(f"Erro na tentativa {attempt} de recarregamento: {e}")
                    time.sleep(1)
            if attempt >= max_attempts and last_error is not None:
                LOGGER.error(f"Falha após {max_attempts} tentativas de recarregamento. Último erro: {last_error}")
                return False
            return self.register_hotkeys()

    def _periodic_reregister_task(self):
        while not self.stop_reregister_event.wait(REREGISTER_INTERVAL_SECONDS):
            current_state = self.state_manager.get_current_state()
            if current_state == sm.STATE_IDLE: # Re-registrar hotkeys apenas quando ocioso
                LOGGER.info(f"Periodic check: State is {current_state}. Attempting hotkey re-registration.")
                try:
                    success = self._reload_keyboard_and_suppress()
                    if success:
                        LOGGER.info("Periodic hotkey re-registration attempt finished successfully.")
                        should_emit = self.state_manager.get_current_state() not in [sm.STATE_RECORDING, sm.STATE_LOADING_MODEL]
                        if should_emit:
                            self.state_manager.set_state(
                                sm.StateEvent.SETTINGS_RECOVERED,
                                details="Periodic hotkey re-registration succeeded",
                                source="hotkeys",
                            )
                    else:
                        LOGGER.warning("Periodic hotkey re-registration attempt failed.")
                        self.state_manager.set_state(
                            sm.StateEvent.SETTINGS_REREGISTER_FAILED,
                            details="Periodic hotkey re-registration failed",
                            source="hotkeys",
                        )
                except Exception as e:
                    LOGGER.error(f"Error during periodic hotkey re-registration: {e}", exc_info=True)
                    self.state_manager.set_state(
                        sm.StateEvent.SETTINGS_REREGISTER_FAILED,
                        details=f"Exception during periodic re-registration: {e}",
                        source="hotkeys",
                    )
            else:
                LOGGER.debug(f"Periodic check: State is {current_state}. Skipping hotkey re-registration.")
        LOGGER.info("Periodic hotkey re-registration thread stopped.")

    def force_reregister_hotkeys(self):
        current_state = self.state_manager.get_current_state()
        if current_state not in [sm.STATE_RECORDING, sm.STATE_LOADING_MODEL]:
            LOGGER.info(f"Manual trigger: State is {current_state}. Attempting hotkey re-registration.")
            with self.hotkey_lock:
                try:
                    if self.ahk_running:
                        self.ahk_manager.stop()
                        self.ahk_running = False
                        time.sleep(0.5)
                    self.ahk_manager.update_config(record_key=self.record_key, agent_key=self.agent_key, record_mode=self.record_mode)
                    self.ahk_manager.set_callbacks(toggle=self.toggle_recording, start=self.start_recording, stop=self.stop_recording_if_needed, agent=self.start_agent_command)
                    success = self.ahk_manager.start()
                    if success:
                        self.ahk_running = True
                        if current_state.startswith("ERROR"):
                            self.state_manager.set_state(
                                sm.StateEvent.SETTINGS_RECOVERED,
                                details="Manual hotkey re-registration succeeded",
                                source="hotkeys",
                            )
                        self._log_status("KeyboardHotkeyManager reload completed.", error=False)
                        return True
                    else:
                        self._log_status("Falha ao recarregar KeyboardHotkeyManager.", error=True)
                        self.state_manager.set_state(
                            sm.StateEvent.SETTINGS_REREGISTER_FAILED,
                            details="Manual hotkey re-registration failed",
                            source="hotkeys",
                        )
                        return False
                except Exception as e:
                    self.ahk_running = False
                    LOGGER.error(f"Exception during manual KeyboardHotkeyManager re-registration: {e}", exc_info=True)
                    self._log_status(f"Erro ao recarregar KeyboardHotkeyManager: {e}", error=True)
                    self.state_manager.set_state(
                        sm.StateEvent.SETTINGS_REREGISTER_FAILED,
                        details=f"Exception during manual hotkey re-registration: {e}",
                        source="hotkeys",
                    )
                    return False
        else:
            LOGGER.warning(f"Manual trigger: Cannot re-register hotkeys. Current state is {current_state}.")
            self._log_status(f"Cannot reload now (State: {current_state}).", error=True)
            return False

    def _hotkey_health_check_task(self):
        while not self.stop_health_check_event.wait(HOTKEY_HEALTH_CHECK_INTERVAL):
            current_state = self.state_manager.get_current_state()
            if current_state == sm.STATE_IDLE: # Only check/fix if IDLE
                if not self.ahk_running:
                    LOGGER.warning("Hotkey health check: KeyboardHotkeyManager not running while IDLE. Attempting restart.")
                    self.force_reregister_hotkeys()
                    self._log_status("Attempting to restart KeyboardHotkeyManager.", error=False)
                else:
                    LOGGER.debug("Hotkey health check: KeyboardHotkeyManager is running correctly while IDLE.")
            # Se o serviço de estabilidade estiver desativado, esta thread não deveria estar rodando.
            # Se estiver rodando, significa que o estado mudou ou houve um erro.
            # Não é necessário logar "Pulando verificação" se o serviço está desativado.
        LOGGER.info("Hotkey health monitoring thread stopped.")

    # --- Recording Control (delegando para AudioHandler) ---
    def start_recording(self):
        with self.recording_lock:
            if self.audio_handler.is_recording:
                return
            current_state = self.state_manager.get_current_state()
            if current_state == sm.STATE_TRANSCRIBING:
                self._log_status("Cannot record: Transcription running.", error=True)
                return
            if (not self.transcription_handler.is_model_ready()) or current_state == sm.STATE_LOADING_MODEL:
                self._log_status("Cannot record: Model not loaded.", error=True)
                return
            if current_state.startswith("ERROR"):
                self._log_status(
                    f"Cannot record: App in error state ({current_state}).",
                    error=True,
                )
                return
        
        # if self.ui_manager:
        #     self.ui_manager.show_live_transcription_window()
        self.audio_handler.start_recording()
        self._reset_full_transcription()
        self.action_orchestrator.deactivate_agent_mode()

    def stop_recording(self):
        with self.recording_lock:
            if not self.audio_handler.is_recording:
                return

        was_valid = self.audio_handler.stop_recording()
        if was_valid is False:
            if hasattr(self.transcription_handler, "stop_transcription"):
                self.transcription_handler.stop_transcription()
            self.state_manager.set_state(
                sm.StateEvent.AUDIO_RECORDING_DISCARDED,
                details="Recording discarded after stop",
                source="audio_handler",
            )
            self.action_orchestrator.deactivate_agent_mode()

    def stop_recording_if_needed(self):
        with self.recording_lock:
            if not self.audio_handler.is_recording:
                return
        self.stop_recording()

    def toggle_recording(self):
        with self.recording_lock:
            rec = self.audio_handler.is_recording
        if rec:
            self.stop_recording()
            return
        if self.state_manager.get_current_state() == sm.STATE_TRANSCRIBING:
            self._log_status("Cannot start recording, transcription in progress.", error=True)
            return
        self.start_recording()

    def start_agent_command(self):
        with self.recording_lock:
            if self.audio_handler.is_recording:
                if self.action_orchestrator.is_agent_mode_active:
                    self.stop_recording()
                    self.action_orchestrator.deactivate_agent_mode()
                return
        current_state = self.state_manager.get_current_state()
        if current_state == sm.STATE_TRANSCRIBING:
            self._log_status("Cannot start command: transcription in progress.", error=True)
            return
        if (not self.transcription_handler.is_model_ready()) or current_state == sm.STATE_LOADING_MODEL:
            self._log_status("Model not loaded.", error=True)
            return
        if current_state.startswith("ERROR"):
            self._log_status(f"Cannot start command: state {current_state}", error=True)
            return
        self.start_recording()
        self.action_orchestrator.activate_agent_mode()

    # --- Cancelamentos e consultas ---
    def is_transcription_running(self) -> bool:
        return self.transcription_handler.is_transcription_running()

    def is_correction_running(self) -> bool:
        return self.transcription_handler.is_text_correction_running()

    def is_any_operation_running(self) -> bool:
        """Indica se h\u00e1 alguma grava\u00e7\u00e3o, transcri\u00e7\u00e3o ou corre\u00e7\u00e3o em andamento."""
        return (
            self.audio_handler.is_recording
            or self.is_state_transcribing()
            or self.transcription_handler.is_text_correction_running()
            or self.state_manager.get_current_state() == sm.STATE_LOADING_MODEL
        )

    # --- Settings Application Logic (delegando para ConfigManager e outros) ---
    def apply_settings_from_external(
        self,
        *,
        force_reload: bool = False,
        forced_keys: Iterable[str] | None = None,
        **kwargs,
    ):
        LOGGER.info("AppCore: Applying new configuration from external source.")

        config_key_map = {
            "new_key": "record_key",
            "new_mode": "record_mode",
            "new_auto_paste": "auto_paste",
            "new_sound_enabled": "sound_enabled",
            "new_sound_frequency": "sound_frequency",
            "new_sound_duration": "sound_duration",
            "new_sound_volume": "sound_volume",
            "new_agent_key": "agent_key",
            "new_text_correction_enabled": "text_correction_enabled",
            "new_text_correction_service": "text_correction_service",
            "new_openrouter_api_key": "openrouter_api_key",
            "new_openrouter_model": "openrouter_model",
            "new_gemini_api_key": "gemini_api_key",
            "new_gemini_model": "gemini_model",
            "new_agent_model": "gemini_agent_model",
            "new_gemini_prompt": GEMINI_PROMPT_CONFIG_KEY,
            "new_batch_size": "batch_size",
            "new_gpu_index": "gpu_index",
            "new_hotkey_stability_service_enabled": "hotkey_stability_service_enabled",
            "new_min_transcription_duration": "min_transcription_duration",
            "new_min_record_duration": "min_record_duration",
            "new_save_temp_recordings": SAVE_TEMP_RECORDINGS_CONFIG_KEY,
            "new_max_memory_seconds_mode": "max_memory_seconds_mode",
            "new_max_memory_seconds": "max_memory_seconds",
            "new_gemini_model_options": "gemini_model_options",
            "new_asr_backend": ASR_BACKEND_CONFIG_KEY,
            "new_asr_model_id": ASR_MODEL_ID_CONFIG_KEY,
            "new_asr_compute_device": ASR_COMPUTE_DEVICE_CONFIG_KEY,
            "new_asr_dtype": ASR_DTYPE_CONFIG_KEY,
            "new_asr_ct2_compute_type": ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            "new_asr_cache_dir": ASR_CACHE_DIR_CONFIG_KEY,
            "new_ct2_cpu_threads": ASR_CT2_CPU_THREADS_CONFIG_KEY,
            "new_clear_gpu_cache": CLEAR_GPU_CACHE_CONFIG_KEY,
            "new_use_vad": USE_VAD_CONFIG_KEY,
            "new_vad_threshold": VAD_THRESHOLD_CONFIG_KEY,
            "new_vad_silence_duration": VAD_SILENCE_DURATION_CONFIG_KEY,
            "new_vad_pre_speech_padding_ms": VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
            "new_vad_post_speech_padding_ms": VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
            "new_display_transcripts_in_terminal": "display_transcripts_in_terminal",
            "new_record_storage_mode": RECORD_STORAGE_MODE_CONFIG_KEY,
            "new_record_storage_limit": RECORD_STORAGE_LIMIT_CONFIG_KEY,
            "new_launch_at_startup": LAUNCH_AT_STARTUP_CONFIG_KEY,
            "new_chunk_length_mode": "chunk_length_mode",
            "new_chunk_length_sec": "chunk_length_sec",
            "new_enable_torch_compile": "enable_torch_compile",
        }

        legacy_key_aliases = {
            "new_asr_model": ASR_MODEL_ID_CONFIG_KEY,
            "asr_model": ASR_MODEL_ID_CONFIG_KEY,
            "new_ct2_quantization": ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            "new_vad_pre_padding_ms": VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
            "new_vad_post_padding_ms": VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
        }

        normalized_updates: dict[str, object] = {}
        forced_key_set = set(forced_keys or [])
        sentinel = object()
        for raw_key, value in kwargs.items():
            if value is None:
                LOGGER.debug("Ignoring None value for configuration key '%s'.", raw_key)
                continue
            mapped_key = config_key_map.get(raw_key)
            if mapped_key is None:
                mapped_key = legacy_key_aliases.get(raw_key, raw_key)
            previous_value = normalized_updates.get(mapped_key, sentinel)
            if previous_value is not sentinel and previous_value != value:
                LOGGER.debug(
                    "Configuration key '%s' was provided multiple times; using latest value %r.",
                    mapped_key,
                    value,
                )
            normalized_updates[mapped_key] = value

        if not normalized_updates:
            if force_reload:
                LOGGER.info(
                    "AppCore: nenhum parâmetro recebido para reaplicar as configurações forçadas.")
            else:
                LOGGER.info("Nenhuma configuração alterada.")
            return

        changed_mapped_keys, warnings = self.config_manager.apply_updates(normalized_updates)
        if force_reload:
            if forced_key_set:
                changed_mapped_keys |= forced_key_set
            else:
                changed_mapped_keys |= set(normalized_updates.keys())
        if warnings:
            summary = "\n".join(f"- {message}" for message in warnings)
            message = (
                "Algumas configurações foram ajustadas automaticamente:\n\n"
                f"{summary}"
            )

            def _show_warning():
                messagebox.showwarning("Configurações ajustadas", message)

            self.main_tk_root.after(0, _show_warning)
        if not changed_mapped_keys:
            if force_reload:
                LOGGER.info(
                    "AppCore: nenhuma configuração alterada, mas forçando reaplicação dos parâmetros.")
            else:
                LOGGER.info("Nenhuma configuração alterada.")
                return

        if force_reload and changed_mapped_keys:
            LOGGER.debug(
                "AppCore: reaplicando configurações para as chaves: %s",
                ", ".join(sorted(changed_mapped_keys)),
            )

        reload_keys = {
            ASR_BACKEND_CONFIG_KEY,
            ASR_MODEL_ID_CONFIG_KEY,
            ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            ASR_COMPUTE_DEVICE_CONFIG_KEY,
            ASR_DTYPE_CONFIG_KEY,
            ASR_CT2_CPU_THREADS_CONFIG_KEY,
            ASR_CACHE_DIR_CONFIG_KEY,
        }
        reload_required = bool(changed_mapped_keys & reload_keys)
        launch_changed = LAUNCH_AT_STARTUP_CONFIG_KEY in changed_mapped_keys

        self._apply_initial_config_to_core_attributes()

        self.audio_handler.config_manager = self.config_manager
        self.transcription_handler.config_manager = self.config_manager

        audio_related_keys = {
            USE_VAD_CONFIG_KEY,
            VAD_THRESHOLD_CONFIG_KEY,
            VAD_SILENCE_DURATION_CONFIG_KEY,
            VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
            VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
            RECORD_STORAGE_MODE_CONFIG_KEY,
            RECORD_STORAGE_LIMIT_CONFIG_KEY,
            MIN_RECORDING_DURATION_CONFIG_KEY,
        }
        if audio_related_keys & changed_mapped_keys:
            self.audio_handler.update_config()

        try:
            reload_needed = self.transcription_handler.update_config(trigger_reload=False)
        except Exception as exc:
            LOGGER.error("Erro ao atualizar configurações do TranscriptionHandler: %s", exc, exc_info=True)
            self.state_manager.set_state(sm.STATE_ERROR_MODEL)
            self._log_status("Erro: Falha ao aplicar configurações do modelo.", error=True)
            return

        reload_required = reload_required or reload_needed

        if reload_required:
            self.state_manager.set_state(sm.STATE_LOADING_MODEL)
            try:
                self.transcription_handler.start_model_loading()
            except Exception as exc:
                LOGGER.error("Falha ao iniciar recarregamento do modelo ASR: %s", exc, exc_info=True)
                self.state_manager.set_state(sm.STATE_ERROR_MODEL)
                self._log_status("Erro: Falha ao iniciar recarregamento do modelo.", error=True)
                return
        else:
            self.state_manager.set_state(
                sm.StateEvent.SETTINGS_RECOVERED,
                details="Configurações aplicadas; mantendo estado IDLE.",
                source="settings",
            )

        if launch_changed:
            from .utils.autostart import set_launch_at_startup

            set_launch_at_startup(self.config_manager.get(LAUNCH_AT_STARTUP_CONFIG_KEY))

        self.gemini_api.reinitialize_client()
        if self.transcription_handler.gemini_client:
            self.transcription_handler.gemini_client.reinitialize_client()
        if self.transcription_handler.openrouter_client:
            openrouter_timeout = self.config_manager.get_timeout(
                OPENROUTER_TIMEOUT_CONFIG_KEY,
                self.transcription_handler.openrouter_client.request_timeout,
            )
            self.transcription_handler.openrouter_client.reinitialize_client(
                api_key=self.config_manager.get("openrouter_api_key"),
                model_id=self.config_manager.get("openrouter_model"),
                request_timeout=openrouter_timeout,
            )

        hotkey_related_keys = {"record_key", "record_mode", "agent_key"}
        if hotkey_related_keys & changed_mapped_keys:
            self.register_hotkeys()

        if "hotkey_stability_service_enabled" in changed_mapped_keys:
            if self.config_manager.get("hotkey_stability_service_enabled"):
                if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                    self.stop_reregister_event.clear()
                    self.reregister_timer_thread = threading.Thread(
                        target=self._periodic_reregister_task,
                        daemon=True,
                        name="PeriodicHotkeyReregister",
                    )
                    self.reregister_timer_thread.start()
                    LOGGER.info("Periodic hotkey re-registration thread launched via settings update.")

                if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                    self.stop_health_check_event.clear()
                    self.health_check_thread = threading.Thread(
                        target=self._hotkey_health_check_task,
                        daemon=True,
                        name="HotkeyHealthThread",
                    )
                    self.health_check_thread.start()
                    LOGGER.info("Hotkey health monitoring thread launched via settings update.")
            else:
                self.stop_reregister_event.set()
                self.stop_health_check_event.set()
                LOGGER.info("Hotkey stability services stopped via settings update.")

        text_correction_keys = {
            "openrouter_api_key",
            "openrouter_model",
            "gemini_api_key",
            "gemini_model",
            "gemini_agent_model",
            TEXT_CORRECTION_ENABLED_CONFIG_KEY,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY,
        }
        if text_correction_keys & changed_mapped_keys:
            self._refresh_text_correction_clients()

        self._log_status("Configurações atualizadas.")

    def update_setting(self, key: str, value):
        """
        Atualiza uma única configuração e propaga a mudança para os módulos relevantes.
        Usado para atualizações de configuração individuais, como do menu da bandeja.
        """
        old_value = self.config_manager.get(key)
        if old_value == value:
            LOGGER.info(f"Configuração '{key}' já possui o valor '{value}'. Nenhuma alteração necessária.")
            return

        self.config_manager.set(key, value)
        if key in {ASR_BACKEND_CONFIG_KEY, ASR_MODEL_ID_CONFIG_KEY, "asr_model"}:
            self.config_manager.reset_last_model_prompt_decision()
        self.config_manager.save_config()
        LOGGER.info(f"Configuração '{key}' alterada para: {value}")

        # Re-aplicar configurações aos atributos do AppCore
        self._apply_initial_config_to_core_attributes()

        if key == "launch_at_startup":
            from .utils.autostart import set_launch_at_startup
            set_launch_at_startup(bool(value))

        transcription_config_keys = {
            "batch_size_mode",
            "manual_batch_size",
            "gpu_index",
            "min_transcription_duration",
            "record_to_memory",
            "max_memory_seconds",
            "max_memory_seconds_mode",
            ASR_MODEL_ID_CONFIG_KEY,
            ASR_BACKEND_CONFIG_KEY,
            ASR_COMPUTE_DEVICE_CONFIG_KEY,
            ASR_DTYPE_CONFIG_KEY,
            ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            ASR_CACHE_DIR_CONFIG_KEY,
        }
        text_correction_keys = {
            "openrouter_api_key",
            "openrouter_model",
            "gemini_api_key",
            "gemini_model",
            "gemini_agent_model",
            TEXT_CORRECTION_ENABLED_CONFIG_KEY,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY,
        }

        # Propagar para TranscriptionHandler se for uma configuração relevante
        if key in transcription_config_keys or key in text_correction_keys:
            self.transcription_handler.config_manager = self.config_manager  # Garantir que a referência esteja atualizada
            reload_needed = self.transcription_handler.update_config(trigger_reload=False)
            LOGGER.info(f"TranscriptionHandler: Configurações de transcrição atualizadas via update_setting para '{key}'.")
            if reload_needed:
                self.state_manager.set_state(sm.STATE_LOADING_MODEL)
                try:
                    self.transcription_handler.start_model_loading()
                except Exception as exc:
                    LOGGER.error(
                        "Falha ao iniciar recarregamento do modelo após update_setting (%s): %s",
                        key,
                        exc,
                        exc_info=True,
                    )
                    self.state_manager.set_state(sm.STATE_ERROR_MODEL)
                    self._log_status("Erro: Falha ao iniciar recarregamento do modelo.", error=True)
                    return

        if key in ["min_record_duration", "use_vad", "vad_threshold", "vad_silence_duration", "record_storage_mode", "record_storage_limit"]:
            self.audio_handler.config_manager = self.config_manager
            self.audio_handler.update_config()
            LOGGER.info(f"AudioHandler: Configurações atualizadas via update_setting para '{key}'.")

        # Re-inicializar clientes de correção de texto quando necessário
        if key in text_correction_keys:
            self._refresh_text_correction_clients()

        # Re-registrar hotkeys se as chaves ou modo mudaram
        if key in ["record_key", "agent_key", "record_mode"]:
            self.register_hotkeys()
            LOGGER.info(f"Hotkeys re-registradas via update_setting para '{key}'.")
        
        # Iniciar/parar serviços de estabilidade de hotkey se a configuração mudou
        if key == "hotkey_stability_service_enabled":
            if value:
                if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                    self.stop_reregister_event.clear()
                    self.reregister_timer_thread = threading.Thread(target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister")
                    self.reregister_timer_thread.start()
                    LOGGER.info("Periodic hotkey re-registration thread launched via update_setting.")

                if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                    self.stop_health_check_event.clear()
                    self.health_check_thread = threading.Thread(target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread")
                    self.health_check_thread.start()
                    LOGGER.info("Hotkey health monitoring thread launched via update_setting.")
            else:
                self.stop_reregister_event.set()
                self.stop_health_check_event.set()
                LOGGER.info("Hotkey stability services stopped via update_setting.")

        LOGGER.info(f"Configuração '{key}' atualizada e propagada com sucesso.")

    def _refresh_text_correction_clients(self) -> None:
        """Reinicializa clientes de correção de texto respeitando a configuração atual."""
        try:
            if getattr(self, "gemini_api", None):
                self.gemini_api.reinitialize_client()
        except Exception as exc:  # pragma: no cover - falhas são registradas apenas
            LOGGER.error(
                "Falha ao reinicializar o cliente Gemini após alteração de configuração: %s",
                exc,
                exc_info=True,
            )

        openrouter_client = getattr(self.transcription_handler, "openrouter_client", None)
        if not openrouter_client:
            return

        try:
            openrouter_timeout = self.config_manager.get_timeout(
                OPENROUTER_TIMEOUT_CONFIG_KEY,
                getattr(openrouter_client, "request_timeout", None),
            )
            openrouter_client.reinitialize_client(
                api_key=self.config_manager.get("openrouter_api_key"),
                model_id=self.config_manager.get("openrouter_model"),
                request_timeout=openrouter_timeout,
            )
        except Exception as exc:  # pragma: no cover - falhas são registradas apenas
            LOGGER.error(
                "Falha ao reinicializar o cliente OpenRouter após alteração de configuração: %s",
                exc,
                exc_info=True,
            )

    # --- Cleanup ---
    def _cleanup_old_audio_files_on_startup(self):
        # Lógica movida de WhisperCore._cleanup_old_audio_files_on_startup
        # ...
        import glob
        import os
        removed_count = 0
        LOGGER.info("Running startup audio file cleanup...")
        try:
            files_to_check = glob.glob("temp_recording_*.wav") + glob.glob("recording_*.wav")
            for f in files_to_check:
                try:
                    os.remove(f)
                    LOGGER.info(f"Deleted old audio file: {f}")
                    removed_count += 1
                except OSError as e:
                    LOGGER.warning(f"Could not delete old audio file '{f}': {e}")
            if removed_count > 0:
                LOGGER.info(f"Cleanup (startup): {removed_count} old audio file(s) removed.")
            else:
                LOGGER.debug("Cleanup (startup): No old audio files found.")
        except Exception as e:
            LOGGER.error(f"Error during startup audio file cleanup: {e}")

    def _delete_temp_audio_file(self):
        path = getattr(self.audio_handler, "temp_file_path", None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
                LOGGER.info(f"Deleted temp audio file: {path}")
            except OSError as e:
                LOGGER.warning(f"Could not delete temp audio file '{path}': {e}")
        self.audio_handler.temp_file_path = None

    def shutdown(self):
        if self.shutting_down:
            return
        self.shutting_down = True
        LOGGER.info("Shutdown sequence initiated.")

        self.cancel_model_download()

        self.stop_reregister_event.set()
        self.stop_health_check_event.set()

        try:
            LOGGER.info("Stopping KeyboardHotkeyManager...")
            self._cleanup_hotkeys()
        except Exception as e:
            LOGGER.error(f"Error during hotkey cleanup in shutdown: {e}")

        if self.transcription_handler:
            try:
                self.transcription_handler.shutdown()
            except Exception as e:
                LOGGER.error(f"Error shutting down TranscriptionHandler executor: {e}")

        if self.ui_manager and getattr(self.ui_manager, "tray_icon", None):
            try:
                self.ui_manager.tray_icon.stop()
            except Exception as e:
                LOGGER.error(f"Erro ao encerrar tray icon: {e}")

        # Sinaliza para o AudioHandler parar a gravação e processamento
        if self.audio_handler.is_recording:
            LOGGER.warning("Recording active during shutdown. Forcing stop...")
            self.audio_handler.is_recording = False # Sinaliza para a thread de gravação parar
            # try:
            #     self.audio_handler.audio_queue.put_nowait(None) # Sinaliza para a thread de processamento parar
            # except queue.Full:
            #     pass
            if self.audio_handler.audio_stream:
                try:
                    if self.audio_handler.audio_stream.active:
                        self.audio_handler.audio_stream.stop()
                        self.audio_handler.audio_stream.close()
                        LOGGER.info("Audio stream stopped and closed during shutdown.")
                except Exception as e:
                    LOGGER.error(f"Error stopping audio stream on close: {e}")

        if hasattr(self.audio_handler, "cleanup"):
            try:
                self.audio_handler.cleanup()
            except Exception as e:
                LOGGER.error(f"Erro no cleanup do AudioHandler: {e}")

        if self.reregister_timer_thread and self.reregister_timer_thread.is_alive():
            self.reregister_timer_thread.join(timeout=1.5)
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=1.5)

        LOGGER.info("Core shutdown sequence complete.")
