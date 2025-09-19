import logging
import threading
import time
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto, unique
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
import numpy as np # Adicionado para np.ndarray no callback
import soundfile as sf
from tkinter import messagebox # Adicionado para messagebox no _on_model_load_failed

# Importar os novos módulos
from .config_manager import (
    ConfigManager,
    REREGISTER_INTERVAL_SECONDS,
    HOTKEY_HEALTH_CHECK_INTERVAL,
    DISPLAY_TRANSCRIPTS_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    ASR_BACKEND_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
    ASR_COMPUTE_DEVICE_CONFIG_KEY,
    ASR_DTYPE_CONFIG_KEY,
    ASR_CACHE_DIR_CONFIG_KEY,
)
from .audio_handler import AudioHandler, AUDIO_SAMPLE_RATE # AUDIO_SAMPLE_RATE ainda é usado em _handle_transcription_result
from .transcription_handler import TranscriptionHandler
from .keyboard_hotkey_manager import KeyboardHotkeyManager # Assumindo que está na raiz
from .gemini_api import GeminiAPI # Adicionado para correção de texto
from .model_manager import ensure_download, list_installed, DownloadCancelledError, get_model_download_size

# Estados da aplicação (movidos de global)
STATE_IDLE = "IDLE"
STATE_LOADING_MODEL = "LOADING_MODEL"
STATE_RECORDING = "RECORDING"
STATE_TRANSCRIBING = "TRANSCRIBING"
STATE_ERROR_MODEL = "ERROR_MODEL"
STATE_ERROR_AUDIO = "ERROR_AUDIO"
STATE_ERROR_TRANSCRIPTION = "ERROR_TRANSCRIPTION"
STATE_ERROR_SETTINGS = "ERROR_SETTINGS"


@unique
class StateEvent(Enum):
    """Eventos normalizados utilizados para transições de estado."""

    MODEL_MISSING = auto()
    MODEL_CACHE_INVALID = auto()
    MODEL_PROMPT_FAILED = auto()
    MODEL_DOWNLOAD_DECLINED = auto()
    MODEL_DOWNLOAD_STARTED = auto()
    MODEL_DOWNLOAD_CANCELLED = auto()
    MODEL_DOWNLOAD_INVALID_CACHE = auto()
    MODEL_DOWNLOAD_FAILED = auto()
    MODEL_CACHE_NOT_CONFIGURED = auto()
    MODEL_CACHE_MISSING = auto()
    MODEL_READY = auto()
    MODEL_LOADING_FAILED = auto()
    AUDIO_RECORDING_STARTED = auto()
    AUDIO_RECORDING_STOPPED = auto()
    AUDIO_RECORDING_DISCARDED = auto()
    AUDIO_ERROR = auto()
    TRANSCRIPTION_STARTED = auto()
    TRANSCRIPTION_COMPLETED = auto()
    AGENT_COMMAND_COMPLETED = auto()
    SETTINGS_MISSING_RECORD_KEY = auto()
    SETTINGS_HOTKEY_START_FAILED = auto()
    SETTINGS_REREGISTER_FAILED = auto()
    SETTINGS_RECOVERED = auto()


@dataclass(frozen=True)
class StateNotification:
    """Mensagem estruturada propagada para assinantes de mudanças de estado."""

    event: StateEvent
    state: str
    previous_state: str | None = None
    details: str | None = None
    source: str | None = None


STATE_FOR_EVENT: dict[StateEvent, str] = {
    StateEvent.MODEL_MISSING: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_INVALID: STATE_ERROR_MODEL,
    StateEvent.MODEL_PROMPT_FAILED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_DECLINED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_STARTED: STATE_LOADING_MODEL,
    StateEvent.MODEL_DOWNLOAD_CANCELLED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_INVALID_CACHE: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_FAILED: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_NOT_CONFIGURED: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_MISSING: STATE_ERROR_MODEL,
    StateEvent.MODEL_READY: STATE_IDLE,
    StateEvent.MODEL_LOADING_FAILED: STATE_ERROR_MODEL,
    StateEvent.AUDIO_RECORDING_STARTED: STATE_RECORDING,
    StateEvent.AUDIO_RECORDING_STOPPED: STATE_IDLE,
    StateEvent.AUDIO_RECORDING_DISCARDED: STATE_IDLE,
    StateEvent.AUDIO_ERROR: STATE_ERROR_AUDIO,
    StateEvent.TRANSCRIPTION_STARTED: STATE_TRANSCRIBING,
    StateEvent.TRANSCRIPTION_COMPLETED: STATE_IDLE,
    StateEvent.AGENT_COMMAND_COMPLETED: STATE_IDLE,
    StateEvent.SETTINGS_MISSING_RECORD_KEY: STATE_ERROR_SETTINGS,
    StateEvent.SETTINGS_HOTKEY_START_FAILED: STATE_ERROR_SETTINGS,
    StateEvent.SETTINGS_REREGISTER_FAILED: STATE_ERROR_SETTINGS,
    StateEvent.SETTINGS_RECOVERED: STATE_IDLE,
}


EVENT_DEFAULT_DETAILS: dict[StateEvent, str] = {
    StateEvent.MODEL_MISSING: "ASR model not found locally",
    StateEvent.MODEL_CACHE_INVALID: "Configured ASR cache directory is invalid",
    StateEvent.MODEL_PROMPT_FAILED: "Failed to prompt user about model download",
    StateEvent.MODEL_DOWNLOAD_DECLINED: "User declined automatic model download",
    StateEvent.MODEL_DOWNLOAD_STARTED: "Starting model download",
    StateEvent.MODEL_DOWNLOAD_CANCELLED: "Model download was cancelled",
    StateEvent.MODEL_DOWNLOAD_INVALID_CACHE: "Model download aborted due to invalid cache directory",
    StateEvent.MODEL_DOWNLOAD_FAILED: "Model download failed",
    StateEvent.MODEL_CACHE_NOT_CONFIGURED: "ASR cache directory not configured",
    StateEvent.MODEL_CACHE_MISSING: "ASR cache directory missing on disk",
    StateEvent.MODEL_READY: "Model loaded successfully",
    StateEvent.MODEL_LOADING_FAILED: "Model failed to load",
    StateEvent.AUDIO_RECORDING_STARTED: "AudioHandler started recording",
    StateEvent.AUDIO_RECORDING_STOPPED: "AudioHandler returned to idle",
    StateEvent.AUDIO_RECORDING_DISCARDED: "Recorded audio discarded before transcription",
    StateEvent.AUDIO_ERROR: "Audio subsystem reported an error",
    StateEvent.TRANSCRIPTION_STARTED: "Transcription pipeline started",
    StateEvent.TRANSCRIPTION_COMPLETED: "Transcription finished",
    StateEvent.AGENT_COMMAND_COMPLETED: "Agent command completed",
    StateEvent.SETTINGS_MISSING_RECORD_KEY: "Record hotkey is not configured",
    StateEvent.SETTINGS_HOTKEY_START_FAILED: "KeyboardHotkeyManager failed to start",
    StateEvent.SETTINGS_REREGISTER_FAILED: "Hotkey re-registration failed",
    StateEvent.SETTINGS_RECOVERED: "Recovered stable hotkey registration",
}


AUDIO_STATE_EVENT_MAP: dict[str, StateEvent] = {
    "RECORDING": StateEvent.AUDIO_RECORDING_STARTED,
    "TRANSCRIBING": StateEvent.TRANSCRIPTION_STARTED,
    "IDLE": StateEvent.AUDIO_RECORDING_STOPPED,
    "ERROR_AUDIO": StateEvent.AUDIO_ERROR,
}

StateUpdateCallback = Callable[[StateNotification], None]


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

        # --- Callbacks para UI (definidos externamente pelo UIManager) ---
        self.state_update_callback: StateUpdateCallback | None = None
        self.on_segment_transcribed = None # Callback para UI ao vivo

        # --- Módulos ---
        self.config_manager = ConfigManager()
        self._pending_tray_tooltips: list[str] = []

        # Sincronizar modelos ASR já presentes no disco no início da aplicação
        try:
            self._refresh_installed_models("__init__", raise_errors=True)
        except OSError:
            messagebox.showerror(
                "Configuração",
                "Diretório de cache inválido. Verifique as configurações.",
            )
        except Exception as e:
            logging.warning(
                "AppCore[__init__]: falha ao sincronizar modelos instalados: %r",
                e,
                exc_info=True,
            )

        self.audio_handler = AudioHandler(
            config_manager=self.config_manager,
            on_audio_segment_ready_callback=self._on_audio_segment_ready,
            on_recording_state_change_callback=self._handle_recording_state_change,
        )
        self.gemini_api = GeminiAPI(self.config_manager) # Instancia o GeminiAPI
        self.transcription_handler = TranscriptionHandler(
            config_manager=self.config_manager,
            gemini_api_client=self.gemini_api,  # Injeta a instância da API
            on_model_ready_callback=self._on_model_loaded,
            on_model_error_callback=self._on_model_load_failed,
            on_transcription_result_callback=self._handle_transcription_result,
            on_agent_result_callback=self._handle_agent_result_final, # Usa o novo callback
            on_segment_transcribed_callback=self._on_segment_transcribed_for_ui,
            is_state_transcribing_fn=self.is_state_transcribing,
        )
        self.transcription_handler.core_instance_ref = self  # Expõe referência do núcleo ao handler

        self.ui_manager = None # Será setado externamente pelo main.py
        # --- Estado da Aplicação ---
        self.current_state = STATE_LOADING_MODEL
        self._last_notification: StateNotification | None = None
        self.shutting_down = False
        self.full_transcription = "" # Acumula transcrição completa
        self.agent_mode_active = False # Adicionado para controle do modo agente
        self.key_detection_active = False # Flag para controle da detecção de tecla

        # --- Hotkey Manager ---
        self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
        self.ahk_running = False
        self.last_key_press_time = 0.0
        self.reregister_timer_thread = None
        self.stop_reregister_event = threading.Event()
        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()
        self.key_detection_callback = None # Callback para atualizar a UI com a tecla detectada

        # Carregar configurações iniciais
        self._apply_initial_config_to_core_attributes()

        try:
            cache_dir = self.config_manager.get("asr_cache_dir")
            model_id = self.asr_model_id
            backend = self.asr_backend
            ct2_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
            model_path = Path(cache_dir) / backend / model_id

            if not (model_path.is_dir() and any(model_path.iterdir())):
                logging.warning("ASR model not found locally; waiting for user confirmation before downloading.")
                self._set_state(
                    StateEvent.MODEL_MISSING,
                    details=f"Model '{model_id}' not present under {model_path}",
                    source="init",
                )
                self._prompt_model_install(model_id, backend, cache_dir, ct2_type)
            else:
                self.transcription_handler.start_model_loading()
        except OSError:
            messagebox.showerror("Erro", "Diretório de cache inválido.")
            self._set_state(
                StateEvent.MODEL_CACHE_INVALID,
                details=f"Invalid cache directory reported during init: {cache_dir}",
                source="init",
            )


        self._cleanup_old_audio_files_on_startup()
        atexit.register(self.shutdown)

    def _prompt_model_install(self, model_id, backend, cache_dir, ct2_type):
        """Agenda um prompt para download do modelo na thread principal."""
        decision_data = self.config_manager.get_last_model_prompt_decision()
        if (
            decision_data.get("model_id") == model_id
            and decision_data.get("backend") == backend
        ):
            decision = decision_data.get("decision")
            timestamp = decision_data.get("timestamp", 0.0)
            formatted = self._format_decision_timestamp(timestamp)
            if decision == "defer":
                logging.info(
                    "Skipping model installation prompt for %s/%s due to prior deferral at %s.",
                    backend,
                    model_id,
                    formatted,
                )
                return
            if decision == "accept":
                logging.info(
                    "Automatically starting model download for %s/%s based on acceptance recorded at %s.",
                    backend,
                    model_id,
                    formatted,
                )
                self.config_manager.record_model_prompt_decision("accept", model_id, backend)
                self._start_model_download(model_id, backend, cache_dir, ct2_type)
                return

        def _ask_user():
            try:
                try:
                    size_bytes, file_count = get_model_download_size(model_id)
                    size_gb = size_bytes / (1024 ** 3)
                    download_msg = f"Download of approximately {size_gb:.2f} GB ({file_count} files)."
                except Exception as size_error:
                    logging.debug(f"Could not fetch download size for {model_id}: {size_error}")
                    download_msg = "Download size unavailable."
                prompt_text = (
                    f"Model '{model_id}' is not installed.\n{download_msg}\nDownload now?"
                )
                if messagebox.askyesno("Model Download", prompt_text):
                    self.config_manager.record_model_prompt_decision("accept", model_id, backend)
                    self._start_model_download(model_id, backend, cache_dir, ct2_type)
                else:
                    self.config_manager.record_model_prompt_decision("defer", model_id, backend)
                    logging.info("User declined model download prompt.")
                    messagebox.showinfo(
                        "Model",
                        "No model installed. You can install one later in the settings.",
                    )
                    self._set_state(
                        StateEvent.MODEL_DOWNLOAD_DECLINED,
                        details=f"User declined download for '{model_id}'",
                        source="model_prompt",
                    )
            except Exception as prompt_error:
                logging.error(f"Failed to display model download prompt: {prompt_error}", exc_info=True)
                self._set_state(
                    StateEvent.MODEL_PROMPT_FAILED,
                    details=f"Prompt failure for '{model_id}': {prompt_error}",
                    source="model_prompt",
                )
        self.main_tk_root.after(0, _ask_user)

    @staticmethod
    def _format_decision_timestamp(ts: float | int) -> str:
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
        except Exception:
            return "unknown"

    def _start_model_download(self, model_id, backend, cache_dir, ct2_type):
        """Inicia o download do modelo em uma thread separada."""
        def _download():
            try:
                self._set_state(
                    StateEvent.MODEL_DOWNLOAD_STARTED,
                    details=f"Downloading '{model_id}' with backend '{backend}'",
                    source="model_download",
                )
                ensure_download(model_id, backend, cache_dir, quant=ct2_type)
            except DownloadCancelledError:
                logging.info("Model download cancelled by user.")
                self._set_state(
                    StateEvent.MODEL_DOWNLOAD_CANCELLED,
                    details=f"Download for '{model_id}' cancelled by user",
                    source="model_download",
                )
                self.main_tk_root.after(0, lambda: messagebox.showinfo("Model", "Download canceled."))
            except OSError:
                logging.error("Invalid cache directory during model download.", exc_info=True)
                self._set_state(
                    StateEvent.MODEL_DOWNLOAD_INVALID_CACHE,
                    details=f"Invalid cache directory '{cache_dir}' during download",
                    source="model_download",
                )
                self.main_tk_root.after(0, lambda: messagebox.showerror("Model", "Diretório de cache inválido. Verifique as configurações."))
            except Exception as e:
                logging.error(f"Model download failed: {e}", exc_info=True)
                self._set_state(
                    StateEvent.MODEL_DOWNLOAD_FAILED,
                    details=f"Download for '{model_id}' failed: {e}",
                    source="model_download",
                )
                self.main_tk_root.after(0, lambda: messagebox.showerror("Model", f"Download failed: {e}"))
            else:
                logging.info("Model download completed successfully.")

                def _after_download():
                    self._set_state(STATE_IDLE)
                    self.transcription_handler.start_model_loading()

                self.main_tk_root.after(0, _after_download)
        threading.Thread(target=_download, daemon=True, name="ModelDownloadThread").start()

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
        self.ct2_quantization = self.config_manager.get("ct2_quantization")
        # ... e outras configurações que AppCore precisa diretamente

    def _sync_installed_models(self):
        """Atualiza o ConfigManager com os modelos ASR instalados."""
        try:
            self._refresh_installed_models("_sync_installed_models", raise_errors=False)
        except Exception as e:  # pragma: no cover - salvaguarda
            logging.warning(
                "AppCore[_sync_installed_models]: falha ao sincronizar modelos instalados: %r",
                e,
                exc_info=True,
            )

    def _refresh_installed_models(self, context: str, *, raise_errors: bool) -> None:
        cache_dir_value = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)
        cache_dir_raw = str(cache_dir_value).strip() if cache_dir_value is not None else ""
        thread_name = threading.current_thread().name
        resolved_path = Path(cache_dir_raw).expanduser() if cache_dir_raw else None

        logging.debug(
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
            installed = list_installed(resolved_path)
        except Exception as exc:  # pragma: no cover - defensivo
            logging.warning(
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

        logging.info(
            "AppCore[%s]: list_installed retornou %d modelo(s) a partir de '%s' (thread='%s').",
            context,
            len(installed),
            resolved_path,
            thread_name,
        )
        self.config_manager.set_asr_installed_models(installed)

    def _handle_missing_cache_dir(self, cache_dir, context: str, reason: str) -> None:
        cache_repr = str(cache_dir) if cache_dir else "<não configurado>"
        logging.warning(
            "AppCore[%s]: diretório de cache de ASR indisponível (%s, motivo=%s).",
            context,
            cache_repr,
            reason,
        )
        if reason == "not_configured":
            event = StateEvent.MODEL_CACHE_NOT_CONFIGURED
            detail = "ASR cache directory not configured"
        elif reason == "missing":
            event = StateEvent.MODEL_CACHE_MISSING
            detail = f"ASR cache directory missing: {cache_repr}"
        else:
            event = StateEvent.MODEL_CACHE_INVALID
            detail = f"ASR cache directory issue ({reason}): {cache_repr}"
        self._set_state(event, details=detail, source=f"cache_dir::{context}")
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
                logging.debug("AppCore: tooltip pendente armazenada: %s", message)

    def flush_pending_ui_notifications(self) -> None:
        if not self._pending_tray_tooltips:
            return
        ui_manager = getattr(self, "ui_manager", None)
        tray_icon = getattr(ui_manager, "tray_icon", None) if ui_manager else None
        if not tray_icon or not hasattr(ui_manager, "show_status_tooltip"):
            logging.debug(
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
        """Registra um callback para receber ``StateNotification`` estruturado."""

        if callback is not None and not callable(callback):
            raise TypeError("state_update_callback must be callable or None")
        self.state_update_callback = callback

    def set_segment_callback(self, callback):
        self.on_segment_transcribed = callback

    def set_key_detection_callback(self, callback):
        """Define o callback para atualizar a UI com a tecla detectada."""
        self.key_detection_callback = callback

    def _handle_recording_state_change(self, audio_state: str):
        """Normaliza notificações do ``AudioHandler`` em eventos estruturados."""

        event = AUDIO_STATE_EVENT_MAP.get(audio_state)
        if event is None:
            logging.warning("AudioHandler emitted unknown state '%s'.", audio_state)
            return
        detail = f"AudioHandler signalled '{audio_state}'"
        self._set_state(event, details=detail, source="audio_handler")

    def _on_audio_segment_ready(self, audio_source: str | np.ndarray):
        """Callback chamado ao finalizar a gravação (arquivo ou array)."""
        self.temp_audio_file = audio_source if isinstance(audio_source, str) else None
        duration_seconds = 0.0
        try:
            if isinstance(audio_source, str):
                with sf.SoundFile(audio_source) as f:
                    duration_seconds = len(f) / f.samplerate
            else:
                duration_seconds = len(audio_source) / AUDIO_SAMPLE_RATE
        except Exception as e:
            logging.warning(f"Não foi possível obter duração do áudio: {e}")

        min_duration = self.config_manager.get('min_transcription_duration')
        
        if duration_seconds < min_duration:
            logging.info(
                f"Segmento de áudio ({duration_seconds:.2f}s) é mais curto que o mínimo configurado ({min_duration}s). Ignorando."
            )
            self._set_state(
                StateEvent.AUDIO_RECORDING_DISCARDED,
                details=f"Segment shorter than minimum ({duration_seconds:.2f}s < {min_duration}s)",
                source="audio_handler",
            )
            return  # Interrompe o processamento

        with self.agent_mode_lock:
            is_agent_mode = self.agent_mode_active
            if is_agent_mode:
                self.agent_mode_active = False

        logging.info(
            f"AppCore: Segmento de áudio pronto ({duration_seconds:.2f}s). Enviando para TranscriptionHandler (Modo Agente: {is_agent_mode})."
        )
        
        # Passa o estado capturado para o handler de transcrição.
        self.transcription_handler.transcribe_audio_segment(audio_source, is_agent_mode)

    def _on_model_loaded(self):
        """Callback do TranscriptionHandler quando o modelo é carregado com sucesso."""
        logging.info("AppCore: Model loaded successfully.")
        self._set_state(
            StateEvent.MODEL_READY,
            details="Transcription model loaded",
            source="transcription_handler",
        )
        self._start_autohotkey()

    def notify_model_loading_started(self):
        """Expõe atualização explícita de estado para carregamento de modelo."""
        self._set_state(STATE_LOADING_MODEL)
        
        # Iniciar serviços de estabilidade de hotkey se habilitados
        if self.hotkey_stability_service_enabled:
            # Iniciar thread de re-registro periódico
            if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                self.stop_reregister_event.clear()
                self.reregister_timer_thread = threading.Thread(
                    target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister"
                )
                self.reregister_timer_thread.start()
                logging.info("Periodic hotkey re-registration thread started.")
            
            # Iniciar thread de verificação de saúde
            if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                self.stop_health_check_event.clear()
                self.health_check_thread = threading.Thread(
                    target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread"
                )
                self.health_check_thread.start()
                logging.info("Hotkey health monitoring thread launched.")
        else:
            logging.info("Hotkey stability services are disabled by configuration.")

    def _on_model_load_failed(self, error_msg):
        """Callback do TranscriptionHandler quando o modelo falha ao carregar."""
        logging.error(f"AppCore: Falha ao carregar o modelo: {error_msg}")
        self._set_state(
            StateEvent.MODEL_LOADING_FAILED,
            details=error_msg,
            source="transcription_handler",
        )
        self._log_status(f"Erro: Falha ao carregar o modelo. {error_msg}", error=True)
        # Exibir messagebox via UI Manager se disponível
        if self.ui_manager:
            if error_msg == "Diretório de cache inválido.":
                self.main_tk_root.after(0, lambda: messagebox.showerror("Erro", "Diretório de cache inválido."))
            else:
                self.main_tk_root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Erro de Carregamento do Modelo",
                        f"Falha ao carregar o modelo Whisper:\n{error_msg}\n\nPor favor, verifique sua conexão com a internet, o nome do modelo nas configurações ou a memória da sua GPU.",
                    ),
                )

    def _on_segment_transcribed_for_ui(self, text):
        """Callback para enviar texto de segmento para a UI ao vivo."""
        if self.on_segment_transcribed:
            self.on_segment_transcribed(text)
        self.full_transcription += text + " " # Acumula a transcrição completa

    def _handle_transcription_result(self, corrected_text, raw_text):
        """Lida com o texto final de transcrição, priorizando a versão corrigida."""
        logging.info("AppCore: Handling final transcription result.")
        # O texto corrigido tem prioridade; se vazio, usa o acumulado durante a gravação
        text_to_display = corrected_text
        final_text = text_to_display.strip() if text_to_display else self.full_transcription.strip()

        if self.display_transcripts_in_terminal:
            print("\n=== COMPLETE TRANSCRIPTION ===\n" + final_text + "\n==============================\n")
        # Métricas de correção e paste
        try:
            # t_corr: medido indiretamente aqui usando logs de início/fim se disponíveis; como fallback, apenas marca etapa
            logging.info("[METRIC] stage=correction_done value_ms=0")
        except Exception:
            pass

        if pyperclip:
            try:
                pyperclip.copy(final_text)
                logging.info("Transcription copied to clipboard.")
            except Exception as e:
                logging.error(f"Erro ao copiar para o clipboard: {e}")
        
        t_clip_copy_start = time.perf_counter()
        if self.auto_paste:
            self._do_paste()
        else:
            self._log_status("Transcription complete. Auto-paste disabled.")
        t_clip_copy_end = time.perf_counter()
        try:
            logging.info(f"[METRIC] stage=clipboard_paste_block value_ms={(t_clip_copy_end - t_clip_copy_start) * 1000:.2f}")
        except Exception:
            pass
        
        char_count = len(final_text)
        self._set_state(
            StateEvent.TRANSCRIPTION_COMPLETED,
            details=f"Transcription finalized ({char_count} chars)",
            source="transcription",
        )
        if self.ui_manager:
            self.main_tk_root.after(0, self.ui_manager.close_live_transcription_window)
        logging.info(f"Corrected text ready for copy/paste: {final_text}")
        self.full_transcription = ""  # Reset para a próxima gravação
        self._delete_temp_audio_file()

    def _handle_agent_result_final(self, agent_response_text: str):
        """
        Lida com o resultado final do modo agente (copia, cola e reseta o estado).
        Esta função é chamada pelo TranscriptionHandler após a API Gemini ser consultada.
        """
        try:
            if not agent_response_text:
                logging.warning("Comando do agente retornou uma resposta vazia.")
                self._log_status("Comando do agente sem resposta.", error=True)
                return

            if pyperclip:
                pyperclip.copy(agent_response_text)
                logging.info("Agent response copied to clipboard.")

            if self.config_manager.get("agent_auto_paste", True): # Usa agent_auto_paste
                self._do_paste()
                self._log_status("Comando do agente executado e colado.")
            else:
                self._log_status("Comando do agente executado (colagem automática desativada).")

        except Exception as e:
            logging.error(f"Erro ao manusear o resultado do agente: {e}", exc_info=True)
            self._log_status(f"Erro ao manusear o resultado do agente: {e}", error=True)
        finally:
            response_size = len(agent_response_text)
            self._set_state(
                StateEvent.AGENT_COMMAND_COMPLETED,
                details=f"Agent response delivered ({response_size} chars)",
                source="agent_mode",
            )
            if self.ui_manager:
                self.main_tk_root.after(0, self.ui_manager.close_live_transcription_window)
            self._delete_temp_audio_file()

    def _do_paste(self):
        # Lógica movida de WhisperCore._do_paste
        try:
            pyautogui.hotkey('ctrl', 'v')
            logging.info("Text pasted.")
            self._log_status("Text pasted.")
        except Exception as e:
            logging.error(f"Erro ao colar: {e}")
            self._log_status("Erro ao colar.", error=True)

    def start_key_detection_thread(self):
        """Inicia uma thread para detectar uma única tecla e atualizar a UI."""
        if self.key_detection_active:
            logging.info("Key detection is already active.")
            return

        self.key_detection_active = True
        logging.info("Starting key detection...")
        
        def detect_key_task():
            try:
                # Temporariamente desativar hotkeys existentes para evitar conflitos
                self._cleanup_hotkeys()
                time.sleep(0.1) # Pequena pausa para garantir que os hooks sejam liberados

                detected_key = self.ahk_manager.detect_single_key()
                if detected_key:
                    logging.info(f"Tecla detectada: {detected_key}")
                    if self.key_detection_callback:
                        self.main_tk_root.after(0, lambda: self.key_detection_callback(detected_key.upper()))
                else:
                    logging.warning("No key detected or stop signal received.")
                    if self.key_detection_callback:
                        self.main_tk_root.after(0, lambda: self.key_detection_callback("N/A")) # Ou algum valor padrão
            except Exception as e:
                logging.error(f"Error during key detection: {e}", exc_info=True)
                if self.key_detection_callback:
                    self.main_tk_root.after(0, lambda: self.key_detection_callback("ERRO"))
            finally:
                self.key_detection_active = False
                # Re-registrar hotkeys após a detecção
                self.register_hotkeys()
                logging.info("Key detection finished. Hotkeys re-registered.")

        threading.Thread(target=detect_key_task, daemon=True, name="KeyDetectionThread").start()

    # --- Gerenciamento de Estado e Logs ---
    def _set_state(self, event: StateEvent, *, details: str | None = None, source: str | None = None):
        """Aplica uma transição de estado baseada em ``StateEvent``."""

        if not isinstance(event, StateEvent):
            raise ValueError(f"Unsupported state event payload: {event!r}")

        try:
            mapped_state = STATE_FOR_EVENT[event]
        except KeyError as exc:
            raise ValueError(f"No state mapping defined for event {event!r}") from exc

        message = details or EVENT_DEFAULT_DETAILS.get(event)

        with self.state_lock:
            previous_state = self.current_state
            last_event = self._last_notification.event if self._last_notification else None
            last_state = self._last_notification.state if self._last_notification else None
            if last_event == event and last_state == mapped_state:
                logging.debug(
                    "Duplicate state event %s suppressed (state=%s, source=%s).",
                    event.name,
                    mapped_state,
                    source,
                )
                return

            notification = StateNotification(
                event=event,
                state=mapped_state,
                previous_state=previous_state,
                details=message,
                source=source,
            )
            self.current_state = mapped_state
            self._last_notification = notification

        transition_log = f"State transition via {event.name}: {previous_state} -> {mapped_state}"
        if message:
            transition_log += f" ({message})"
        if source:
            transition_log += f" [source={source}]"
        logging.info(transition_log)

        callback = self.state_update_callback
        if not callback:
            return

        def _notify():
            try:
                callback(notification)
            except Exception as exc:  # pragma: no cover - proteção defensiva
                logging.error(
                    "State update callback failed for %s: %s",
                    event.name,
                    exc,
                    exc_info=True,
                )

        try:
            self.main_tk_root.after(0, _notify)
        except Exception:  # pragma: no cover - fallback quando Tk não estiver disponível
            logging.debug(
                "Tkinter scheduling failed for state event %s; calling callback directly.",
                event.name,
                exc_info=True,
            )
            _notify()

    def is_state_transcribing(self) -> bool:
        """Indica se o estado atual é TRANSCRIBING."""
        with self.state_lock:
            return self.current_state == STATE_TRANSCRIBING

    def _log_status(self, text, error=False):
        if error: logging.error(text)
        else: logging.info(text)

    # --- Hotkey Logic (movida de WhisperCore) ---
    def _start_autohotkey(self):
        with self.hotkey_lock:
            if self.ahk_running: return True
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
                self._set_state(
                    StateEvent.SETTINGS_HOTKEY_START_FAILED,
                    details="KeyboardHotkeyManager.start returned False",
                    source="hotkeys",
                )
                self._log_status("Erro: Falha ao iniciar KeyboardHotkeyManager.", error=True)
            return success

    def register_hotkeys(self):
        self._cleanup_hotkeys()
        time.sleep(0.2)
        if not self.record_key:
            self._set_state(
                StateEvent.SETTINGS_MISSING_RECORD_KEY,
                details="Record hotkey not configured",
                source="hotkeys",
            )
            self._log_status("Error: No record key set!", error=True)
            return False
        success = self._start_autohotkey()
        if success:
            self._log_status(f"Global hotkey registered: {self.record_key.upper()} (mode: {self.record_mode})")
            if self.current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
                self._set_state(
                    StateEvent.SETTINGS_RECOVERED,
                    details="Hotkeys registered successfully",
                    source="hotkeys",
                )
        else:
            self._set_state(
                StateEvent.SETTINGS_HOTKEY_START_FAILED,
                details="Hotkey registration failed",
                source="hotkeys",
            )
            self._log_status("Error: Hotkey registration failed.", error=True)
        return success

    def _cleanup_hotkeys(self):
        with self.keyboard_lock:
            try:
                if self.ahk_running:
                    if hasattr(self.ahk_manager, 'hotkey_handlers'):
                        self.ahk_manager.hotkey_handlers.clear()
                    self.ahk_manager.stop()
                    self.ahk_running = False
                    time.sleep(0.2)
            except Exception as e:
                logging.error(f"Error stopping KeyboardHotkeyManager: {e}")

    def _reload_keyboard_and_suppress(self):
        with self.keyboard_lock:
            max_attempts = 3; attempt = 0; last_error = None
            self._cleanup_hotkeys(); time.sleep(0.3)
            while attempt < max_attempts:
                attempt += 1
                try:
                    if self.ahk_running: self.ahk_manager.stop(); self.ahk_running = False; time.sleep(0.2)
                    self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
                    logging.info("KeyboardHotkeyManager reload completed successfully.")
                    break
                except Exception as e: last_error = e; logging.error(f"Erro na tentativa {attempt} de recarregamento: {e}"); time.sleep(1)
            if attempt >= max_attempts and last_error is not None:
                logging.error(f"Falha após {max_attempts} tentativas de recarregamento. Último erro: {last_error}")
                return False
            return self.register_hotkeys()

    def _periodic_reregister_task(self):
        while not self.stop_reregister_event.wait(REREGISTER_INTERVAL_SECONDS):
            with self.state_lock: current_state = self.current_state
            if current_state == STATE_IDLE: # Re-registrar hotkeys apenas quando ocioso
                logging.info(f"Periodic check: State is {current_state}. Attempting hotkey re-registration.")
                try:
                    success = self._reload_keyboard_and_suppress()
                    if success:
                        logging.info("Periodic hotkey re-registration attempt finished successfully.")
                        with self.state_lock:
                            should_emit = self.current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]
                        if should_emit:
                            self._set_state(
                                StateEvent.SETTINGS_RECOVERED,
                                details="Periodic hotkey re-registration succeeded",
                                source="hotkeys",
                            )
                    else:
                        logging.warning("Periodic hotkey re-registration attempt failed.")
                        self._set_state(
                            StateEvent.SETTINGS_REREGISTER_FAILED,
                            details="Periodic hotkey re-registration failed",
                            source="hotkeys",
                        )
                except Exception as e:
                    logging.error(f"Error during periodic hotkey re-registration: {e}", exc_info=True)
                    self._set_state(
                        StateEvent.SETTINGS_REREGISTER_FAILED,
                        details=f"Exception during periodic re-registration: {e}",
                        source="hotkeys",
                    )
            else:
                logging.debug(f"Periodic check: State is {current_state}. Skipping hotkey re-registration.")
        logging.info("Periodic hotkey re-registration thread stopped.")

    def force_reregister_hotkeys(self):
        with self.state_lock: current_state = self.current_state
        if current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
            logging.info(f"Manual trigger: State is {current_state}. Attempting hotkey re-registration.")
            with self.hotkey_lock:
                try:
                    if self.ahk_running: self.ahk_manager.stop(); self.ahk_running = False; time.sleep(0.5)
                    self.ahk_manager.update_config(record_key=self.record_key, agent_key=self.agent_key, record_mode=self.record_mode)
                    self.ahk_manager.set_callbacks(toggle=self.toggle_recording, start=self.start_recording, stop=self.stop_recording_if_needed, agent=self.start_agent_command)
                    success = self.ahk_manager.start()
                    if success:
                        self.ahk_running = True
                        if current_state.startswith("ERROR"):
                            self._set_state(
                                StateEvent.SETTINGS_RECOVERED,
                                details="Manual hotkey re-registration succeeded",
                                source="hotkeys",
                            )
                        self._log_status("KeyboardHotkeyManager reload completed.", error=False)
                        return True
                    else:
                        self._log_status("Falha ao recarregar KeyboardHotkeyManager.", error=True)
                        self._set_state(
                            StateEvent.SETTINGS_REREGISTER_FAILED,
                            details="Manual hotkey re-registration failed",
                            source="hotkeys",
                        )
                        return False
                except Exception as e:
                    self.ahk_running = False
                    logging.error(f"Exception during manual KeyboardHotkeyManager re-registration: {e}", exc_info=True)
                    self._log_status(f"Erro ao recarregar KeyboardHotkeyManager: {e}", error=True)
                    self._set_state(
                        StateEvent.SETTINGS_REREGISTER_FAILED,
                        details=f"Exception during manual hotkey re-registration: {e}",
                        source="hotkeys",
                    )
                    return False
        else:
            logging.warning(f"Manual trigger: Cannot re-register hotkeys. Current state is {current_state}.")
            self._log_status(f"Cannot reload now (State: {current_state}).", error=True)
            return False

    def _hotkey_health_check_task(self):
        while not self.stop_health_check_event.wait(HOTKEY_HEALTH_CHECK_INTERVAL):
            with self.state_lock: current_state = self.current_state
            if current_state == STATE_IDLE: # Only check/fix if IDLE
                if not self.ahk_running:
                    logging.warning("Hotkey health check: KeyboardHotkeyManager not running while IDLE. Attempting restart.")
                    self.force_reregister_hotkeys()
                    self._log_status("Attempting to restart KeyboardHotkeyManager.", error=False)
                else:
                    logging.debug("Hotkey health check: KeyboardHotkeyManager is running correctly while IDLE.")
            # Se o serviço de estabilidade estiver desativado, esta thread não deveria estar rodando.
            # Se estiver rodando, significa que o estado mudou ou houve um erro.
            # Não é necessário logar "Pulando verificação" se o serviço está desativado.
        logging.info("Hotkey health monitoring thread stopped.")

    # --- Recording Control (delegando para AudioHandler) ---
    def start_recording(self):
        with self.recording_lock:
            if self.audio_handler.is_recording:
                return
            with self.state_lock:
                if self.current_state == STATE_TRANSCRIBING:
                    self._log_status("Cannot record: Transcription running.", error=True)
                    return
                if self.transcription_handler.pipe is None or self.current_state == STATE_LOADING_MODEL:
                    self._log_status("Cannot record: Model not loaded.", error=True)
                    return
                if self.current_state.startswith("ERROR"):
                    self._log_status(
                        f"Cannot record: App in error state ({self.current_state}).",
                        error=True,
                    )
                    return
        
        # if self.ui_manager:
        #     self.ui_manager.show_live_transcription_window()
        self.audio_handler.start_recording()
        self.full_transcription = "" # Reset full transcription on new recording

    def stop_recording(self, agent_mode=False):
        with self.recording_lock:
            if not self.audio_handler.is_recording:
                return

        was_valid = self.audio_handler.stop_recording()
        if was_valid is False:
            # Se a gravação foi descartada por ser muito curta, garanta que
            # nenhum processo de transcrição fique pendente.
            if hasattr(self.transcription_handler, "stop_transcription"):
                self.transcription_handler.stop_transcription()
            self._set_state(
                StateEvent.AUDIO_RECORDING_DISCARDED,
                details="Recording discarded after stop",
                source="audio_handler",
            )

        # A janela de UI ao vivo será fechada pelo _handle_transcription_result

    def stop_recording_if_needed(self):
        with self.recording_lock:
            if not self.audio_handler.is_recording: return
        self.stop_recording()

    def toggle_recording(self):
        with self.recording_lock:
            rec = self.audio_handler.is_recording
        if rec:
            self.stop_recording()
            return
        with self.state_lock:
            if self.current_state == STATE_TRANSCRIBING:
                self._log_status("Cannot start recording, transcription in progress.", error=True)
                return
        self.start_recording()

    def start_agent_command(self):
        with self.recording_lock:
            if self.audio_handler.is_recording:
                if self.agent_mode_active:
                    self.stop_recording(agent_mode=True)
                    self.agent_mode_active = False
                return
        with self.state_lock:
            if self.current_state == STATE_TRANSCRIBING:
                self._log_status("Cannot start command: transcription in progress.", error=True)
                return
            if self.transcription_handler.pipe is None or self.current_state == STATE_LOADING_MODEL:
                self._log_status("Model not loaded.", error=True)
                return
            if self.current_state.startswith("ERROR"):
                self._log_status(f"Cannot start command: state {self.current_state}", error=True)
                return
        self.agent_mode_active = True
        self.start_recording()

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
            or self.current_state == STATE_LOADING_MODEL
        )

    # --- Settings Application Logic (delegando para ConfigManager e outros) ---
    def apply_settings_from_external(self, **kwargs):
        logging.info("AppCore: Applying new configuration from external source.")
        config_changed = False

        # Atualizar ConfigManager e verificar se houve mudanças
        launch_changed = False
        reload_required = False
        model_prompt_reset_required = False
        reload_keys = {
            ASR_BACKEND_CONFIG_KEY,
            ASR_MODEL_ID_CONFIG_KEY,
            ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            "asr_model",
        }
        for key, value in kwargs.items():
            # Mapear nomes de kwargs para chaves de config_manager se necessário
            config_key_map = {
                "new_key": "record_key", "new_mode": "record_mode", "new_auto_paste": "auto_paste",
                "new_sound_enabled": "sound_enabled", "new_sound_frequency": "sound_frequency",
                "new_sound_duration": "sound_duration", "new_sound_volume": "sound_volume",
                "new_agent_key": "agent_key", "new_text_correction_enabled": "text_correction_enabled",
                "new_text_correction_service": "text_correction_service",
                "new_openrouter_api_key": "openrouter_api_key", "new_openrouter_model": "openrouter_model",
                "new_gemini_api_key": "gemini_api_key", "new_gemini_model": "gemini_model",
                "new_agent_model": "gemini_agent_model",
                "new_gemini_prompt": GEMINI_PROMPT_CONFIG_KEY,
                "new_batch_size": "batch_size", "new_gpu_index": "gpu_index",
                "new_hotkey_stability_service_enabled": "hotkey_stability_service_enabled", # Nova configuração unificada
                "new_min_transcription_duration": "min_transcription_duration",
                "new_min_record_duration": "min_record_duration",
                "new_save_temp_recordings": SAVE_TEMP_RECORDINGS_CONFIG_KEY,
                "new_record_to_memory": "record_to_memory",
                "new_max_memory_seconds_mode": "max_memory_seconds_mode",
                "new_max_memory_seconds": "max_memory_seconds",
                "new_gemini_model_options": "gemini_model_options",
                "new_asr_backend": ASR_BACKEND_CONFIG_KEY,
                "new_asr_model": ASR_MODEL_ID_CONFIG_KEY,
                "new_asr_compute_device": ASR_COMPUTE_DEVICE_CONFIG_KEY,
                "new_asr_dtype": ASR_DTYPE_CONFIG_KEY,
                "new_asr_ct2_compute_type": ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
                "new_asr_cache_dir": ASR_CACHE_DIR_CONFIG_KEY,
                "new_use_vad": "use_vad",
                "new_vad_threshold": "vad_threshold",
                "new_vad_silence_duration": "vad_silence_duration",
                "new_display_transcripts_in_terminal": "display_transcripts_in_terminal",
                "new_record_storage_mode": "record_storage_mode",
                "new_record_storage_limit": "record_storage_limit",
                "new_launch_at_startup": "launch_at_startup",
                # Novas chaves para opções de chunk e recurso experimental
                "new_chunk_length_mode": "chunk_length_mode",
                "new_chunk_length_sec": "chunk_length_sec",
                "new_enable_torch_compile": "enable_torch_compile",
                "new_asr_model": "asr_model",
                "new_asr_backend": ASR_BACKEND_CONFIG_KEY,
                "new_asr_model_id": ASR_MODEL_ID_CONFIG_KEY,
                "new_ct2_quantization": ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            }
            mapped_key = config_key_map.get(key, key) # Usa o nome original se não mapeado

            current_value = self.config_manager.get(mapped_key)
            if current_value != value:
                self.config_manager.set(mapped_key, value)
                config_changed = True
                if mapped_key in reload_keys:
                    reload_required = True
                if mapped_key in {ASR_BACKEND_CONFIG_KEY, ASR_MODEL_ID_CONFIG_KEY, "asr_model"}:
                    model_prompt_reset_required = True
                if mapped_key == "launch_at_startup":
                    launch_changed = True
                logging.info(f"Configuração '{mapped_key}' alterada para: {value}")
        
        # Lógica para unificar auto_paste: se new_auto_paste foi passado, ele se aplica a ambos
        if "new_auto_paste" in kwargs:
            new_auto_paste_value = kwargs["new_auto_paste"]
            if self.config_manager.get("auto_paste") != new_auto_paste_value:
                self.config_manager.set("auto_paste", new_auto_paste_value)
                config_changed = True
                logging.info(f"Configuração 'auto_paste' alterada para: {new_auto_paste_value}")
            # Garantir que agent_auto_paste seja sempre igual a auto_paste
            if self.config_manager.get("agent_auto_paste") != new_auto_paste_value:
                self.config_manager.set("agent_auto_paste", new_auto_paste_value)
                config_changed = True
                logging.info(f"Configuração 'agent_auto_paste' (unificada) alterada para: {new_auto_paste_value}")
        
        if model_prompt_reset_required:
            self.config_manager.reset_last_model_prompt_decision()
            config_changed = True

        if config_changed:
            self.config_manager.save_config()
            self._apply_initial_config_to_core_attributes() # Re-aplicar configs ao AppCore

            self.audio_handler.config_manager = self.config_manager # Atualizar referência
            self.transcription_handler.config_manager = self.config_manager # Atualizar referência
            if any(
                key in kwargs
                for key in [
                    "new_use_vad",
                    "new_vad_threshold",
                    "new_vad_silence_duration",
                    "new_record_storage_mode",
                    "new_record_storage_limit",
                    "new_min_record_duration",
                ]
            ):
                self.audio_handler.update_config()
            self.transcription_handler.update_config() # Chamar para recarregar configs específicas do handler
            if reload_required:
                self.transcription_handler.start_model_loading()
            if launch_changed:
                from .utils.autostart import set_launch_at_startup
                set_launch_at_startup(self.config_manager.get("launch_at_startup"))
            # Re-inicializar clientes API existentes em vez de recriá-los
            self.gemini_api.reinitialize_client() # Re-inicializar cliente principal
            if self.transcription_handler.gemini_client:
                self.transcription_handler.gemini_client.reinitialize_client() # Re-inicializar cliente Gemini do TranscriptionHandler
            if self.transcription_handler.openrouter_client:
                self.transcription_handler.openrouter_client.reinitialize_client(
                    api_key=self.config_manager.get("openrouter_api_key"),
                    model_id=self.config_manager.get("openrouter_model")
                ) # Re-inicializar cliente OpenRouter do TranscriptionHandler
            
            # Re-registrar hotkeys se as chaves ou modo mudaram
            if kwargs.get("new_key") is not None or kwargs.get("new_mode") is not None or kwargs.get("new_agent_key") is not None:
                self.register_hotkeys()
            
            # Reiniciar/parar serviços de estabilidade de hotkey se a configuração mudou
            if kwargs.get("new_hotkey_stability_service_enabled") is not None:
                if kwargs["new_hotkey_stability_service_enabled"]:
                    # Iniciar thread de re-registro periódico
                    if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                        self.stop_reregister_event.clear()
                        self.reregister_timer_thread = threading.Thread(target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister")
                        self.reregister_timer_thread.start()
                        logging.info("Periodic hotkey re-registration thread launched via settings update.")
                    
                    # Iniciar thread de verificação de saúde
                    if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                        self.stop_health_check_event.clear()
                        self.health_check_thread = threading.Thread(target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread")
                        self.health_check_thread.start()
                        logging.info("Hotkey health monitoring thread launched via settings update.")
                else:
                    self.stop_reregister_event.set()
                    self.stop_health_check_event.set()
                    logging.info("Hotkey stability services stopped via settings update.")

            # Atualizar min_transcription_duration
            if kwargs.get('new_min_transcription_duration') is not None:
                if self.config_manager.get('min_transcription_duration') != kwargs['new_min_transcription_duration']:
                    self.config_manager.set('min_transcription_duration', kwargs['new_min_transcription_duration'])
                    logging.info(f"Configuração 'min_transcription_duration' alterada para: {kwargs['new_min_transcription_duration']}")

            if kwargs.get('new_min_record_duration') is not None:
                if self.config_manager.get('min_record_duration') != kwargs['new_min_record_duration']:
                    self.config_manager.set('min_record_duration', kwargs['new_min_record_duration'])
                    logging.info(f"Configuração 'min_record_duration' alterada para: {kwargs['new_min_record_duration']}")

            self._log_status("Configurações atualizadas.")
        else:
            logging.info("Nenhuma configuração alterada.")

    def update_setting(self, key: str, value):
        """
        Atualiza uma única configuração e propaga a mudança para os módulos relevantes.
        Usado para atualizações de configuração individuais, como do menu da bandeja.
        """
        old_value = self.config_manager.get(key)
        if old_value == value:
            logging.info(f"Configuração '{key}' já possui o valor '{value}'. Nenhuma alteração necessária.")
            return

        self.config_manager.set(key, value)
        if key in {ASR_BACKEND_CONFIG_KEY, ASR_MODEL_ID_CONFIG_KEY, "asr_model"}:
            self.config_manager.reset_last_model_prompt_decision()
        self.config_manager.save_config()
        logging.info(f"Configuração '{key}' alterada para: {value}")

        # Re-aplicar configurações aos atributos do AppCore
        self._apply_initial_config_to_core_attributes()

        if key == "launch_at_startup":
            from .utils.autostart import set_launch_at_startup
            set_launch_at_startup(bool(value))

        # Propagar para TranscriptionHandler se for uma configuração relevante
        if key in [
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
        ]:
            self.transcription_handler.config_manager = self.config_manager # Garantir que a referência esteja atualizada
            self.transcription_handler.update_config()
            logging.info(f"TranscriptionHandler: Configurações de transcrição atualizadas via update_setting para '{key}'.")

        if key in ["min_record_duration", "use_vad", "vad_threshold", "vad_silence_duration", "record_storage_mode", "record_storage_limit"]:
            self.audio_handler.config_manager = self.config_manager
            self.audio_handler.update_config()
            logging.info(f"AudioHandler: Configurações atualizadas via update_setting para '{key}'.")

        # Re-inicializar clientes API se a chave ou modelo mudou
        if key in ["gemini_api_key", "gemini_model", "gemini_agent_model", "openrouter_api_key", "openrouter_model"]:
            self.gemini_api.reinitialize_client()
            if self.transcription_handler.gemini_client:
                self.transcription_handler.gemini_client.reinitialize_client()
            if self.transcription_handler.openrouter_client:
                self.transcription_handler.openrouter_client.reinitialize_client(
                    api_key=self.config_manager.get("openrouter_api_key"),
                    model_id=self.config_manager.get("openrouter_model")
                )
            logging.info(f"Clientes API re-inicializados via update_setting para '{key}'.")

        # Re-registrar hotkeys se as chaves ou modo mudaram
        if key in ["record_key", "agent_key", "record_mode"]:
            self.register_hotkeys()
            logging.info(f"Hotkeys re-registradas via update_setting para '{key}'.")
        
        # Iniciar/parar serviços de estabilidade de hotkey se a configuração mudou
        if key == "hotkey_stability_service_enabled":
            if value:
                if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                    self.stop_reregister_event.clear()
                    self.reregister_timer_thread = threading.Thread(target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister")
                    self.reregister_timer_thread.start()
                    logging.info("Periodic hotkey re-registration thread launched via update_setting.")
                
                if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                    self.stop_health_check_event.clear()
                    self.health_check_thread = threading.Thread(target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread")
                    self.health_check_thread.start()
                    logging.info("Hotkey health monitoring thread launched via update_setting.")
            else:
                self.stop_reregister_event.set()
                self.stop_health_check_event.set()
                logging.info("Hotkey stability services stopped via update_setting.")
        
        logging.info(f"Configuração '{key}' atualizada e propagada com sucesso.")

    # --- Cleanup ---
    def _cleanup_old_audio_files_on_startup(self):
        # Lógica movida de WhisperCore._cleanup_old_audio_files_on_startup
        # ...
        import glob
        import os
        removed_count = 0
        logging.info("Running startup audio file cleanup...")
        try:
            files_to_check = glob.glob("temp_recording_*.wav") + glob.glob("recording_*.wav")
            for f in files_to_check:
                try:
                    os.remove(f)
                    logging.info(f"Deleted old audio file: {f}")
                    removed_count += 1
                except OSError as e:
                    logging.warning(f"Could not delete old audio file '{f}': {e}")
            if removed_count > 0:
                logging.info(f"Cleanup (startup): {removed_count} old audio file(s) removed.")
            else:
                logging.debug("Cleanup (startup): No old audio files found.")
        except Exception as e:
            logging.error(f"Error during startup audio file cleanup: {e}")

    def _delete_temp_audio_file(self):
        path = getattr(self.audio_handler, "temp_file_path", None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logging.info(f"Deleted temp audio file: {path}")
            except OSError as e:
                logging.warning(f"Could not delete temp audio file '{path}': {e}")
        self.audio_handler.temp_file_path = None

    def shutdown(self):
        if self.shutting_down: return
        self.shutting_down = True
        logging.info("Shutdown sequence initiated.")

        self.stop_reregister_event.set()
        self.stop_health_check_event.set()

        try:
            logging.info("Stopping KeyboardHotkeyManager...")
            self._cleanup_hotkeys()
        except Exception as e:
            logging.error(f"Error during hotkey cleanup in shutdown: {e}")

        if self.transcription_handler:
            try:
                self.transcription_handler.shutdown()
            except Exception as e:
                logging.error(f"Error shutting down TranscriptionHandler executor: {e}")

        if self.ui_manager and getattr(self.ui_manager, "tray_icon", None):
            try:
                self.ui_manager.tray_icon.stop()
            except Exception as e:
                logging.error(f"Erro ao encerrar tray icon: {e}")

        # Sinaliza para o AudioHandler parar a gravação e processamento
        if self.audio_handler.is_recording:
            logging.warning("Recording active during shutdown. Forcing stop...")
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
                        logging.info("Audio stream stopped and closed during shutdown.")
                except Exception as e:
                    logging.error(f"Error stopping audio stream on close: {e}")

        if hasattr(self.audio_handler, "cleanup"):
            try:
                self.audio_handler.cleanup()
            except Exception as e:
                logging.error(f"Erro no cleanup do AudioHandler: {e}")

        if self.reregister_timer_thread and self.reregister_timer_thread.is_alive():
            self.reregister_timer_thread.join(timeout=1.5)
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=1.5)

        logging.info("Core shutdown sequence complete.")
