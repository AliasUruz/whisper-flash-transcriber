
import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import simpledialog  # Adicionado para askstring
import logging
import threading
import time
import pystray
from PIL import Image, ImageDraw
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
    ASR_COMPUTE_DEVICE_CONFIG_KEY,
    ASR_BACKEND_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_DTYPE_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
    ASR_CACHE_DIR_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    DEFAULT_CONFIG,
)

from .utils.form_validation import safe_get_float, safe_get_int
from .utils.tooltip import Tooltip

# Importar get_available_devices_for_ui (pode ser movido para um utils ou ficar aqui)
# Por enquanto, vamos assumir que está disponível globalmente ou será movido para cá.
# Para este plano, vamos movê-lo para cá.
# import torch # Necessário para get_available_devices_for_ui - REMOVIDO

try:
    from .model_manager import DownloadCancelledError as _DefaultDownloadCancelledError
except Exception:  # pragma: no cover - fallback se a exceção não existir
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
            return ""

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
        if backend in ("faster-whisper", "ctranslate2"):
            backend = "ct2"
        if backend not in ("transformers", "ct2"):
            return None
        return backend

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
                quant_menu.configure(state="normal" if backend == "ct2" else "disabled")
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
            self.model_manager.ensure_download(
                model_id,
                backend,
                cache_dir,
                quant if backend == "ct2" else None,
            )
            installed_models = self.model_manager.list_installed(cache_dir)
            self.config_manager.set_asr_installed_models(installed_models)
            self.config_manager.save_config()
            self._update_model_info(model_id)
            messagebox.showinfo("Model", "Download completed.")
        except self._download_cancelled_error:
            messagebox.showinfo("Model", "Download canceled.")
        except OSError:
            messagebox.showerror(
                "Model",
                "Unable to write to the ASR cache directory. Please check permissions.",
            )
        except Exception as exc:
            messagebox.showerror("Model", f"Download failed: {exc}")

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
        record_storage_mode_var = _var("record_storage_mode_var")
        max_memory_seconds_mode_var = _var("max_memory_seconds_mode_var")
        max_memory_seconds_var = _var("max_memory_seconds_var")
        chunk_length_mode_var = _var("chunk_length_mode_var")
        chunk_length_sec_var = _var("chunk_length_sec_var")
        launch_at_startup_var = _var("launch_at_startup_var")
        asr_backend_var = _var("asr_backend_var")
        asr_model_id_var = _var("asr_model_id_var")
        asr_compute_device_var = _var("asr_compute_device_var")
        asr_dtype_var = _var("asr_dtype_var")
        asr_ct2_compute_type_var = _var("asr_ct2_compute_type_var")
        asr_cache_dir_var = _var("asr_cache_dir_var")

        if detected_key_var is None or mode_var is None or auto_paste_var is None:
            return

        record_key_value = detected_key_var.get()
        key_to_apply = record_key_value.lower() if record_key_value != "PRESS KEY..." else self.config_manager.get("record_key")
        agent_key_value = agent_key_var.get() if agent_key_var else self.config_manager.get("agent_key")
        agent_key_to_apply = agent_key_value.lower() if agent_key_value != "PRESS KEY..." else self.config_manager.get("agent_key")
        mode_to_apply = mode_var.get()
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

        vad_pre_speech_padding_ms_to_apply = self._safe_get_int(vad_pre_speech_padding_ms_var, "Pre-speech Padding", settings_win)
        if vad_pre_speech_padding_ms_to_apply is None:
            return
        vad_post_speech_padding_ms_to_apply = self._safe_get_int(vad_post_speech_padding_ms_var, "Post-speech Padding", settings_win)
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
        asr_dtype_to_apply = asr_dtype_var.get() if asr_dtype_var else self.config_manager.get_asr_dtype()
        asr_ct2_compute_type_to_apply = asr_ct2_compute_type_var.get() if asr_ct2_compute_type_var else self.config_manager.get_asr_ct2_compute_type()
        asr_cache_dir_to_apply = asr_cache_dir_var.get() if asr_cache_dir_var else self.config_manager.get_asr_cache_dir()

        try:
            Path(asr_cache_dir_to_apply).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Invalid Path", f"ASR cache directory is invalid:\n{exc}", parent=settings_win)
            return

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
            new_record_to_memory=(record_storage_mode_to_apply == "memory"),
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
            new_enable_torch_compile=bool(self._get_settings_var("enable_torch_compile_var").get()) if self._get_settings_var("enable_torch_compile_var") else False,
            new_asr_backend=asr_backend_to_apply,
            new_asr_model_id=asr_model_id_to_apply,
            new_asr_compute_device=asr_compute_device_to_apply,
            new_asr_dtype=asr_dtype_to_apply,
            new_asr_ct2_compute_type=asr_ct2_compute_type_to_apply,
            new_asr_cache_dir=asr_cache_dir_to_apply,
        )
        self._close_settings_window()

    def _restore_default_settings(self) -> None:
        defaults = DEFAULT_CONFIG
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

        self.core_instance_ref.apply_settings_from_external(**defaults)

        def _set_var(name: str, value) -> None:
            var = self._get_settings_var(name)
            if var is not None:
                try:
                    var.set(value)
                except Exception:
                    pass

        _set_var("auto_paste_var", defaults["auto_paste"])
        _set_var("mode_var", defaults["record_mode"])
        _set_var("detected_key_var", defaults["record_key"].upper())
        _set_var("agent_key_var", defaults["agent_key"].upper())
        _set_var("agent_model_var", defaults["gemini_agent_model"])
        _set_var("hotkey_stability_service_enabled_var", defaults["hotkey_stability_service_enabled"])
        _set_var("min_transcription_duration_var", defaults["min_transcription_duration"])
        _set_var("min_record_duration_var", defaults["min_record_duration"])
        _set_var("sound_enabled_var", defaults["sound_enabled"])
        _set_var("sound_frequency_var", str(defaults["sound_frequency"]))
        _set_var("sound_duration_var", str(defaults["sound_duration"]))
        _set_var("sound_volume_var", defaults["sound_volume"])
        _set_var("text_correction_enabled_var", defaults["text_correction_enabled"])
        _set_var("text_correction_service_var", defaults["text_correction_service"])
        _set_var("openrouter_api_key_var", defaults["openrouter_api_key"])
        _set_var("openrouter_model_var", defaults["openrouter_model"])
        _set_var("gemini_api_key_var", defaults["gemini_api_key"])
        _set_var("gemini_model_var", defaults["gemini_model"])
        _set_var("batch_size_var", str(defaults["batch_size"]))
        _set_var("asr_backend_var", defaults["asr_backend"])
        _set_var("asr_model_id_var", defaults["asr_model_id"])
        _set_var("asr_dtype_var", defaults["asr_dtype"])
        _set_var("asr_ct2_compute_type_var", defaults["asr_ct2_compute_type"])
        _set_var("asr_cache_dir_var", defaults["asr_cache_dir"])
        _set_var("use_vad_var", defaults["use_vad"])
        _set_var("vad_threshold_var", defaults["vad_threshold"])
        _set_var("vad_silence_duration_var", defaults["vad_silence_duration"])
        _set_var("vad_pre_speech_padding_ms_var", defaults["vad_pre_speech_padding_ms"])
        _set_var("vad_post_speech_padding_ms_var", defaults["vad_post_speech_padding_ms"])
        _set_var("save_temp_recordings_var", defaults[SAVE_TEMP_RECORDINGS_CONFIG_KEY])
        _set_var("display_transcripts_var", defaults["display_transcripts_in_terminal"])
        _set_var("record_storage_mode_var", defaults["record_storage_mode"])
        _set_var("max_memory_seconds_var", defaults["max_memory_seconds"])
        _set_var("max_memory_seconds_mode_var", defaults["max_memory_seconds_mode"])
        _set_var("launch_at_startup_var", defaults["launch_at_startup"])
        _set_var("chunk_length_mode_var", defaults.get("chunk_length_mode", "manual"))
        _set_var("chunk_length_sec_var", defaults["chunk_length_sec"])
        _set_var("enable_torch_compile_var", defaults.get("enable_torch_compile", False))

        gemini_prompt_textbox = self._get_settings_var("gemini_prompt_correction_textbox")
        if gemini_prompt_textbox is not None:
            gemini_prompt_textbox.delete("1.0", "end")
            gemini_prompt_textbox.insert("1.0", defaults["gemini_prompt"])
        agentico_prompt_textbox = self._get_settings_var("agentico_prompt_textbox")
        if agentico_prompt_textbox is not None:
            agentico_prompt_textbox.delete("1.0", "end")
            agentico_prompt_textbox.insert("1.0", defaults["prompt_agentico"])
        gemini_models_textbox = self._get_settings_var("gemini_models_textbox")
        if gemini_models_textbox is not None:
            gemini_models_textbox.delete("1.0", "end")
            gemini_models_textbox.insert("1.0", "\n".join(defaults["gemini_model_options"]))

        service_map = self._get_settings_meta("service_display_map", {})
        service_label = next(
            (label for label, val in service_map.items() if val == defaults["text_correction_service"]),
            "None",
        )
        _set_var("text_correction_service_label_var", service_label)
        service_menu = self._get_settings_var("service_menu")
        if service_menu is not None:
            try:
                service_menu.set(service_label)
            except Exception:
                pass
        self._on_service_menu_change(service_label)

        available_devices = self._get_settings_meta("available_devices", get_available_devices_for_ui())
        asr_compute_device_var = self._get_settings_var("asr_compute_device_var")
        if asr_compute_device_var is not None:
            selection = "Auto-select (Recommended)"
            if defaults["asr_compute_device"] == "cpu":
                selection = "Force CPU"
            elif defaults["asr_compute_device"] == "cuda" and defaults.get("gpu_index", -1) >= 0:
                selection = next(
                    (dev for dev in available_devices if dev.startswith(f"GPU {defaults['gpu_index']}")),
                    selection,
                )
            try:
                asr_compute_device_var.set(selection)
            except Exception:
                pass

        id_to_display = self._get_settings_meta("id_to_display", {})
        default_model_display = id_to_display.get(defaults["asr_model_id"], defaults["asr_model_id"])
        _set_var("asr_model_display_var", default_model_display)
        asr_model_menu = self._get_settings_var("asr_model_menu")
        if asr_model_menu is not None:
            try:
                asr_model_menu.set(default_model_display)
            except Exception:
                pass
        self._on_model_change(default_model_display)

        self.config_manager.save_config()

        self._update_text_correction_fields()
        self._update_chunk_length_state()
        model_var = self._get_settings_var("asr_model_id_var")
        if model_var is not None:
            self._update_model_info(model_var.get())
        self._update_install_button_state()

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

        _set_var("asr_cache_dir_var", DEFAULT_CONFIG["asr_cache_dir"])

        self.config_manager.set_asr_model_id(default_model_id)
        self.config_manager.set_asr_backend(DEFAULT_CONFIG["asr_backend"])
        self.config_manager.set_asr_ct2_compute_type(DEFAULT_CONFIG["asr_ct2_compute_type"])
        self.config_manager.set_asr_cache_dir(DEFAULT_CONFIG["asr_cache_dir"])
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
    def update_live_transcription_threadsafe(self, text):
        self.main_tk_root.after(0, lambda: self._update_live_transcription(text))

    def create_image(self, width, height, color1, color2=None):
        # Logic moved from global
        image = Image.new('RGB', (width, height), color1)
        if color2:
            dc = ImageDraw.Draw(image)
            dc.rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=color2)
        return image

    def _build_vad_section(self, parent, use_vad_var, vad_threshold_var, vad_silence_duration_var, vad_pre_speech_padding_ms_var, vad_post_speech_padding_ms_var):
        vad_frame = ctk.CTkFrame(parent)
        vad_frame.pack(fill="x", pady=5)
        ctk.CTkCheckBox(vad_frame, text="Enable Voice Activity Detection (VAD)", variable=use_vad_var).pack(side="left", padx=5)
        Tooltip(vad_frame, "Only record when speech is detected.")

        vad_options_frame = ctk.CTkFrame(parent)
        vad_options_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(vad_options_frame, text="VAD Threshold:").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_threshold_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Speech detection sensitivity.")

        ctk.CTkLabel(vad_options_frame, text="VAD Silence Duration (s):").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_silence_duration_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Duration of silence to trigger end of speech.")

        ctk.CTkLabel(vad_options_frame, text="Pre-speech Padding (ms):").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_pre_speech_padding_ms_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Milliseconds of audio to keep before speech is detected.")

        ctk.CTkLabel(vad_options_frame, text="Post-speech Padding (ms):").pack(side="left", padx=5)
        ctk.CTkEntry(vad_options_frame, textvariable=vad_post_speech_padding_ms_var, width=50).pack(side="left", padx=5)
        Tooltip(vad_options_frame, "Milliseconds of audio to keep after speech ends.")

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
        asr_dtype_var,
        asr_ct2_compute_type_var,
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
                button.configure(text='Ocultar avancado' if show else 'Mostrar avancado')

        def _toggle_advanced() -> None:
            _set_advanced_visibility(not advanced_state['visible'])

        advanced_toggle = ctk.CTkButton(
            asr_frame,
            text='Mostrar avancado',
            command=_toggle_advanced,
        )
        advanced_toggle.pack(fill='x', pady=(0, 5))
        toggle_button_ref['widget'] = advanced_toggle

        asr_backend_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(asr_backend_frame, fill="x", pady=5)
        ctk.CTkLabel(asr_backend_frame, text="ASR Backend:").pack(side="left", padx=(5, 0))
        ctk.CTkButton(
            asr_backend_frame,
            text="?",
            width=20,
            command=lambda: messagebox.showinfo(
                "ASR Backend",
                "Selects the inference engine used for speech recognition.",
            ),
        ).pack(side="left", padx=(0, 10))

        def _on_backend_change(choice: str) -> None:
            asr_backend_var.set(choice)
            _update_install_button_state()
            _update_model_info(asr_model_id_var.get())

        asr_backend_menu = ctk.CTkOptionMenu(
            asr_backend_frame,
            variable=asr_backend_var,
            values=["auto", "transformers", "ct2"],
            command=_on_backend_change,
        )
        asr_backend_menu.pack(side="left", padx=5)
        Tooltip(
            asr_backend_menu,
            "Inference backend for speech recognition.\nDerived from selected model; override in advanced mode.",
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
        catalog = model_manager.list_catalog()
        ui_elements["catalog"] = catalog
        catalog_display_map = {entry["id"]: entry.get("display_name", entry["id"]) for entry in catalog}
        try:
            installed_ids = {
                m["id"] for m in model_manager.list_installed(asr_cache_dir_var.get())
            }
        except OSError:
            messagebox.showerror(
                "Configuration",
                "Unable to access the model cache directory. Please review the path in Settings.",
            )
            installed_ids = set()
        all_ids = sorted({m["id"] for m in catalog} | installed_ids)
        id_to_display = {model_id: catalog_display_map.get(model_id, model_id) for model_id in all_ids}
        display_to_id = {display: model_id for model_id, display in id_to_display.items()}
        ui_elements["id_to_display"] = id_to_display
        ui_elements["display_to_id"] = display_to_id
        asr_model_display_var = ctk.StringVar(
            value=id_to_display.get(asr_model_id_var.get(), asr_model_id_var.get())
        )
        ui_elements["asr_model_display_var"] = asr_model_display_var
        asr_model_frame = ctk.CTkFrame(asr_frame)
        asr_model_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(asr_model_frame, text="ASR Model:").pack(side="left", padx=(5, 10))
        asr_model_menu = ctk.CTkOptionMenu(
            asr_model_frame,
            variable=asr_model_display_var,
            values=[id_to_display[mid] for mid in all_ids],
        )
        ui_elements["asr_model_menu"] = asr_model_menu

        model_size_label = ctk.CTkLabel(asr_model_frame, text="Download: calculating... | Installed: -")
        model_size_label.pack(side="left", padx=5)
        ui_elements["model_size_label"] = model_size_label

        def _derive_backend_from_model(model_ref: str) -> str | None:
            model_id = ui_elements["display_to_id"].get(model_ref, model_ref)
            entry = next((m for m in ui_elements["catalog"] if m["id"] == model_id), None)
            if not entry:
                installed = model_manager.list_installed(asr_cache_dir_var.get())
                entry = next((m for m in installed if m["id"] == model_id), None)
            backend = entry.get("backend") if entry else None
            if backend in ("faster-whisper", "ctranslate2"):
                backend = "ct2"
            if backend not in ("transformers", "ct2"):
                return None
            return backend

        def _update_model_info(model_ref: str) -> None:
            ui_elements["model_size_label"].configure(text="Download: calculating... | Installed: -")
            model_id = ui_elements["display_to_id"].get(model_ref, model_ref)
            try:
                d_bytes, d_files = model_manager.get_model_download_size(model_id)
                d_mb = d_bytes / (1024 * 1024)
                download_text = f"{d_mb:.1f} MB ({d_files} files)"
            except Exception:
                download_text = "?"

            try:
                installed_models = model_manager.list_installed(asr_cache_dir_var.get())
            except OSError:
                messagebox.showerror(
                    "Configuration",
                    "Unable to access the model cache directory. Please review the path in Settings.",
                )
                installed_models = []
            entry = next((m for m in installed_models if m["id"] == model_id), None)
            if entry:
                i_bytes, i_files = model_manager.get_installed_size(entry["path"])
                i_mb = i_bytes / (1024 * 1024)
                installed_text = f"{i_mb:.1f} MB ({i_files} files)"
            else:
                installed_text = "-"

            ui_elements["model_size_label"].configure(
                text=f"Download: {download_text} | Installed: {installed_text}"
            )

        def _update_install_button_state() -> None:
            backend = _derive_backend_from_model(asr_model_id_var.get())
            install_button.configure(state="normal" if backend else "disabled")
            ui_elements["quant_menu"].configure(state="normal" if backend == "ct2" else "disabled")

        def _on_model_change(choice_display: str) -> None:
            model_id = ui_elements["display_to_id"].get(choice_display, choice_display)
            asr_model_id_var.set(model_id)
            ui_elements["asr_model_display_var"].set(ui_elements["id_to_display"].get(model_id, model_id))
            backend = _derive_backend_from_model(model_id)
            if backend:
                asr_backend_var.set(backend)
                asr_backend_menu.configure(state="disabled")
            else:
                asr_backend_menu.configure(state="normal")
            self.config_manager.set_asr_model_id(model_id)
            self.config_manager.set_asr_backend(asr_backend_var.get())
            self.config_manager.save_config()
            _on_backend_change(asr_backend_var.get())
            _update_model_info(model_id)

        asr_model_menu.configure(command=_on_model_change)
        _update_model_info(asr_model_display_var.get())

        def _reset_asr() -> None:
            default_model_id = DEFAULT_CONFIG["asr_model_id"]
            default_display = id_to_display.get(default_model_id, default_model_id)
            asr_model_id_var.set(default_model_id)
            asr_model_display_var.set(default_display)
            asr_model_menu.set(default_display)
            asr_backend_var.set(DEFAULT_CONFIG["asr_backend"])
            asr_backend_menu.set(DEFAULT_CONFIG["asr_backend"])
            asr_ct2_compute_type_var.set(DEFAULT_CONFIG["asr_ct2_compute_type"])
            asr_ct2_menu.set(DEFAULT_CONFIG["asr_ct2_compute_type"])
            asr_cache_dir_var.set(DEFAULT_CONFIG["asr_cache_dir"])
            _on_backend_change(asr_backend_var.get())
            _update_model_info(default_model_id)
            self.config_manager.set_asr_model_id(default_model_id)
            self.config_manager.set_asr_backend(DEFAULT_CONFIG["asr_backend"])
            self.config_manager.set_asr_ct2_compute_type(DEFAULT_CONFIG["asr_ct2_compute_type"])
            self.config_manager.set_asr_cache_dir(DEFAULT_CONFIG["asr_cache_dir"])
            self.config_manager.save_config()

        reset_asr_button = ctk.CTkButton(
            asr_model_frame, text="Reset ASR", command=_reset_asr
        )
        reset_asr_button.pack(side="left", padx=5)
        Tooltip(reset_asr_button, "Restore default ASR settings.")

        asr_device_frame = ctk.CTkFrame(asr_frame)
        asr_device_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(asr_device_frame, text="ASR Compute Device:").pack(side="left", padx=(5, 10))
        asr_device_menu = ctk.CTkOptionMenu(asr_device_frame, variable=asr_compute_device_var, values=available_devices)
        asr_device_menu.pack(side="left", padx=5)
        Tooltip(asr_device_menu, "Select compute device for ASR model.")

        asr_dtype_frame = ctk.CTkFrame(asr_frame)
        _register_advanced(asr_dtype_frame, fill="x", pady=5)
        ctk.CTkLabel(asr_dtype_frame, text="ASR DType:").pack(side="left", padx=(5, 0))
        ctk.CTkButton(
            asr_dtype_frame,
            text="?",
            width=20,
            command=lambda: messagebox.showinfo(
                "ASR DType",
                "Torch tensor precision for ASR weights and activations.",
            ),
        ).pack(side="left", padx=(0, 10))
        asr_dtype_menu = ctk.CTkOptionMenu(
            asr_dtype_frame, variable=asr_dtype_var, values=["auto", "float16", "float32"]
        )
        asr_dtype_menu.pack(side="left", padx=5)
        Tooltip(asr_dtype_menu, "Torch dtype for ASR model.")

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

            download_cancelled_error = self._download_cancelled_error

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
                model_manager.ensure_download(
                    model_id,
                    backend,
                    cache_dir,
                    asr_ct2_compute_type_var.get() if backend == "ct2" else None,
                )
                installed_models = model_manager.list_installed(cache_dir)
                self.config_manager.set_asr_installed_models(installed_models)
                self.config_manager.save_config()
                _update_model_info(model_id)
                messagebox.showinfo("Model", "Download completed.")
            except download_cancelled_error:
                messagebox.showinfo("Model", "Download canceled.")
            except OSError:
                messagebox.showerror(
                    "Model",
                    "Unable to write to the ASR cache directory. Please check permissions.",
                )
            except Exception as e:
                messagebox.showerror("Model", f"Download failed: {e}")

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

        _on_model_change(asr_model_display_var.get())
        _update_model_info(asr_model_id_var.get())
        _update_install_button_state()
        _on_backend_change(asr_backend_var.get())
        should_show_advanced = any(
            [
                asr_backend_var.get() not in ("auto", ""),
                asr_dtype_var.get() not in ("auto", ""),
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
                    import torch
                    device_in_use = getattr(th, "device_in_use", None)
                    if device_in_use:
                        device = str(device_in_use)
                    else:
                        device = (
                            f"cuda:{getattr(th, 'gpu_index', -1)}"
                            if torch.cuda.is_available() and getattr(th, 'gpu_index', -1) >= 0
                            else "cpu"
                        )
                    dtype = "fp16" if str(device).startswith("cuda") else "fp32"
                    try:
                        import importlib.util as _spec_util
                        attn_impl = "FA2" if _spec_util.find_spec("flash_attn") is not None else "SDPA"
                    except Exception:
                        attn_impl = "SDPA"
                    chunk = getattr(th, "chunk_length_sec", None)
                    # Se disponível no handler, podemos expor last_dynamic_batch_size; fallback em None
                    bs = getattr(th, "last_dynamic_batch_size", None) if hasattr(th, "last_dynamic_batch_size") else None
                    tech = f" [{device} {dtype} | {attn_impl} | chunk={chunk}s | batch={bs}]"
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
        if context:
            self._last_state_notification = context.get("notification")
            self._state_context_suffix = self._build_state_context_suffix(state_str, context)
        else:
            self._last_state_notification = None
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
                            import torch
                            device_in_use = getattr(th, "device_in_use", None)
                            if device_in_use:
                                device = str(device_in_use)
                            else:
                                device = (
                                    f"cuda:{getattr(th, 'gpu_index', -1)}"
                                    if torch.cuda.is_available() and getattr(th, 'gpu_index', -1) >= 0
                                    else "cpu"
                                )
                            dtype = "fp16" if str(device).startswith("cuda") else "fp32"
                            # Determinar attn_impl conforme detecção feita no handler
                            try:
                                import importlib.util as _spec_util
                                attn_impl = "FA2" if _spec_util.find_spec("flash_attn") is not None else "SDPA"
                            except Exception:
                                attn_impl = "SDPA"
                            chunk = getattr(th, "chunk_length_sec", None)
                            bs = getattr(th, "last_dynamic_batch_size", None)
                            if bs is None:
                                bs = getattr(th, "batch_size", None) if hasattr(th, "batch_size") else None
                            rich_tooltip = _apply_suffix(
                                f"Whisper Recorder (TRANSCRIBING) [{device} {dtype} | {attn_impl} | chunk={chunk}s | batch={bs}]"
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
                    Tooltip(apply_button, "Save all settings and exit.")
                    close_button = ctk.CTkButton(button_frame, text="Cancel", command=self._close_settings_window, fg_color="gray50")
                    close_button.pack(side="right", padx=5)
                    Tooltip(close_button, "Discard changes and exit.")

                    restore_button = ctk.CTkButton(button_frame, text="Restore Defaults", command=self._restore_default_settings)
                    restore_button.pack(side="left", padx=5)

                    force_reregister_button = ctk.CTkButton(
                        button_frame, text="Force Hotkey Re-registration", command=self.core_instance_ref.force_reregister_hotkeys
                    )
                    force_reregister_button.pack(side="left", padx=5)
                    Tooltip(force_reregister_button, "Re-register all global hotkeys.")
                except Exception as btn_err:
                    logging.error(f"Failed to create action buttons: {btn_err}", exc_info=True)

                settings_win.protocol("WM_DELETE_WINDOW", self._close_settings_window)

                self._clear_settings_context()
                self._set_settings_var("window", settings_win)
                model_manager = self.model_manager
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
                    coerce=lambda v: str(v).lower(),
                    allowed={"toggle", "press", "hold"},
                )
                if mode_initial == "press":
                    mode_initial = "hold"
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
                display_transcripts_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        DISPLAY_TRANSCRIPTS_KEY,
                        var_name="display_transcripts_in_terminal",
                        coerce=bool,
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

                enable_torch_compile_var = ctk.BooleanVar(
                    value=self._resolve_initial_value(
                        "enable_torch_compile",
                        var_name="enable_torch_compile",
                        getter=self.config_manager.get_enable_torch_compile,
                        coerce=bool,
                    )
                )

                asr_backend_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        ASR_BACKEND_CONFIG_KEY,
                        var_name="asr_backend",
                        getter=self.config_manager.get_asr_backend,
                        coerce=lambda v: str(v).lower(),
                    )
                )
                asr_model_id_var = ctk.StringVar(
                    value=self._resolve_initial_value(
                        ASR_MODEL_ID_CONFIG_KEY,
                        var_name="asr_model_id",
                        getter=self.config_manager.get_asr_model_id,
                        coerce=str,
                    )
                )
                # New: Chunk length controls
                chunk_length_mode_var = ctk.StringVar(value=self.config_manager.get_chunk_length_mode())
                chunk_length_sec_var = ctk.DoubleVar(value=self.config_manager.get_chunk_length_sec())
                # New: Torch compile switch variable
                enable_torch_compile_var = ctk.BooleanVar(value=self.config_manager.get_enable_torch_compile())
                asr_backend_var = ctk.StringVar(value=self.config_manager.get_asr_backend())
                asr_model_id_var = ctk.StringVar(value=self.config_manager.get_asr_model_id())
                asr_dtype_var = ctk.StringVar(value=self.config_manager.get_asr_dtype())
                asr_ct2_compute_type_var = ctk.StringVar(value=self.config_manager.get_asr_ct2_compute_type())
                asr_cache_dir_var = ctk.StringVar(value=self.config_manager.get_asr_cache_dir())

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
                    ("record_storage_mode_var", record_storage_mode_var),
                    ("max_memory_seconds_mode_var", max_memory_seconds_mode_var),
                    ("max_memory_seconds_var", max_memory_seconds_var),
                    ("chunk_length_mode_var", chunk_length_mode_var),
                    ("chunk_length_sec_var", chunk_length_sec_var),
                    ("enable_torch_compile_var", enable_torch_compile_var),
                    ("asr_backend_var", asr_backend_var),
                    ("asr_model_id_var", asr_model_id_var),
                    ("asr_dtype_var", asr_dtype_var),
                    ("asr_ct2_compute_type_var", asr_ct2_compute_type_var),
                    ("asr_cache_dir_var", asr_cache_dir_var),
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
                    asr_dtype_to_apply = asr_dtype_var.get()
                    asr_ct2_compute_type_to_apply = asr_ct2_compute_type_var.get()
                    asr_cache_dir_to_apply = asr_cache_dir_var.get()
                    try:
                        Path(asr_cache_dir_to_apply).mkdir(parents=True, exist_ok=True)
                    except Exception as e:
                        messagebox.showerror("Invalid Path", f"ASR cache directory is invalid:\n{e}", parent=settings_win)
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
                    asr_dtype_to_apply = asr_dtype_var.get()
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
                            "new_record_to_memory": (record_storage_mode_var.get() == "memory"),
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
                            # New: torch compile setting
                            "new_enable_torch_compile": bool(enable_torch_compile_var.get()),
                            "new_asr_backend": asr_backend_to_apply,
                            "new_asr_model_id": asr_model_id_to_apply,
                            "new_asr_compute_device": asr_compute_device_to_apply,
                            "new_asr_dtype": asr_dtype_to_apply,
                            "new_asr_ct2_compute_type": asr_ct2_compute_type_to_apply,
                            "new_asr_cache_dir": asr_cache_dir_to_apply,
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
                    try:
                        gemini_model_menu
                    except NameError:
                        pass
                    else:
                        gemini_model_menu.configure(values=DEFAULT_CONFIG[GEMINI_MODEL_OPTIONS_CONFIG_KEY])
                        gemini_model_menu.set(DEFAULT_CONFIG[GEMINI_MODEL_CONFIG_KEY])
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
                    display_transcripts_var.set(DEFAULT_CONFIG[DISPLAY_TRANSCRIPTS_KEY])
                    record_storage_mode_var.set(DEFAULT_CONFIG["record_storage_mode"])
                    max_memory_seconds_var.set(DEFAULT_CONFIG["max_memory_seconds"])
                    max_memory_seconds_mode_var.set(DEFAULT_CONFIG["max_memory_seconds_mode"])
                    launch_at_startup_var.set(DEFAULT_CONFIG["launch_at_startup"])
                    asr_backend_var.set(DEFAULT_CONFIG[ASR_BACKEND_CONFIG_KEY])
                    asr_compute_device_var.set(DEFAULT_CONFIG[ASR_COMPUTE_DEVICE_CONFIG_KEY])
                    asr_dtype_var.set(DEFAULT_CONFIG[ASR_DTYPE_CONFIG_KEY])
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
                ctk.CTkLabel(general_frame, text="General Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                # Record Hotkey
                key_frame = ctk.CTkFrame(general_frame)
                key_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(key_frame, text="Record Hotkey:").pack(side="left", padx=(5, 10))
                key_display = ctk.CTkLabel(key_frame, textvariable=detected_key_var, fg_color="gray20", corner_radius=5, width=120)
                key_display.pack(side="left", padx=5)
                Tooltip(key_display, "Current hotkey for recording.")
                
                detect_key_button = ctk.CTkButton(
                    key_frame,
                    text="Detect Key",
                    command=lambda: self._start_key_detection_for("detected_key_var"),
                )
                detect_key_button.pack(side="left", padx=5)
                Tooltip(detect_key_button, "Capture a new recording hotkey.")

                # Agent Hotkey (Moved here)
                agent_key_frame = ctk.CTkFrame(general_frame)
                agent_key_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(agent_key_frame, text="Agent Hotkey:").pack(side="left", padx=(5, 10))
                agent_key_display = ctk.CTkLabel(agent_key_frame, textvariable=agent_key_var, fg_color="gray20", corner_radius=5, width=120)
                agent_key_display.pack(side="left", padx=5)
                Tooltip(agent_key_display, "Current hotkey for agent mode.")
                detect_agent_key_button = ctk.CTkButton(
                    agent_key_frame,
                    text="Detect Key",
                    command=lambda: self._start_key_detection_for("agent_key_var"),
                )
                detect_agent_key_button.pack(side="left", padx=5)
                Tooltip(detect_agent_key_button, "Capture a new agent hotkey.")

                # Recording Mode
                mode_frame = ctk.CTkFrame(general_frame)
                mode_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(mode_frame, text="Recording Mode:").pack(side="left", padx=(5, 10))
                toggle_rb = ctk.CTkRadioButton(mode_frame, text="Toggle", variable=mode_var, value="toggle")
                toggle_rb.pack(side="left", padx=5)
                Tooltip(toggle_rb, "Press once to start or stop recording.")
                hold_rb = ctk.CTkRadioButton(mode_frame, text="Hold", variable=mode_var, value="hold")
                hold_rb.pack(side="left", padx=5)
                Tooltip(hold_rb, "Record only while the key is held down.")

                # Auto-Paste
                paste_frame = ctk.CTkFrame(general_frame)
                paste_frame.pack(fill="x", pady=5)
                paste_switch = ctk.CTkSwitch(paste_frame, text="Auto-Paste", variable=auto_paste_var)
                paste_switch.pack(side="left", padx=5)
                Tooltip(paste_switch, "Automatically paste the transcription.")

                # Hotkey Stability Service
                stability_service_frame = ctk.CTkFrame(general_frame)
                stability_service_frame.pack(fill="x", pady=5)
                stability_switch = ctk.CTkSwitch(stability_service_frame, text="Enable Hotkey Stability Service", variable=hotkey_stability_service_enabled_var)
                stability_switch.pack(side="left", padx=5)
                Tooltip(stability_switch, "Keep hotkeys active when focus changes.")

                startup_frame = ctk.CTkFrame(general_frame)
                startup_frame.pack(fill="x", pady=5)
                startup_switch = ctk.CTkSwitch(startup_frame, text="Launch at Startup", variable=launch_at_startup_var)
                startup_switch.pack(side="left", padx=5)
                Tooltip(startup_switch, "Start the app automatically with Windows.")

                # --- Sound Settings Section ---
                sound_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                sound_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(sound_frame, text="Sound Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")
                
                sound_enabled_frame = ctk.CTkFrame(sound_frame)
                sound_enabled_frame.pack(fill="x", pady=5)
                sound_switch = ctk.CTkSwitch(sound_enabled_frame, text="Enable Sounds", variable=sound_enabled_var)
                sound_switch.pack(side="left", padx=5)
                Tooltip(sound_switch, "Play a beep when recording starts or stops.")

                sound_details_frame = ctk.CTkFrame(sound_frame)
                sound_details_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(sound_details_frame, text="Frequency (Hz):").pack(side="left", padx=(5, 10))
                freq_entry = ctk.CTkEntry(sound_details_frame, textvariable=sound_frequency_var, width=60)
                freq_entry.pack(side="left", padx=5)
                Tooltip(freq_entry, "Beep frequency in hertz.")
                ctk.CTkLabel(sound_details_frame, text="Duration (s):").pack(side="left", padx=(5, 10))
                duration_entry = ctk.CTkEntry(sound_details_frame, textvariable=sound_duration_var, width=60)
                duration_entry.pack(side="left", padx=5)
                Tooltip(duration_entry, "Beep duration in seconds.")
                ctk.CTkLabel(sound_details_frame, text="Volume:").pack(side="left", padx=(5, 10))
                volume_slider = ctk.CTkSlider(sound_details_frame, from_=0.0, to=1.0, variable=sound_volume_var)
                volume_slider.pack(side="left", padx=5, fill="x", expand=True)
                Tooltip(volume_slider, "Beep volume.")

                # --- Text Correction (AI Services) Section ---
                ai_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                ai_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(ai_frame, text="Text Correction (AI Services)", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                text_correction_frame = ctk.CTkFrame(ai_frame)
                text_correction_frame.pack(fill="x", pady=5)
                correction_switch = ctk.CTkSwitch(
                    text_correction_frame,
                    text="Enable Text Correction",
                    variable=text_correction_enabled_var,
                    command=self._update_text_correction_fields,
                )
                correction_switch.pack(side="left", padx=5)
                Tooltip(correction_switch, "Use an AI service to polish the text.")

                service_frame = ctk.CTkFrame(ai_frame)
                service_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(service_frame, text="Service:").pack(side="left", padx=(5, 10))
                service_menu = ctk.CTkOptionMenu(
                    service_frame,
                    variable=text_correction_service_label_var,
                    values=list(service_display_map.keys()),
                    command=self._on_service_menu_change,
                )
                service_menu.pack(side="left", padx=5)
                self._set_settings_var("service_menu", service_menu)
                Tooltip(service_menu, "Select the service for text correction.")
                service_menu.set(text_correction_service_label_var.get())

                # --- OpenRouter Settings ---
                openrouter_frame = ctk.CTkFrame(ai_frame)
                openrouter_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(openrouter_frame, text="OpenRouter API Key:").pack(side="left", padx=(5, 10))
                openrouter_key_entry = ctk.CTkEntry(openrouter_frame, textvariable=openrouter_api_key_var, show="*", width=250)
                openrouter_key_entry.pack(side="left", padx=5)
                self._set_settings_var("openrouter_key_entry", openrouter_key_entry)
                Tooltip(openrouter_key_entry, "API key for the OpenRouter service.")
                ctk.CTkLabel(openrouter_frame, text="OpenRouter Model:").pack(side="left", padx=(5, 10))
                openrouter_model_entry = ctk.CTkEntry(openrouter_frame, textvariable=openrouter_model_var, width=200)
                openrouter_model_entry.pack(side="left", padx=5)
                self._set_settings_var("openrouter_model_entry", openrouter_model_entry)
                Tooltip(openrouter_model_entry, "Model name for OpenRouter.")

                # --- Gemini Settings ---
                gemini_frame = ctk.CTkFrame(ai_frame)
                gemini_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_frame, text="Gemini API Key:").pack(side="left", padx=(5, 10))
                gemini_key_entry = ctk.CTkEntry(gemini_frame, textvariable=gemini_api_key_var, show="*", width=250)
                gemini_key_entry.pack(side="left", padx=5)
                self._set_settings_var("gemini_key_entry", gemini_key_entry)
                Tooltip(gemini_key_entry, "API key for the Gemini service.")
                ctk.CTkLabel(gemini_frame, text="Gemini Model:").pack(side="left", padx=(5, 10))
                gemini_model_menu = ctk.CTkOptionMenu(gemini_frame, variable=gemini_model_var, values=gemini_model_options)
                gemini_model_menu.pack(side="left", padx=5)
                self._set_settings_var("gemini_model_menu", gemini_model_menu)
                Tooltip(gemini_model_menu, "Model used for Gemini requests.")

                # --- Gemini Prompt ---
                gemini_prompt_frame = ctk.CTkFrame(ai_frame)
                gemini_prompt_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_prompt_frame, text="Gemini Correction Prompt:").pack(anchor="w", pady=(5,0))
                gemini_prompt_correction_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=100, wrap="word")
                gemini_prompt_correction_textbox.pack(fill="x", expand=True, pady=5)
                gemini_prompt_correction_textbox.insert("1.0", gemini_prompt_initial)
                self._set_settings_var("gemini_prompt_correction_textbox", gemini_prompt_correction_textbox)
                Tooltip(gemini_prompt_correction_textbox, "Prompt used to refine text.")

                ctk.CTkLabel(gemini_prompt_frame, text="Agent Mode Prompt:").pack(anchor="w", pady=(5,0))
                agentico_prompt_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                agentico_prompt_textbox.pack(fill="x", expand=True, pady=5)
                agentico_prompt_textbox.insert("1.0", agent_prompt_initial)
                self._set_settings_var("agentico_prompt_textbox", agentico_prompt_textbox)
                Tooltip(agentico_prompt_textbox, "Prompt executed in agent mode.")

                ctk.CTkLabel(gemini_prompt_frame, text="Gemini Models (one per line):").pack(anchor="w", pady=(5,0))
                gemini_models_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                gemini_models_textbox.pack(fill="x", expand=True, pady=5)
                gemini_models_textbox.insert("1.0", "\n".join(gemini_model_options))
                self._set_settings_var("gemini_models_textbox", gemini_models_textbox)
                Tooltip(gemini_models_textbox, "List of models to try, one per line.")

                transcription_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                transcription_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(
                    transcription_frame,
                    text="Transcription Settings",
                    font=ctk.CTkFont(weight="bold"),
                ).pack(pady=(5, 10), anchor="w")

                asr_frame = ctk.CTkFrame(transcription_frame, fg_color="transparent")
                asr_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(
                    asr_frame,
                    text="ASR Model",
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
            asr_dtype_var=asr_dtype_var,
            asr_ct2_compute_type_var=asr_ct2_compute_type_var,
            asr_cache_dir_var=asr_cache_dir_var,
            ui_elements={},
        )

        # New: Chunk Length Mode
        chunk_mode_frame = ctk.CTkFrame(transcription_frame)
        chunk_mode_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(chunk_mode_frame, text="Chunk Length Mode:").pack(side="left", padx=(5, 10))
        chunk_mode_menu = ctk.CTkOptionMenu(
            chunk_mode_frame,
            variable=chunk_length_mode_var,
            values=["auto", "manual"],
            command=self._on_chunk_mode_change,
        )
        chunk_mode_menu.pack(side="left", padx=5)
        Tooltip(chunk_mode_menu, "Choose how chunk size is determined.")

        # New: Chunk Length (sec)
        chunk_len_frame = ctk.CTkFrame(transcription_frame)
        chunk_len_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(chunk_len_frame, text="Chunk Length (sec):").pack(side="left", padx=(5, 10))
        chunk_len_entry = ctk.CTkEntry(chunk_len_frame, textvariable=chunk_length_sec_var, width=80)
        chunk_len_entry.pack(side="left", padx=5)
        self._set_settings_var("chunk_len_entry", chunk_len_entry)
        Tooltip(chunk_len_entry, "Fixed chunk duration when in manual mode.")

        # New: Ignore Transcriptions Shorter Than
        min_transcription_duration_frame = ctk.CTkFrame(transcription_frame)
        min_transcription_duration_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            min_transcription_duration_frame,
            text="Ignore Transcriptions Shorter Than (sec):",
        ).pack(side="left", padx=(5, 10))
        min_transcription_duration_entry = ctk.CTkEntry(
            min_transcription_duration_frame,
            textvariable=min_transcription_duration_var,
            width=80,
        )
        min_transcription_duration_entry.pack(side="left", padx=5)
        Tooltip(min_transcription_duration_entry, "Discard segments shorter than this.")

        min_record_duration_frame = ctk.CTkFrame(transcription_frame)
        min_record_duration_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(
            min_record_duration_frame,
            text="Minimum Record Duration (sec):",
        ).pack(side="left", padx=(5, 10))
        min_record_duration_entry = ctk.CTkEntry(
            min_record_duration_frame,
            textvariable=min_record_duration_var,
            width=80,
        )
        min_record_duration_entry.pack(side="left", padx=5)
        Tooltip(min_record_duration_entry, "Discard recordings shorter than this.")

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
                '⏹️ Stop Recording' if is_recording else '▶️ Start Recording',
                lambda: self.core_instance_ref.toggle_recording(),
                default=True,
                enabled=lambda item: self._get_core_state() in ['RECORDING', 'IDLE']
            ),
            pystray.MenuItem(
                '📝 Text Correction',
                lambda: self.toggle_text_correction_from_tray(),
                checked=lambda item: bool(
                    self.config_manager.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY, False)
                ),
            ),
            pystray.MenuItem(
                '⚙️ Settings',
                lambda: self.main_tk_root.after(0, self.run_settings_gui), # Call on main thread
                enabled=lambda item: self._get_core_state() not in ['LOADING_MODEL', 'RECORDING']
            ),
            pystray.MenuItem(
                'Gemini Model',
                pystray.Menu(
                    *[
                        pystray.MenuItem(
                            model,
                            # Action: Pass the menu item text, not the entire object.
                            lambda icon, item: self.core_instance_ref.apply_settings_from_external(new_gemini_model=item.text),
                            radio=True,
                            # Check: Compare the current model with the item text.
                            checked=lambda item: self.config_manager.get('gemini_model') == item.text
                        ) for model in self.config_manager.get(GEMINI_MODEL_OPTIONS_CONFIG_KEY, [])
                    ]
                )
            ),
            pystray.MenuItem(
                'Batch Size',
                pystray.Menu(
                    pystray.MenuItem(
                        'Auto (VRAM)',
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
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('❌ Exit', self.on_exit_app)
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
        """Toggle text correction directly from the tray menu."""
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
