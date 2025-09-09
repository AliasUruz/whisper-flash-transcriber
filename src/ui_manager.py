import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import simpledialog # Adicionado para askstring
import logging
import threading
import time
import pystray
from PIL import Image, ImageDraw

# Importar constantes de configuração
from .config_manager import (
    SETTINGS_WINDOW_GEOMETRY,
    SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI,
    GEMINI_MODEL_OPTIONS_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
    DEFAULT_CONFIG,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
)

from .utils.tooltip import Tooltip

# Importar get_available_devices_for_ui (pode ser movido para um utils ou ficar aqui)
# Por enquanto, vamos assumir que está disponível globalmente ou será movido para cá.
# Para este plano, vamos movê-lo para cá.
import torch # Necessário para get_available_devices_for_ui

# Importar gerenciador de modelos de ASR
try:
    from . import model_manager
except Exception:  # pragma: no cover - fallback caso o módulo não exista
    class _DummyModelManager:
        CURATED = {}

        def asr_installed_models(self):
            return {}

        def ensure_download(self, *_args, **_kwargs):
            logging.warning("model_manager module not available.")

    model_manager = _DummyModelManager()

def get_available_devices_for_ui():
    """Returns a list of devices for the settings interface."""
    devices = ["Auto-select (Recommended)"]
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
    devices.append("Force CPU")
    return devices

class UIManager:
    def __init__(self, main_tk_root, config_manager, core_instance_ref):
        self.main_tk_root = main_tk_root
        self.config_manager = config_manager
        self.core_instance_ref = core_instance_ref # Reference to the AppCore instance

        self.tray_icon = None
        self.settings_window_instance = None
        self.settings_thread_running = False
        self.settings_window_lock = threading.Lock()

        self.live_window = None
        self.live_textbox = None


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

    def _safe_get_int(self, var, field_name, parent):
        """Converte o valor de uma StringVar para int com validação."""
        try:
            return int(var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Valor inválido", f"Valor inválido para {field_name}.", parent=parent)
            return None

    def _safe_get_float(self, var, field_name, parent):
        """Converte o valor de uma StringVar para float com validação."""
        try:
            return float(var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Valor inválido", f"Valor inválido para {field_name}.", parent=parent)
            return None

    def _recording_tooltip_updater(self):
        """Atualiza a tooltip com a duração da gravação a cada segundo."""
        while not self.stop_recording_timer_event.is_set():
            start_time = getattr(self.core_instance_ref.audio_handler, "start_time", None)
            if start_time is None:
                break
            elapsed = time.time() - start_time
            tooltip = f"Whisper Recorder (RECORDING - {self._format_elapsed(elapsed)})"
            self.tray_icon.title = tooltip
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
                    device = f"cuda:{getattr(th, 'gpu_index', -1)}" if torch.cuda.is_available() and getattr(th, 'gpu_index', -1) >= 0 else "cpu"
                    dtype = "fp16" if (torch.cuda.is_available() and getattr(th, 'gpu_index', -1) >= 0) else "fp32"
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
                if self.tray_icon:
                    self.tray_icon.title = tooltip
            except Exception:
                # Em caso de falha, mantém somente o tempo
                elapsed = time.time() - start_ts
                if self.tray_icon:
                    self.tray_icon.title = f"Whisper Recorder (TRANSCRIBING - {self._format_elapsed(elapsed)})"
            time.sleep(1)

    def update_tray_icon(self, state):
        # Logic moved from global, adjusted to use self.tray_icon and self.core_instance_ref
        if self.tray_icon:
            color1, color2 = self.ICON_COLORS.get(state, self.DEFAULT_ICON_COLOR)
            icon_image = self.create_image(64, 64, color1, color2)
            self.tray_icon.icon = icon_image
            tooltip = f"Whisper Recorder ({state})"

            # Controle de threads de tooltip por estado
            if state == "RECORDING":
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

            elif state == "TRANSCRIBING":
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
                            device = f"cuda:{getattr(th, 'gpu_index', -1)}" if torch.cuda.is_available() and getattr(th, 'gpu_index', -1) >= 0 else "cpu"
                            dtype = "fp16" if (torch.cuda.is_available() and getattr(th, 'gpu_index', -1) >= 0) else "fp32"
                            # Determinar attn_impl conforme detecção feita no handler
                            try:
                                import importlib.util as _spec_util
                                attn_impl = "FA2" if _spec_util.find_spec("flash_attn") is not None else "SDPA"
                            except Exception:
                                attn_impl = "SDPA"
                            chunk = getattr(th, "chunk_length_sec", None)
                            # batch_size dinâmico só é conhecido no runtime; mostrar o valor padrão/configurado
                            bs = getattr(th, "batch_size", None) if hasattr(th, "batch_size") else None
                            self.tray_icon.title = f"Whisper Recorder (TRANSCRIBING) [{device} {dtype} | {attn_impl} | chunk={chunk}s | batch={bs}]"
                        else:
                            self.tray_icon.title = "Whisper Recorder (TRANSCRIBING - 00:00)"
                    except Exception:
                        self.tray_icon.title = "Whisper Recorder (TRANSCRIBING - 00:00)"
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

                if state == "IDLE" and self.core_instance_ref:
                    tooltip += f" - Record: {self.core_instance_ref.record_key.upper()} - Agent: {self.core_instance_ref.agent_key.upper()}"
                elif state.startswith("ERROR") and self.core_instance_ref:
                    tooltip += " - Check Logs/Settings"

            # Ajusta tooltip final conforme estado atual para evitar mensagens inconsistentes
            if state == "TRANSCRIBING":
                # Garante que o texto base esteja correto mesmo antes do primeiro tick
                tooltip = "Whisper Recorder (TRANSCRIBING)"
            elif state == "RECORDING":
                tooltip = "Whisper Recorder (RECORDING)"
            elif state == "LOADING_MODEL":
                tooltip = "Whisper Recorder (LOADING_MODEL)"
            else:
                tooltip = tooltip

            self.tray_icon.title = tooltip
            self.tray_icon.update_menu()
            logging.debug(f"Tray icon updated for state: {state}")

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
                text_correction_service_label_var = ctk.StringVar(
                    value=next((label for label, val in service_display_map.items()
                                if val == text_correction_service_var.get()), "None")
                )
                openrouter_api_key_var = ctk.StringVar(value=self.config_manager.get("openrouter_api_key"))
                openrouter_model_var = ctk.StringVar(value=self.config_manager.get("openrouter_model"))
                gemini_api_key_var = ctk.StringVar(value=self.config_manager.get("gemini_api_key"))
                gemini_model_var = ctk.StringVar(value=self.config_manager.get("gemini_model"))
                asr_model_var = ctk.StringVar(value=self.config_manager.get_asr_model())
                batch_size_var = ctk.StringVar(value=str(self.config_manager.get("batch_size")))
                use_vad_var = ctk.BooleanVar(value=self.config_manager.get("use_vad"))
                launch_at_startup_var = ctk.BooleanVar(value=self.config_manager.get("launch_at_startup"))
                vad_threshold_var = ctk.DoubleVar(value=self.config_manager.get("vad_threshold"))
                vad_silence_duration_var = ctk.DoubleVar(value=self.config_manager.get("vad_silence_duration"))
                save_temp_recordings_var = ctk.BooleanVar(value=self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY))
                display_transcripts_var = ctk.BooleanVar(value=self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY))
                record_storage_mode_var = ctk.StringVar(
                    value=self.config_manager.get("record_storage_mode", "auto")
                )
                max_memory_seconds_mode_var = ctk.StringVar(
                    value=self.config_manager.get("max_memory_seconds_mode", "manual")
                )
                max_memory_seconds_var = ctk.DoubleVar(
                    value=self.config_manager.get("max_memory_seconds")
                )
                # New: Chunk length controls
                chunk_length_mode_var = ctk.StringVar(value=self.config_manager.get_chunk_length_mode())
                chunk_length_sec_var = ctk.DoubleVar(value=self.config_manager.get_chunk_length_sec())
                # New: Torch compile switch variable
                enable_torch_compile_var = ctk.BooleanVar(value=self.config_manager.get_enable_torch_compile())

                def update_text_correction_fields():
                    enabled = text_correction_enabled_var.get()
                    state = "normal" if enabled else "disabled"
                    service_menu.configure(state=state)
                    openrouter_key_entry.configure(state=state)
                    openrouter_model_entry.configure(state=state)
                    gemini_key_entry.configure(state=state)
                    gemini_model_menu.configure(state=state)
                    gemini_prompt_correction_textbox.configure(state=state)
                    agentico_prompt_textbox.configure(state=state)
                    gemini_models_textbox.configure(state=state)

                # GPU selection variable
                available_devices = get_available_devices_for_ui()
                current_device_selection = "Auto-select (Recommended)"
                if self.config_manager.get("gpu_index_specified"):
                    if self.config_manager.get("gpu_index") >= 0:
                        for dev in available_devices:
                            if dev.startswith(f"GPU {self.config_manager.get('gpu_index')}"):
                                current_device_selection = dev
                                break
                    elif self.config_manager.get("gpu_index") == -1:
                        current_device_selection = "Force CPU"
                gpu_selection_var = ctk.StringVar(value=current_device_selection)

                # Internal GUI functions (detect_key_task_internal, apply_settings, close_settings, etc.)
                # Will need to be adapted to call methods of self.core_instance_ref and self.config_manager
                # Example of adapted apply_settings:
                def apply_settings():
                    logging.info("Apply settings clicked (in Tkinter thread).")
                    # State validations (moved to AppCore or handled via callbacks)
                    if self.core_instance_ref.current_state in ["RECORDING", "TRANSCRIBING", "LOADING_MODEL"]:
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
                    sound_duration_to_apply = self._safe_get_float(sound_duration_var, "Duração do Som", settings_win)
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
                    asr_model_to_apply = asr_model_var.get()
                    gemini_prompt_correction_to_apply = gemini_prompt_correction_textbox.get("1.0", "end-1c")
                    agentico_prompt_to_apply = agentico_prompt_textbox.get("1.0", "end-1c")
                    batch_size_to_apply = self._safe_get_int(batch_size_var, "Batch Size", settings_win)
                    if batch_size_to_apply is None:
                        return
                    min_transcription_duration_to_apply = self._safe_get_float(min_transcription_duration_var, "Duração Mínima", settings_win)
                    if min_transcription_duration_to_apply is None:
                        return
                    min_record_duration_to_apply = self._safe_get_float(min_record_duration_var, "Duração Mínima da Gravação", settings_win)
                    if min_record_duration_to_apply is None:
                        return
                    use_vad_to_apply = use_vad_var.get()
                    vad_threshold_to_apply = self._safe_get_float(vad_threshold_var, "Limiar do VAD", settings_win)
                    if vad_threshold_to_apply is None:
                        return
                    vad_silence_duration_to_apply = self._safe_get_float(vad_silence_duration_var, "Duração do Silêncio", settings_win)
                    if vad_silence_duration_to_apply is None:
                        return
                    save_temp_recordings_to_apply = save_temp_recordings_var.get()
                    display_transcripts_to_apply = display_transcripts_var.get()
                    max_memory_seconds_mode_to_apply = max_memory_seconds_mode_var.get()
                    max_memory_seconds_to_apply = self._safe_get_float(max_memory_seconds_var, "Max Memory Retention", settings_win)
                    if max_memory_seconds_to_apply is None:
                        return

                    # Logic for converting UI to GPU index
                    selected_device_str = gpu_selection_var.get()
                    gpu_index_to_apply = -1 # Default to "Auto-select"
                    if "Force CPU" in selected_device_str:
                        gpu_index_to_apply = -1 # We use -1 for forced CPU
                    elif selected_device_str.startswith("GPU"):
                        try:
                            gpu_index_to_apply = int(selected_device_str.split(":")[0].replace("GPU", "").strip())
                        except (ValueError, IndexError):
                            messagebox.showerror("Valor inválido", "Índice de GPU inválido.", parent=settings_win)
                            return

                    models_text = gemini_models_textbox.get("1.0", "end-1c")
                    new_models_list = [line.strip() for line in models_text.split("\n") if line.strip()]
                    if not new_models_list:
                        messagebox.showwarning("Invalid Value", "The model list cannot be empty. Please add at least one model.", parent=settings_win)
                        return

                    # Call AppCore method to apply settings
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
                        new_asr_model=asr_model_to_apply,
                        new_gemini_prompt=gemini_prompt_correction_to_apply,
                        prompt_agentico=agentico_prompt_to_apply,
                        new_agent_model=model_to_apply,
                        new_gemini_model_options=new_models_list,
                        new_batch_size=batch_size_to_apply,
                        new_gpu_index=gpu_index_to_apply,
                        new_hotkey_stability_service_enabled=hotkey_stability_service_enabled_to_apply, # Nova configuração unificada
                        new_min_transcription_duration=min_transcription_duration_to_apply,
                        new_min_record_duration=min_record_duration_to_apply,
                        new_save_temp_recordings=save_temp_recordings_to_apply,
                        new_record_storage_mode=record_storage_mode_var.get(),
                        new_record_to_memory=(record_storage_mode_var.get() == "memory"),
                        new_max_memory_seconds_mode=max_memory_seconds_mode_to_apply,
                        new_max_memory_seconds=max_memory_seconds_to_apply,
                        new_use_vad=use_vad_to_apply,
                        new_vad_threshold=vad_threshold_to_apply,
                        new_vad_silence_duration=vad_silence_duration_to_apply,
                        new_display_transcripts_in_terminal=display_transcripts_to_apply,
                        new_launch_at_startup=launch_at_startup_var.get(),
                        # New chunk settings
                        new_chunk_length_mode=chunk_length_mode_var.get(),
                        new_chunk_length_sec=float(chunk_length_sec_var.get()),
                        # New: torch compile setting
                        new_enable_torch_compile=bool(enable_torch_compile_var.get())
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
                    agent_model_var.set(DEFAULT_CONFIG["gemini_agent_model"])
                    hotkey_stability_service_enabled_var.set(DEFAULT_CONFIG["hotkey_stability_service_enabled"])
                    min_transcription_duration_var.set(DEFAULT_CONFIG["min_transcription_duration"])
                    min_record_duration_var.set(DEFAULT_CONFIG["min_record_duration"])
                    sound_enabled_var.set(DEFAULT_CONFIG["sound_enabled"])
                    sound_frequency_var.set(str(DEFAULT_CONFIG["sound_frequency"]))
                    sound_duration_var.set(str(DEFAULT_CONFIG["sound_duration"]))
                    sound_volume_var.set(DEFAULT_CONFIG["sound_volume"])
                    text_correction_enabled_var.set(DEFAULT_CONFIG["text_correction_enabled"])
                    text_correction_service_var.set(DEFAULT_CONFIG["text_correction_service"])
                    text_correction_service_label_var.set(
                        next(
                            (
                                label
                                for label, val in service_display_map.items()
                                if val == DEFAULT_CONFIG["text_correction_service"]
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
                        update_text_correction_fields()
                    openrouter_api_key_var.set(DEFAULT_CONFIG["openrouter_api_key"])
                    openrouter_model_var.set(DEFAULT_CONFIG["openrouter_model"])
                    gemini_api_key_var.set(DEFAULT_CONFIG["gemini_api_key"])
                    gemini_model_var.set(DEFAULT_CONFIG["gemini_model"])
                    asr_model_var.set(DEFAULT_CONFIG["asr_model"])
                    gemini_prompt_correction_textbox.delete("1.0", "end")
                    gemini_prompt_correction_textbox.insert("1.0", DEFAULT_CONFIG["gemini_prompt"])
                    agentico_prompt_textbox.delete("1.0", "end")
                    agentico_prompt_textbox.insert("1.0", DEFAULT_CONFIG["prompt_agentico"])
                    gemini_models_textbox.delete("1.0", "end")
                    gemini_models_textbox.insert("1.0", "\n".join(DEFAULT_CONFIG["gemini_model_options"]))
                    batch_size_var.set(str(DEFAULT_CONFIG["batch_size"]))

                    if DEFAULT_CONFIG["gpu_index"] == -1:
                        gpu_selection_var.set("Force CPU")
                    else:
                        found = "Auto-select (Recommended)"
                        for dev in available_devices:
                            if dev.startswith(f"GPU {DEFAULT_CONFIG['gpu_index']}"):
                                found = dev
                                break
                        gpu_selection_var.set(found)

                    use_vad_var.set(DEFAULT_CONFIG["use_vad"])
                    vad_threshold_var.set(DEFAULT_CONFIG["vad_threshold"])
                    vad_silence_duration_var.set(DEFAULT_CONFIG["vad_silence_duration"])
                    save_temp_recordings_var.set(DEFAULT_CONFIG[SAVE_TEMP_RECORDINGS_CONFIG_KEY])
                    display_transcripts_var.set(DEFAULT_CONFIG["display_transcripts_in_terminal"])
                    record_storage_mode_var.set(DEFAULT_CONFIG["record_storage_mode"])
                    max_memory_seconds_var.set(DEFAULT_CONFIG["max_memory_seconds"])
                    max_memory_seconds_mode_var.set(DEFAULT_CONFIG["max_memory_seconds_mode"])
                    launch_at_startup_var.set(DEFAULT_CONFIG["launch_at_startup"])

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
                
                def detect_key_task_internal(key_var):
                    key_var.set("PRESS KEY...")
                    settings_win.update_idletasks()
                    # Pass the callback to AppCore to update the StringVar
                    self.core_instance_ref.set_key_detection_callback(lambda key: key_var.set(key))
                    self.core_instance_ref.start_key_detection_thread()

                detect_key_button = ctk.CTkButton(key_frame, text="Detect Key", command=lambda: detect_key_task_internal(detected_key_var))
                detect_key_button.pack(side="left", padx=5)
                Tooltip(detect_key_button, "Capture a new recording hotkey.")

                # Agent Hotkey (Moved here)
                agent_key_frame = ctk.CTkFrame(general_frame)
                agent_key_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(agent_key_frame, text="Agent Hotkey:").pack(side="left", padx=(5, 10))
                agent_key_display = ctk.CTkLabel(agent_key_frame, textvariable=agent_key_var, fg_color="gray20", corner_radius=5, width=120)
                agent_key_display.pack(side="left", padx=5)
                Tooltip(agent_key_display, "Current hotkey for agent mode.")
                detect_agent_key_button = ctk.CTkButton(agent_key_frame, text="Detect Key", command=lambda: detect_key_task_internal(agent_key_var))
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
                correction_switch = ctk.CTkSwitch(text_correction_frame, text="Enable Text Correction", variable=text_correction_enabled_var, command=update_text_correction_fields)
                correction_switch.pack(side="left", padx=5)
                Tooltip(correction_switch, "Use an AI service to polish the text.")

                service_frame = ctk.CTkFrame(ai_frame)
                service_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(service_frame, text="Service:").pack(side="left", padx=(5, 10))
                service_menu = ctk.CTkOptionMenu(
                    service_frame,
                    variable=text_correction_service_label_var,
                    values=list(service_display_map.keys()),
                    command=lambda choice: text_correction_service_var.set(service_display_map[choice]),
                )
                service_menu.pack(side="left", padx=5)
                Tooltip(service_menu, "Select the service for text correction.")

                # --- OpenRouter Settings ---
                openrouter_frame = ctk.CTkFrame(ai_frame)
                openrouter_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(openrouter_frame, text="OpenRouter API Key:").pack(side="left", padx=(5, 10))
                openrouter_key_entry = ctk.CTkEntry(openrouter_frame, textvariable=openrouter_api_key_var, show="*", width=250)
                openrouter_key_entry.pack(side="left", padx=5)
                Tooltip(openrouter_key_entry, "API key for the OpenRouter service.")
                ctk.CTkLabel(openrouter_frame, text="OpenRouter Model:").pack(side="left", padx=(5, 10))
                openrouter_model_entry = ctk.CTkEntry(openrouter_frame, textvariable=openrouter_model_var, width=200)
                openrouter_model_entry.pack(side="left", padx=5)
                Tooltip(openrouter_model_entry, "Model name for OpenRouter.")

                # --- Gemini Settings ---
                gemini_frame = ctk.CTkFrame(ai_frame)
                gemini_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_frame, text="Gemini API Key:").pack(side="left", padx=(5, 10))
                gemini_key_entry = ctk.CTkEntry(gemini_frame, textvariable=gemini_api_key_var, show="*", width=250)
                gemini_key_entry.pack(side="left", padx=5)
                Tooltip(gemini_key_entry, "API key for the Gemini service.")
                ctk.CTkLabel(gemini_frame, text="Gemini Model:").pack(side="left", padx=(5, 10))
                gemini_model_menu = ctk.CTkOptionMenu(gemini_frame, variable=gemini_model_var, values=self.config_manager.get("gemini_model_options", []))
                gemini_model_menu.pack(side="left", padx=5)
                Tooltip(gemini_model_menu, "Model used for Gemini requests.")
                
                # --- Gemini Prompt ---
                gemini_prompt_frame = ctk.CTkFrame(ai_frame)
                gemini_prompt_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_prompt_frame, text="Gemini Correction Prompt:").pack(anchor="w", pady=(5,0))
                gemini_prompt_correction_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=100, wrap="word")
                gemini_prompt_correction_textbox.pack(fill="x", expand=True, pady=5)
                gemini_prompt_correction_textbox.insert("1.0", self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY))
                Tooltip(gemini_prompt_correction_textbox, "Prompt used to refine text.")

                ctk.CTkLabel(gemini_prompt_frame, text="Prompt do Modo Agêntico:").pack(anchor="w", pady=(5,0))
                agentico_prompt_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                agentico_prompt_textbox.pack(fill="x", expand=True, pady=5)
                agentico_prompt_textbox.insert("1.0", self.config_manager.get("prompt_agentico"))
                Tooltip(agentico_prompt_textbox, "Prompt executed in agent mode.")

                ctk.CTkLabel(gemini_prompt_frame, text="Gemini Models (one per line):").pack(anchor="w", pady=(5,0))
                gemini_models_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                gemini_models_textbox.pack(fill="x", expand=True, pady=5)
                gemini_models_textbox.insert("1.0", "\n".join(self.config_manager.get("gemini_model_options", [])))
                Tooltip(gemini_models_textbox, "List of models to try, one per line.")

                # --- ASR Settings ---
                asr_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                asr_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(asr_frame, text="ASR", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                # Backend selection
                asr_backend_frame = ctk.CTkFrame(asr_frame)
                asr_backend_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(asr_backend_frame, text="Backend:").pack(side="left", padx=(5, 10))
                asr_backend_var = ctk.StringVar(value=self.config_manager.get("asr_backend", "Transformers"))
                backend_menu = ctk.CTkOptionMenu(
                    asr_backend_frame,
                    variable=asr_backend_var,
                    values=["Transformers", "Faster-Whisper"],
                )
                backend_menu.pack(side="left", padx=5)
                Tooltip(backend_menu, "Mecanismo de reconhecimento de fala.")

                # Model selection
                asr_model_frame = ctk.CTkFrame(asr_frame)
                asr_model_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(asr_model_frame, text="Modelo:").pack(side="left", padx=(5, 10))

                installed = (
                    model_manager.asr_installed_models()
                    if hasattr(model_manager, "asr_installed_models")
                    else {}
                )
                curated = getattr(model_manager, "CURATED", {})
                combined_models = {**installed, **curated}
                asr_model_var = ctk.StringVar(
                    value=self.config_manager.get("asr_model", "")
                )
                model_menu = ctk.CTkOptionMenu(
                    asr_model_frame, variable=asr_model_var, values=[]
                )
                model_menu.pack(side="left", padx=5)
                model_tooltip = Tooltip(model_menu, "")

                def _update_model_tooltip(name: str) -> None:
                    info = combined_models.get(name, {})
                    tip = " | ".join(
                        str(info.get(k, "?"))
                        for k in ("backend", "dtype", "chunk", "batch", "device")
                    )
                    model_tooltip.text = tip

                def _refresh_model_options() -> None:
                    backend = asr_backend_var.get()
                    options = [
                        m for m, meta in combined_models.items() if meta.get("backend") == backend
                    ]
                    model_menu.configure(values=options)
                    if asr_model_var.get() not in options and options:
                        asr_model_var.set(options[0])
                        self.config_manager.set("asr_model", options[0])
                    _update_model_tooltip(asr_model_var.get())

                def _reload_if_idle() -> None:
                    handler = getattr(self.core_instance_ref, "transcription_handler", None)
                    state = getattr(self.core_instance_ref, "current_state", "IDLE")
                    if handler and state not in ["LOADING_MODEL", "RECORDING", "TRANSCRIBING"]:
                        try:
                            handler.start_model_loading()
                        except Exception as e:  # pragma: no cover
                            logging.error("Model reload failed: %s", e)
                            messagebox.showerror("Model", f"Falha ao carregar modelo: {e}")

                def _on_backend_change(choice: str) -> None:
                    prev = self.config_manager.get("asr_backend")
                    self.config_manager.set("asr_backend", choice)
                    self.config_manager.save_config()
                    _refresh_model_options()
                    if prev != choice:
                        _reload_if_idle()

                def _on_model_change(choice: str) -> None:
                    prev = self.config_manager.get("asr_model")
                    self.config_manager.set("asr_model", choice)
                    self.config_manager.save_config()
                    _update_model_tooltip(choice)
                    if prev != choice:
                        _reload_if_idle()

                backend_menu.configure(command=_on_backend_change)
                model_menu.configure(command=_on_model_change)
                _refresh_model_options()

                def _install_model() -> None:
                    try:
                        model_manager.ensure_download(
                            asr_backend_var.get(), asr_model_var.get()
                        )
                    except Exception as e:  # pragma: no cover
                        logging.error("Model download failed: %s", e)
                        messagebox.showerror(
                            "Model Manager", f"Falha ao baixar modelo: {e}"
                        )
                    else:
                        messagebox.showinfo(
                            "Model Manager", "Modelo instalado/atualizado com sucesso."
                        )

                install_button = ctk.CTkButton(
                    asr_frame, text="Instalar/Atualizar", command=_install_model
                )
                install_button.pack(pady=5)
                Tooltip(install_button, "Baixa ou atualiza o modelo selecionado.")


                # --- Transcription Settings (Advanced) Section ---
                transcription_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                transcription_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(transcription_frame, text="Transcription Settings (Advanced)", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                asr_models = self.config_manager.get_asr_installed_models()
                if not asr_models:
                    asr_models = [self.config_manager.get_asr_model()]
                asr_model_frame = ctk.CTkFrame(transcription_frame)
                asr_model_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(asr_model_frame, text="ASR Model:").pack(side="left", padx=(5, 10))
                asr_model_menu = ctk.CTkOptionMenu(asr_model_frame, variable=asr_model_var, values=asr_models)
                asr_model_menu.pack(side="left", padx=5)
                Tooltip(asr_model_menu, "Select the speech recognition model.")

                device_frame = ctk.CTkFrame(transcription_frame)
                device_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(device_frame, text="Processing Device:").pack(side="left", padx=(5, 10))
                device_menu = ctk.CTkOptionMenu(device_frame, variable=gpu_selection_var, values=available_devices)
                device_menu.pack(side="left", padx=5)
                Tooltip(device_menu, "Select CPU or GPU for processing.")

                batch_size_frame = ctk.CTkFrame(transcription_frame)
                batch_size_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(batch_size_frame, text="Batch Size:").pack(side="left", padx=(5, 10))
                batch_entry = ctk.CTkEntry(batch_size_frame, textvariable=batch_size_var, width=60)
                batch_entry.pack(side="left", padx=5)
                Tooltip(batch_entry, "Number of segments processed together.")

                # New: Torch Compile
                torch_compile_frame = ctk.CTkFrame(transcription_frame)
                torch_compile_frame.pack(fill="x", pady=5)
                torch_compile_switch = ctk.CTkSwitch(
                    torch_compile_frame,
                    text="Enable Torch Compile (Experimental)",
                    variable=enable_torch_compile_var
                )
                torch_compile_switch.pack(side="left", padx=5)
                Tooltip(torch_compile_switch, "Enable PyTorch compile (may improve performance on supported GPUs).")

                # New: Chunk Length Mode
                chunk_mode_frame = ctk.CTkFrame(transcription_frame)
                chunk_mode_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(chunk_mode_frame, text="Chunk Length Mode:").pack(side="left", padx=(5, 10))
                chunk_mode_menu = ctk.CTkOptionMenu(chunk_mode_frame, variable=chunk_length_mode_var, values=["auto", "manual"])
                chunk_mode_menu.pack(side="left", padx=5)
                Tooltip(chunk_mode_menu, "Choose how chunk size is determined.")

                # New: Chunk Length (sec)
                chunk_len_frame = ctk.CTkFrame(transcription_frame)
                chunk_len_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(chunk_len_frame, text="Chunk Length (sec):").pack(side="left", padx=(5, 10))
                chunk_len_entry = ctk.CTkEntry(chunk_len_frame, textvariable=chunk_length_sec_var, width=80)
                chunk_len_entry.pack(side="left", padx=5)
                Tooltip(chunk_len_entry, "Fixed chunk duration when in manual mode.")

                def update_chunk_length_state():
                    try:
                        mode_val = chunk_length_mode_var.get().lower()
                    except Exception:
                        mode_val = "manual"
                    state = "normal" if mode_val == "manual" else "disabled"
                    try:
                        chunk_len_entry.configure(state=state)
                    except Exception:
                        pass

                # initialize state
                update_chunk_length_state()
                # bind changes
                chunk_mode_menu.configure(command=lambda _: update_chunk_length_state())
    
                # New: Ignore Transcriptions Shorter Than
                min_transcription_duration_frame = ctk.CTkFrame(transcription_frame)
                min_transcription_duration_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(min_transcription_duration_frame, text="Ignore Transcriptions Shorter Than (sec):").pack(side="left", padx=(5, 10))
                min_transcription_duration_entry = ctk.CTkEntry(min_transcription_duration_frame, textvariable=min_transcription_duration_var, width=80)
                min_transcription_duration_entry.pack(side="left", padx=5)
                Tooltip(min_transcription_duration_entry, "Discard segments shorter than this.")

                min_record_duration_frame = ctk.CTkFrame(transcription_frame)
                min_record_duration_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(min_record_duration_frame, text="Minimum Record Duration (sec):").pack(side="left", padx=(5, 10))
                min_record_duration_entry = ctk.CTkEntry(min_record_duration_frame, textvariable=min_record_duration_var, width=80)
                min_record_duration_entry.pack(side="left", padx=5)
                Tooltip(min_record_duration_entry, "Discard recordings shorter than this.")

                vad_enable_frame = ctk.CTkFrame(transcription_frame)
                vad_enable_frame.pack(fill="x", pady=5)
                vad_checkbox = ctk.CTkCheckBox(vad_enable_frame, text="Use VAD", variable=use_vad_var)
                vad_checkbox.pack(side="left", padx=5)
                Tooltip(vad_checkbox, "Enable voice activity detection.")

                vad_params_frame = ctk.CTkFrame(transcription_frame)
                vad_params_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(vad_params_frame, text="VAD Threshold:").pack(side="left", padx=(5, 10))
                vad_threshold_entry = ctk.CTkEntry(vad_params_frame, textvariable=vad_threshold_var, width=60)
                vad_threshold_entry.pack(side="left", padx=5)
                Tooltip(vad_threshold_entry, "Voice probability to trigger splitting.")
                ctk.CTkLabel(vad_params_frame, text="Duração do silêncio (s):").pack(side="left", padx=(5, 10))
                vad_silence_entry = ctk.CTkEntry(vad_params_frame, textvariable=vad_silence_duration_var, width=60)
                vad_silence_entry.pack(side="left", padx=5)
                Tooltip(vad_silence_entry, "Length of silence before a cut.")

                temp_recordings_frame = ctk.CTkFrame(transcription_frame)
                temp_recordings_frame.pack(fill="x", pady=5)
                temp_recordings_switch = ctk.CTkSwitch(temp_recordings_frame, text="Save Temporary Recordings", variable=save_temp_recordings_var)
                temp_recordings_switch.pack(side="left", padx=5)
                Tooltip(temp_recordings_switch, "Keep temporary audio files after processing.")

                storage_mode_frame = ctk.CTkFrame(transcription_frame)
                storage_mode_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(storage_mode_frame, text="Record Storage Mode:").pack(side="left", padx=(5, 10))
                storage_mode_menu = ctk.CTkOptionMenu(
                    storage_mode_frame,
                    variable=record_storage_mode_var,
                    values=["auto", "memory", "disk"],
                )
                storage_mode_menu.pack(side="left", padx=5)
                Tooltip(storage_mode_menu, "Where recordings are kept during capture.")

                mem_time_frame = ctk.CTkFrame(transcription_frame)
                mem_time_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(mem_time_frame, text="Max Memory Retention (s):").pack(side="left", padx=(5, 10))
                mem_time_entry = ctk.CTkEntry(mem_time_frame, textvariable=max_memory_seconds_var, width=60)
                mem_time_entry.pack(side="left", padx=5)
                Tooltip(mem_time_entry, "Limit for in-memory recordings.")
                mem_mode_menu = ctk.CTkOptionMenu(
                    mem_time_frame,
                    variable=max_memory_seconds_mode_var,
                    values=["manual", "auto"],
                    width=80,
                )
                mem_mode_menu.pack(side="left", padx=5)
                Tooltip(mem_mode_menu, "Choose manual or auto calculation.")

                display_transcripts_frame = ctk.CTkFrame(transcription_frame)
                display_transcripts_frame.pack(fill="x", pady=5)
                display_switch = ctk.CTkSwitch(display_transcripts_frame, text="Display Transcript in Terminal", variable=display_transcripts_var)
                display_switch.pack(side="left", padx=5)
                Tooltip(display_switch, "Print transcripts to the terminal window.")

                # --- Action Buttons ---
                button_frame = ctk.CTkFrame(settings_win) # Move outside scrollable_frame to keep fixed
                button_frame.pack(fill="x", padx=10, pady=(20, 10))
                
                apply_button = ctk.CTkButton(button_frame, text="Apply and Close", command=apply_settings)
                apply_button.pack(side="right", padx=5)
                Tooltip(apply_button, "Save all settings and exit.")
                close_button = ctk.CTkButton(button_frame, text="Cancel", command=self._close_settings_window, fg_color="gray50")
                close_button.pack(side="right", padx=5)
                Tooltip(close_button, "Discard changes and exit.")

                restore_button = ctk.CTkButton(button_frame, text="Restore Defaults", command=restore_defaults)
                restore_button.pack(side="left", padx=5)

                force_reregister_button = ctk.CTkButton(button_frame, text="Force Hotkey Re-registration", command=self.core_instance_ref.force_reregister_hotkeys)
                force_reregister_button.pack(side="left", padx=5)
                Tooltip(force_reregister_button, "Re-register all global hotkeys.")

                update_text_correction_fields()

                settings_win.protocol("WM_DELETE_WINDOW", self._close_settings_window)

            except Exception as e:
                logging.error(f"Failed to create Toplevel for settings: {e}", exc_info=True)
                self.settings_thread_running = False # Ensure flag is cleared in case of error
                return

    def setup_tray_icon(self):
        # Logic moved from global, adjusted to use self.
        initial_state = self.core_instance_ref.current_state
        color1, color2 = self.ICON_COLORS.get(initial_state, self.DEFAULT_ICON_COLOR)
        initial_image = self.create_image(64, 64, color1, color2)
        initial_tooltip = f"Whisper Recorder ({initial_state})"

        self.tray_icon = pystray.Icon(
            "whisper_recorder",
            initial_image,
            initial_tooltip,
            menu=pystray.Menu(lambda: self.create_dynamic_menu())
        )
        # Set update callback in core_instance
        self.core_instance_ref.set_state_update_callback(self.update_tray_icon)
        self.core_instance_ref.set_segment_callback(self.update_live_transcription_threadsafe) # Connect segment callback

        def run_tray_icon_in_thread(icon):
            icon.run()

        tray_thread = threading.Thread(target=run_tray_icon_in_thread, args=(self.tray_icon,), daemon=True, name="PystrayThread")
        tray_thread.start()
    
    def _close_settings_window(self):
        """Closes the settings window and resets the flag."""
        with self.settings_window_lock:
            if self.settings_window_instance:
                self.settings_window_instance.destroy()
                self.settings_window_instance = None
            self.settings_thread_running = False

    def create_dynamic_menu(self):
        # Logic moved from global, adjusted to use self.core_instance_ref
        # ...
        current_state = self.core_instance_ref.current_state
        is_recording = current_state == "RECORDING"
        is_idle = current_state == "IDLE"
        is_loading = current_state == "LOADING_MODEL"

        menu_items = [
            pystray.MenuItem(
                '⏹️ Stop Recording' if is_recording else '▶️ Start Recording',
                lambda: self.core_instance_ref.toggle_recording(),
                default=True,
                enabled=lambda item: self.core_instance_ref.current_state in ['RECORDING', 'IDLE']
            ),
            pystray.MenuItem(
                '⚙️ Settings',
                lambda: self.main_tk_root.after(0, self.run_settings_gui), # Call on main thread
                enabled=lambda item: self.core_instance_ref.current_state not in ['LOADING_MODEL', 'RECORDING']
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
                    messagebox.showerror("Erro de Entrada", "O Batch Size deve ser um número inteiro positivo.", parent=self.settings_window_instance)
            except ValueError:
                messagebox.showerror("Erro de Entrada", "Entrada inválida. Por favor, insira um número inteiro.", parent=self.settings_window_instance)
