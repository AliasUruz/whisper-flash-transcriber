import customtkinter as ctk
from typing import Dict, Any, Callable
import logging
import threading # Necessário para a detecção de hotkeys
import time      # Necessário para delays na detecção de hotkeys

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, core_instance):
        super().__init__(master)
        self.core_instance = core_instance
        self.initial_config = core_instance.config
        self.is_dirty = False

        self.title("Whisper Recorder Settings")
        self.geometry("620x600")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.attributes("-topmost", True)
        self.transient(master)
        self.grab_set()

        self.tab_view = ctk.CTkTabview(self, anchor="nw")
        self.tab_view.pack(padx=10, pady=10, fill="both", expand=True)
        self.tab_view.add("General"); self.tab_view.add("Hotkeys"); self.tab_view.add("AI Correction"); self.tab_view.add("Advanced")

        self.general_frame = GeneralSettingsFrame(self.tab_view.tab("General"), self.initial_config, self.set_dirty)
        self.hotkeys_frame = HotkeysSettingsFrame(self.tab_view.tab("Hotkeys"), self.initial_config, self.set_dirty, self.core_instance)
        self.correction_frame = CorrectionSettingsFrame(self.tab_view.tab("AI Correction"), self.initial_config, self.set_dirty)
        self.advanced_frame = AdvancedSettingsFrame(self.tab_view.tab("Advanced"), self.initial_config, self.set_dirty)

        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(padx=10, pady=(0, 10), fill="x", side="bottom")
        self.apply_button = ctk.CTkButton(self.action_frame, text="Apply", command=self.apply_and_close, state="disabled")
        self.apply_button.pack(side="right", padx=(10, 0))
        self.cancel_button = ctk.CTkButton(self.action_frame, text="Cancel", command=self.destroy, fg_color="#555555", hover_color="#6E6E6E")
        self.cancel_button.pack(side="right")

    def set_dirty(self, *args):
        if not self.is_dirty:
            self.is_dirty = True
            self.apply_button.configure(state="normal")

    def apply_and_close(self):
        if not self.is_dirty:
            self.destroy()
            return

        final_settings = {}
        final_settings.update(self.general_frame.get_settings())
        final_settings.update(self.hotkeys_frame.get_settings()) # Adicionar hotkeys
        final_settings.update(self.correction_frame.get_settings()) # Adicionar correção com IA
        final_settings.update(self.advanced_frame.get_settings()) # Adicionar avançado

        self.core_instance.apply_settings_from_external(**final_settings)
        self.settings_applied = True # Flag para indicar que as configurações foram aplicadas
        self.destroy()

class GeneralSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, initial_config: Dict[str, Any], set_dirty_callback: Callable):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="both", expand=True, padx=5, pady=5)

        # Auto Paste
        self.auto_paste_var = ctk.BooleanVar(value=initial_config.get("auto_paste"))
        auto_paste_switch = ctk.CTkSwitch(self, text="Auto-paste text...", variable=self.auto_paste_var)
        auto_paste_switch.pack(anchor="w", pady=(5, 20), padx=5)
        self.auto_paste_var.trace_add("write", set_dirty_callback)

        # Sound Settings
        sound_frame = ctk.CTkFrame(self, fg_color="transparent")
        sound_frame.pack(fill="x", expand=True, anchor="w", padx=5)
        ctk.CTkLabel(sound_frame, text="Sound Feedback", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))

        self.sound_enabled_var = ctk.BooleanVar(value=initial_config.get("sound_enabled"))
        sound_enabled_switch = ctk.CTkSwitch(sound_frame, text="Enable sounds...", variable=self.sound_enabled_var)
        sound_enabled_switch.pack(anchor="w", padx=10, pady=5)
        self.sound_enabled_var.trace_add("write", set_dirty_callback)

        ctk.CTkLabel(sound_frame, text="Volume").pack(anchor="w", padx=10, pady=(5, 0))
        self.sound_volume_var = ctk.DoubleVar(value=initial_config.get("sound_volume"))
        sound_volume_slider = ctk.CTkSlider(sound_frame, from_=0.0, to=1.0, variable=self.sound_volume_var)
        sound_volume_slider.pack(fill="x", padx=10, pady=(0, 10))
        self.sound_volume_var.trace_add("write", set_dirty_callback)

        grid_frame = ctk.CTkFrame(sound_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=10)
        grid_frame.columnconfigure((0, 1), weight=1)

        # Frequency
        ctk.CTkLabel(grid_frame, text="Frequency (Hz)").grid(row=0, column=0, sticky="w", padx=5)
        self.sound_frequency_var = ctk.StringVar(value=str(initial_config.get("sound_frequency")))
        sound_frequency_entry = ctk.CTkEntry(grid_frame, textvariable=self.sound_frequency_var)
        sound_frequency_entry.grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.sound_frequency_var.trace_add("write", set_dirty_callback)

        # Duration
        ctk.CTkLabel(grid_frame, text="Duration (s)").grid(row=0, column=1, sticky="w", padx=5)
        self.sound_duration_var = ctk.StringVar(value=str(initial_config.get("sound_duration")))
        sound_duration_entry = ctk.CTkEntry(grid_frame, textvariable=self.sound_duration_var)
        sound_duration_entry.grid(row=1, column=1, sticky="ew", padx=(5, 0))
        self.sound_duration_var.trace_add("write", set_dirty_callback)

    def get_settings(self) -> Dict[str, Any]:
        settings = {
            "auto_paste": self.auto_paste_var.get(),
            "sound_enabled": self.sound_enabled_var.get(),
            "sound_volume": self.sound_volume_var.get(),
        }
        # Import DEFAULT_CONFIG and logging from whisper_tkinter or define a placeholder
        try:
            from whisper_tkinter import DEFAULT_CONFIG
        except ImportError:
            logging.warning("Could not import DEFAULT_CONFIG from whisper_tkinter in GeneralSettingsFrame. Using a placeholder.")
            DEFAULT_CONFIG = {"sound_frequency": 400, "sound_duration": 0.3} # Minimal placeholder

        try:
            settings["sound_frequency"] = int(self.sound_frequency_var.get())
        except ValueError:
            settings["sound_frequency"] = DEFAULT_CONFIG["sound_frequency"] # Fallback
            logging.warning(f"Invalid sound frequency value: {self.sound_frequency_var.get()}. Using default.")
        try:
            settings["sound_duration"] = float(self.sound_duration_var.get())
        except ValueError:
            settings["sound_duration"] = DEFAULT_CONFIG["sound_duration"] # Fallback
            logging.warning(f"Invalid sound duration value: {self.sound_duration_var.get()}. Using default.")
        return settings

class HotkeysSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, initial_config: Dict[str, Any], set_dirty_callback: Callable, core_instance):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="both", expand=True, padx=5, pady=5)

        self.initial_config = initial_config
        self.set_dirty_callback = set_dirty_callback
        self.core_instance = core_instance # Referência à instância do WhisperCore

        # Variáveis
        self.record_key_var = ctk.StringVar(value=initial_config.get("record_key", "F3").upper())
        self.record_mode_var = ctk.StringVar(value=initial_config.get("record_mode", "toggle"))
        self.agent_key_var = ctk.StringVar(value=initial_config.get("agent_key", "F5").upper())
        self.agent_model_var = ctk.StringVar(value=initial_config.get("gemini_agent_model", "gemini-2.0-flash-001"))
        self.agent_auto_paste_var = ctk.BooleanVar(value=initial_config.get("agent_auto_paste", True))
        self.auto_reregister_var = ctk.BooleanVar(value=initial_config.get("auto_reregister_hotkeys", True))

        # Trace changes to variables
        self.record_key_var.trace_add("write", self.set_dirty_callback)
        self.record_mode_var.trace_add("write", self.set_dirty_callback)
        self.agent_key_var.trace_add("write", self.set_dirty_callback)
        self.agent_model_var.trace_add("write", self.set_dirty_callback)
        self.agent_auto_paste_var.trace_add("write", self.set_dirty_callback)
        self.auto_reregister_var.trace_add("write", self.set_dirty_callback)

        # --- Seção de Hotkey de Gravação ---
        record_hotkey_frame = ctk.CTkFrame(self, fg_color="transparent")
        record_hotkey_frame.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(record_hotkey_frame, text="Record Hotkey", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))

        key_display_label = ctk.CTkLabel(record_hotkey_frame, textvariable=self.record_key_var, font=("Segoe UI", 12, "bold"), width=120, fg_color="#393E46", text_color="#00a0ff", corner_radius=8)
        key_display_label.pack(side="left", padx=5)
        self.detect_record_key_button = ctk.CTkButton(record_hotkey_frame, text="Detect Key", command=self._start_detect_record_key, width=100)
        self.detect_record_key_button.pack(side="left", padx=5)

        mode_row = ctk.CTkFrame(record_hotkey_frame, fg_color="transparent")
        mode_row.pack(fill="x", pady=(5, 0), padx=5)
        ctk.CTkLabel(mode_row, text="Mode:").pack(side="left", padx=5)
        ctk.CTkRadioButton(mode_row, text="Toggle", variable=self.record_mode_var, value="toggle").pack(side="left", padx=5)
        ctk.CTkRadioButton(mode_row, text="Press/Hold", variable=self.record_mode_var, value="press").pack(side="left", padx=5)

        # --- Seção de Hotkey do Agente ---
        agent_hotkey_frame = ctk.CTkFrame(self, fg_color="transparent")
        agent_hotkey_frame.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(agent_hotkey_frame, text="Agent Hotkey", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))

        agent_key_display_label = ctk.CTkLabel(agent_hotkey_frame, textvariable=self.agent_key_var, font=("Segoe UI", 12, "bold"), width=120, fg_color="#393E46", text_color="#00a0ff", corner_radius=8)
        agent_key_display_label.pack(side="left", padx=5)
        self.detect_agent_key_button = ctk.CTkButton(agent_hotkey_frame, text="Detect Key", command=self._start_detect_agent_key, width=100)
        self.detect_agent_key_button.pack(side="left", padx=5)

        agent_model_row = ctk.CTkFrame(agent_hotkey_frame, fg_color="transparent")
        agent_model_row.pack(fill="x", pady=(5, 0), padx=5)
        ctk.CTkLabel(agent_model_row, text="Agent Model:").pack(side="left", padx=5)
        # Certifique-se de que core_instance.gemini_model_options está disponível
        model_options = self.core_instance.gemini_model_options if hasattr(self.core_instance, 'gemini_model_options') else ["gemini-2.0-flash-001"]
        ctk.CTkOptionMenu(agent_model_row, variable=self.agent_model_var, values=model_options).pack(side="left", fill="x", expand=True, padx=5)

        agent_auto_paste_row = ctk.CTkFrame(agent_hotkey_frame, fg_color="transparent")
        agent_auto_paste_row.pack(fill="x", pady=(5, 0), padx=5)
        ctk.CTkSwitch(agent_auto_paste_row, text="Auto-paste Agent Response", variable=self.agent_auto_paste_var).pack(anchor="w", padx=5)

        # --- Seção de Recarregamento Automático de Hotkeys ---
        auto_reregister_frame = ctk.CTkFrame(self, fg_color="transparent")
        auto_reregister_frame.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(auto_reregister_frame, text="Auto Hotkey Re-registration", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))
        ctk.CTkSwitch(auto_reregister_frame, text="Enable Auto Re-registration", variable=self.auto_reregister_var).pack(anchor="w", padx=5)

    def _start_detect_record_key(self):
        """Inicia a thread de detecção da hotkey de gravação."""
        self.detect_record_key_button.configure(text="PRESS KEY...", state="disabled")
        threading.Thread(target=self._detect_key_task, args=(self.record_key_var, self.detect_record_key_button, False), daemon=True).start()

    def _start_detect_agent_key(self):
        """Inicia a thread de detecção da hotkey do agente."""
        self.detect_agent_key_button.configure(text="PRESS KEY...", state="disabled")
        threading.Thread(target=self._detect_key_task, args=(self.agent_key_var, self.detect_agent_key_button, True), daemon=True).start()

    def _detect_key_task(self, key_var: ctk.StringVar, button: ctk.CTkButton, is_agent_key: bool):
        """Tarefa interna para detectar a tecla pressionada."""
        detected_key_str = "ERRO"
        new_key_temp = None
        logging.info(f"Iniciando detecção de tecla (agente: {is_agent_key})...")

        try:
            # Importar KeyboardHotkeyManager aqui para evitar circular import se settings_ui for importado primeiro
            from keyboard_hotkey_manager import KeyboardHotkeyManager
            
            if self.core_instance:
                logging.info("Desativando hotkeys globais para detecção...")
                with self.core_instance.keyboard_lock:
                    self.core_instance._cleanup_hotkeys()
                    time.sleep(0.3) # Pequeno delay para garantir limpeza

            temp_manager = KeyboardHotkeyManager()
            detected_key = temp_manager.detect_key()
            temp_manager.stop()

            if detected_key:
                if len(detected_key) > 0:
                    if is_agent_key and detected_key.lower() == self.record_key_var.get().lower():
                        logging.warning(f"Agent key cannot be the same as record key: {detected_key}")
                        detected_key_str = "SAME AS RECORD"
                    else:
                        new_key_temp = detected_key.lower()
                        detected_key_str = new_key_temp.upper()
                        logging.info(f"Tecla detectada: {detected_key_str}")
                else:
                    logging.warning("Empty key combination detected.")
                    detected_key_str = "INVALID KEY"
            else:
                detected_key_str = "DETECTION FAILED"

        except Exception as e:
            logging.error(f"Error during key detection: {e}", exc_info=True)
            detected_key_str = "ERROR"
        finally:
            # Atualizar UI na thread principal do Tkinter
            if button.winfo_exists(): # Verifica se o widget ainda existe
                button.master.after(0, lambda: (
                    key_var.set(detected_key_str),
                    button.configure(text="Detect Key", state="normal"),
                    self.set_dirty_callback() # Marcar como dirty após a detecção
                ))
            
            # Reativar hotkeys originais
            if self.core_instance:
                logging.info("Reativando hotkeys originais após detecção...")
                self.core_instance.register_hotkeys() # Isso irá re-registrar com as configurações atuais

    def get_settings(self) -> Dict[str, Any]:
        settings = {
            "record_key": self.record_key_var.get().lower(),
            "record_mode": self.record_mode_var.get(),
            "agent_key": self.agent_key_var.get().lower(),
            "gemini_agent_model": self.agent_model_var.get(),
            "agent_auto_paste": self.agent_auto_paste_var.get(),
            "auto_reregister_hotkeys": self.auto_reregister_var.get(),
        }
        return settings

class AdvancedSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, initial_config: Dict[str, Any], set_dirty_callback: Callable):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="both", expand=True, padx=5, pady=5)

        self.initial_config = initial_config
        self.set_dirty_callback = set_dirty_callback

        # Variáveis
        self.batch_size_var = ctk.StringVar(value=str(initial_config.get("batch_size", 16)))
        self.gpu_index_var = ctk.StringVar(value=str(initial_config.get("gpu_index", -1)))
        self.min_record_duration_var = ctk.StringVar(value=str(initial_config.get("min_record_duration", 0.5)))

        # Trace changes to variables
        self.batch_size_var.trace_add("write", self.set_dirty_callback)
        self.gpu_index_var.trace_add("write", self.set_dirty_callback)
        self.min_record_duration_var.trace_add("write", self.set_dirty_callback)

        # --- Seção de Configurações de GPU/Processamento ---
        gpu_settings_frame = ctk.CTkFrame(self, fg_color="transparent")
        gpu_settings_frame.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(gpu_settings_frame, text="GPU/Processing Settings", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))

        batch_row = ctk.CTkFrame(gpu_settings_frame, fg_color="transparent")
        batch_row.pack(fill="x", padx=5, pady=(5, 0))
        ctk.CTkLabel(batch_row, text="Batch Size:").pack(side="left", padx=5)
        ctk.CTkEntry(batch_row, textvariable=self.batch_size_var).pack(side="left", fill="x", expand=True, padx=5)

        gpu_index_row = ctk.CTkFrame(gpu_settings_frame, fg_color="transparent")
        gpu_index_row.pack(fill="x", padx=5, pady=(5, 0))
        ctk.CTkLabel(gpu_index_row, text="GPU Index:").pack(side="left", padx=5)
        ctk.CTkEntry(gpu_index_row, textvariable=self.gpu_index_var).pack(side="left", fill="x", expand=True, padx=5)
        ctk.CTkLabel(gpu_settings_frame, text="(-1 for automatic selection, 0 for first GPU, etc.)", font=("Segoe UI", 10)).pack(anchor="w", padx=10, pady=(0, 5))

        # --- Seção de Duração Mínima de Gravação ---
        min_duration_frame = ctk.CTkFrame(self, fg_color="transparent")
        min_duration_frame.pack(fill="x", pady=(0, 10), padx=5)
        ctk.CTkLabel(min_duration_frame, text="Minimum Recording Duration", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 5))

        min_duration_row = ctk.CTkFrame(min_duration_frame, fg_color="transparent")
        min_duration_row.pack(fill="x", padx=5, pady=(5, 0))
        ctk.CTkLabel(min_duration_row, text="Minimum Duration (seconds):").pack(side="left", padx=5)
        ctk.CTkEntry(min_duration_row, textvariable=self.min_record_duration_var).pack(side="left", fill="x", expand=True, padx=5)

    def get_settings(self) -> Dict[str, Any]:
        settings = {}
        # Import DEFAULT_CONFIG and logging from whisper_tkinter or define a placeholder
        try:
            from whisper_tkinter import DEFAULT_CONFIG
        except ImportError:
            logging.warning("Could not import DEFAULT_CONFIG from whisper_tkinter in AdvancedSettingsFrame. Using a placeholder.")
            DEFAULT_CONFIG = {"batch_size": 16, "gpu_index": -1, "min_record_duration": 0.5} # Minimal placeholder

        try:
            settings["batch_size"] = int(self.batch_size_var.get())
            if settings["batch_size"] <= 0:
                raise ValueError("Batch size must be positive.")
        except ValueError:
            settings["batch_size"] = DEFAULT_CONFIG["batch_size"]
            logging.warning(f"Invalid batch size value: {self.batch_size_var.get()}. Using default.")

        try:
            settings["gpu_index"] = int(self.gpu_index_var.get())
        except ValueError:
            settings["gpu_index"] = DEFAULT_CONFIG["gpu_index"]
            logging.warning(f"Invalid GPU index value: {self.gpu_index_var.get()}. Using default.")

        try:
            settings["min_record_duration"] = float(self.min_record_duration_var.get())
            if settings["min_record_duration"] < 0.1: # Mínimo razoável
                raise ValueError("Minimum record duration must be at least 0.1 seconds.")
        except ValueError:
            settings["min_record_duration"] = DEFAULT_CONFIG["min_record_duration"]
            logging.warning(f"Invalid minimum record duration value: {self.min_record_duration_var.get()}. Using default.")
        
        return settings

# Import DEFAULT_CONFIG from whisper_tkinter for default values
try:
    from whisper_tkinter import DEFAULT_CONFIG, SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI
except ImportError:
    logging.warning("Could not import DEFAULT_CONFIG from whisper_tkinter. Using a placeholder.")
    DEFAULT_CONFIG = {
        "text_correction_enabled": False,
        "text_correction_service": "none",
        "openrouter_api_key": "",
        "openrouter_model": "deepseek/deepseek-chat-v3-0324:free",
        "gemini_api_key": "",
        "gemini_model": "gemini-2.0-flash-001",
        "gemini_prompt": "You are a speech-to-text correction specialist. Your task is to refine the following transcribed speech. ...",
        "gemini_mode": "correction",
        "gemini_general_prompt": "Based on the following text, generate a short response: {text}",
        "gemini_model_options": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-pro"]
    }
    SERVICE_NONE = "none"
    SERVICE_OPENROUTER = "openrouter"
    SERVICE_GEMINI = "gemini"


class CorrectionSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, initial_config: Dict[str, Any], set_dirty_callback: Callable):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="both", expand=True, padx=5, pady=5)

        self.initial_config = initial_config
        self.set_dirty_callback = set_dirty_callback

        # Variables
        self.enabled_var = ctk.BooleanVar(value=initial_config.get("text_correction_enabled", DEFAULT_CONFIG["text_correction_enabled"]))
        self.service_var = ctk.StringVar(value=initial_config.get("text_correction_service", DEFAULT_CONFIG["text_correction_service"]))
        self.openrouter_api_key_var = ctk.StringVar(value=initial_config.get("openrouter_api_key", DEFAULT_CONFIG["openrouter_api_key"]))
        self.openrouter_model_var = ctk.StringVar(value=initial_config.get("openrouter_model", DEFAULT_CONFIG["openrouter_model"]))
        self.gemini_api_key_var = ctk.StringVar(value=initial_config.get("gemini_api_key", DEFAULT_CONFIG["gemini_api_key"]))
        self.gemini_model_var = ctk.StringVar(value=initial_config.get("gemini_model", DEFAULT_CONFIG["gemini_model"]))
        self.gemini_prompt_var = ctk.StringVar(value=initial_config.get("gemini_prompt", DEFAULT_CONFIG["gemini_prompt"]))
        self.gemini_mode_var = ctk.StringVar(value=initial_config.get("gemini_mode", DEFAULT_CONFIG["gemini_mode"]))
        self.gemini_general_prompt_var = ctk.StringVar(value=initial_config.get("gemini_general_prompt", DEFAULT_CONFIG["gemini_general_prompt"]))
        self.gemini_model_options = initial_config.get("gemini_model_options", DEFAULT_CONFIG["gemini_model_options"])

        # Trace changes to variables
        self.enabled_var.trace_add("write", self.set_dirty_callback)
        self.service_var.trace_add("write", self.set_dirty_callback)
        self.openrouter_api_key_var.trace_add("write", self.set_dirty_callback)
        self.openrouter_model_var.trace_add("write", self.set_dirty_callback)
        self.gemini_api_key_var.trace_add("write", self.set_dirty_callback)
        self.gemini_model_var.trace_add("write", self.set_dirty_callback)
        self.gemini_prompt_var.trace_add("write", self.set_dirty_callback)
        self.gemini_mode_var.trace_add("write", self.set_dirty_callback)
        self.gemini_general_prompt_var.trace_add("write", self.set_dirty_callback)

        # UI Elements
        ctk.CTkSwitch(self, text="Enable AI Text Correction", variable=self.enabled_var, onvalue=True, offvalue=False).pack(anchor="w", pady=(5, 10), padx=5)

        self.service_selection_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(self.service_selection_frame, text="AI Service:").pack(side="left", padx=5)
        ctk.CTkRadioButton(self.service_selection_frame, text="None", variable=self.service_var, value=SERVICE_NONE).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.service_selection_frame, text="OpenRouter", variable=self.service_var, value=SERVICE_OPENROUTER).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.service_selection_frame, text="Gemini", variable=self.service_var, value=SERVICE_GEMINI).pack(side="left", padx=5)

        self.gemini_frame = self._create_gemini_frame()
        self.openrouter_frame = self._create_openrouter_frame()

        # Bind visibility update
        self.enabled_var.trace_add("write", self._update_visibility)
        self.service_var.trace_add("write", self._update_visibility)

        # Initial visibility update
        self._update_visibility()

    def _create_gemini_frame(self):
        frame = ctk.CTkFrame(self, fg_color="transparent")

        # API Key
        api_key_row = ctk.CTkFrame(frame, fg_color="transparent")
        api_key_row.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(api_key_row, text="Gemini API Key:").pack(side="left", padx=5)
        ctk.CTkEntry(api_key_row, textvariable=self.gemini_api_key_var, show="*").pack(side="left", fill="x", expand=True, padx=5)

        # Model Selection
        model_row = ctk.CTkFrame(frame, fg_color="transparent")
        model_row.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(model_row, text="Gemini Model:").pack(side="left", padx=5)
        ctk.CTkOptionMenu(model_row, variable=self.gemini_model_var, values=self.gemini_model_options).pack(side="left", fill="x", expand=True, padx=5)

        # Editable Model List
        ctk.CTkLabel(frame, text="Editable Model List (one per line):", font=("Segoe UI", 12)).pack(anchor="w", padx=5, pady=(10, 0))
        self.gemini_models_textbox = ctk.CTkTextbox(frame, width=400, height=100, wrap="none")
        self.gemini_models_textbox.pack(fill="x", expand=True, padx=5, pady=(0, 5))
        self.gemini_models_textbox.insert("0.0", "\n".join(self.gemini_model_options))
        self.gemini_models_textbox.bind("<KeyRelease>", lambda event: self.set_dirty_callback()) # Mark dirty on change

        def restore_default_models():
            self.gemini_models_textbox.delete("1.0", "end")
            self.gemini_models_textbox.insert("0.0", "\n".join(DEFAULT_CONFIG["gemini_model_options"]))
            self.set_dirty_callback()

        ctk.CTkButton(frame, text="Restore Default List", command=restore_default_models).pack(anchor="w", padx=5, pady=(0, 10))

        # Gemini Mode Selection
        mode_row = ctk.CTkFrame(frame, fg_color="transparent")
        mode_row.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(mode_row, text="Gemini Mode:").pack(side="left", padx=5)
        ctk.CTkRadioButton(mode_row, text="Correction", variable=self.gemini_mode_var, value="correction").pack(side="left", padx=5)
        ctk.CTkRadioButton(mode_row, text="General", variable=self.gemini_mode_var, value="general").pack(side="left", padx=5)
        self.gemini_mode_var.trace_add("write", self._update_prompt_visibility) # New trace for prompt visibility

        # Correction Prompt
        self.gemini_correction_prompt_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctk.CTkLabel(self.gemini_correction_prompt_frame, text="Correction Prompt:", font=("Segoe UI", 12)).pack(anchor="w", padx=5, pady=(5, 0))
        self.gemini_correction_prompt_textbox = ctk.CTkTextbox(self.gemini_correction_prompt_frame, width=500, height=150, wrap="word")
        self.gemini_correction_prompt_textbox.pack(fill="x", expand=True, padx=5, pady=(0, 5))
        self.gemini_correction_prompt_textbox.insert("0.0", self.gemini_prompt_var.get())
        self.gemini_correction_prompt_textbox.bind("<KeyRelease>", lambda event: self.gemini_prompt_var.set(self.gemini_correction_prompt_textbox.get("1.0", "end-1c")))

        def restore_default_correction_prompt():
            self.gemini_correction_prompt_textbox.delete("1.0", "end")
            self.gemini_correction_prompt_textbox.insert("0.0", DEFAULT_CONFIG["gemini_prompt"])
            self.gemini_prompt_var.set(DEFAULT_CONFIG["gemini_prompt"])
            self.set_dirty_callback()

        ctk.CTkButton(self.gemini_correction_prompt_frame, text="Restore Default Correction Prompt", command=restore_default_correction_prompt).pack(anchor="w", padx=5, pady=(0, 10))

        # General Prompt
        self.gemini_general_prompt_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctk.CTkLabel(self.gemini_general_prompt_frame, text="General Prompt:", font=("Segoe UI", 12)).pack(anchor="w", padx=5, pady=(5, 0))
        self.gemini_general_prompt_textbox = ctk.CTkTextbox(self.gemini_general_prompt_frame, width=500, height=150, wrap="word")
        self.gemini_general_prompt_textbox.pack(fill="x", expand=True, padx=5, pady=(0, 5))
        self.gemini_general_prompt_textbox.insert("0.0", self.gemini_general_prompt_var.get())
        self.gemini_general_prompt_textbox.bind("<KeyRelease>", lambda event: self.gemini_general_prompt_var.set(self.gemini_general_prompt_textbox.get("1.0", "end-1c")))

        def load_correction_prompt_as_base():
            self.gemini_general_prompt_textbox.delete("1.0", "end")
            self.gemini_general_prompt_textbox.insert("0.0", self.gemini_prompt_var.get())
            self.gemini_general_prompt_var.set(self.gemini_prompt_var.get())
            self.set_dirty_callback()

        ctk.CTkButton(self.gemini_general_prompt_frame, text="Load Correction Prompt as Base", command=load_correction_prompt_as_base).pack(anchor="w", padx=5, pady=(0, 10))

        self._update_prompt_visibility() # Initial call for prompt visibility

        return frame

    def _create_openrouter_frame(self):
        frame = ctk.CTkFrame(self, fg_color="transparent")

        # API Key
        api_key_row = ctk.CTkFrame(frame, fg_color="transparent")
        api_key_row.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(api_key_row, text="OpenRouter API Key:").pack(side="left", padx=5)
        ctk.CTkEntry(api_key_row, textvariable=self.openrouter_api_key_var, show="*").pack(side="left", fill="x", expand=True, padx=5)

        # Model
        model_row = ctk.CTkFrame(frame, fg_color="transparent")
        model_row.pack(fill="x", pady=(5, 0))
        ctk.CTkLabel(model_row, text="OpenRouter Model:").pack(side="left", padx=5)
        ctk.CTkEntry(model_row, textvariable=self.openrouter_model_var).pack(side="left", fill="x", expand=True, padx=5)

        return frame

    def _update_visibility(self, *args):
        # Hide all frames first
        self.service_selection_frame.pack_forget()
        self.gemini_frame.pack_forget()
        self.openrouter_frame.pack_forget()

        if self.enabled_var.get():
            self.service_selection_frame.pack(fill="x", pady=(10, 0), padx=5)
            if self.service_var.get() == SERVICE_GEMINI:
                self.gemini_frame.pack(fill="both", expand=True, padx=5, pady=5)
                self._update_prompt_visibility() # Ensure correct prompt is shown
            elif self.service_var.get() == SERVICE_OPENROUTER:
                self.openrouter_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def _update_prompt_visibility(self, *args):
        # Hide both prompt frames first
        self.gemini_correction_prompt_frame.pack_forget()
        self.gemini_general_prompt_frame.pack_forget()

        if self.enabled_var.get() and self.service_var.get() == SERVICE_GEMINI:
            if self.gemini_mode_var.get() == "correction":
                self.gemini_correction_prompt_frame.pack(fill="x", padx=5, pady=(5, 0))
            elif self.gemini_mode_var.get() == "general":
                self.gemini_general_prompt_frame.pack(fill="x", padx=5, pady=(5, 0))

    def get_settings(self) -> Dict[str, Any]:
        settings = {
            "text_correction_enabled": self.enabled_var.get(),
            "text_correction_service": self.service_var.get(),
            "openrouter_api_key": self.openrouter_api_key_var.get(),
            "openrouter_model": self.openrouter_model_var.get(),
            "gemini_api_key": self.gemini_api_key_var.get(),
            "gemini_model": self.gemini_model_var.get(),
            "gemini_prompt": self.gemini_prompt_var.get(),
            "gemini_mode": self.gemini_mode_var.get(),
            "gemini_general_prompt": self.gemini_general_prompt_var.get(),
            "gemini_model_options": [line.strip() for line in self.gemini_models_textbox.get("1.0", "end-1c").split("\n") if line.strip()]
        }
        return settings