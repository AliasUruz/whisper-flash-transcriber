import tkinter as tk
import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import simpledialog # Adicionado para askstring
import logging
import threading
import time
import pystray
from PIL import Image, ImageDraw
from tkinter import simpledialog # Adicionado para askstring

# Importar constantes de configuração
from .config_manager import (
    DEFAULT_CONFIG,
    SETTINGS_WINDOW_GEOMETRY,
    SERVICE_NONE,
    SERVICE_OPENROUTER,
    SERVICE_GEMINI,
    GEMINI_MODEL_OPTIONS_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
)

# Importar get_available_devices_for_ui (pode ser movido para um utils ou ficar aqui)
# Por enquanto, vamos assumir que está disponível globalmente ou será movido para cá.
# Para este plano, vamos movê-lo para cá.
import torch # Necessário para get_available_devices_for_ui

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
            "SAVING": ('orange', 'white'),
            "TRANSCRIBING": ('blue', 'white'),
            "ERROR_MODEL": ('black', 'red'),
            "ERROR_AUDIO": ('black', 'red'),
            "ERROR_TRANSCRIPTION": ('black', 'red'),
            "ERROR_SETTINGS": ('black', 'red'),
        }
        self.DEFAULT_ICON_COLOR = ('black', 'white')

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

    def update_tray_icon(self, state):
        # Logic moved from global, adjusted to use self.tray_icon and self.core_instance_ref
        if self.tray_icon:
            color1, color2 = self.ICON_COLORS.get(state, self.DEFAULT_ICON_COLOR)
            icon_image = self.create_image(64, 64, color1, color2)
            self.tray_icon.icon = icon_image
            tooltip = f"Whisper Recorder ({state})"
            if state == "IDLE" and self.core_instance_ref:
                tooltip += f" - Record: {self.core_instance_ref.record_key.upper()} - Agent: {self.core_instance_ref.agent_key.upper()}"
            elif state.startswith("ERROR") and self.core_instance_ref:
                tooltip += f" - Check Logs/Settings"
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
                sound_enabled_var = ctk.BooleanVar(value=self.config_manager.get("sound_enabled"))
                sound_frequency_var = ctk.StringVar(value=str(self.config_manager.get("sound_frequency")))
                sound_duration_var = ctk.StringVar(value=str(self.config_manager.get("sound_duration")))
                sound_volume_var = ctk.DoubleVar(value=self.config_manager.get("sound_volume"))
                text_correction_enabled_var = ctk.BooleanVar(value=self.config_manager.get("text_correction_enabled"))
                text_correction_service_var = ctk.StringVar(value=self.config_manager.get("text_correction_service"))
                openrouter_api_key_var = ctk.StringVar(value=self.config_manager.get("openrouter_api_key"))
                openrouter_model_var = ctk.StringVar(value=self.config_manager.get("openrouter_model"))
                gemini_api_key_var = ctk.StringVar(value=self.config_manager.get("gemini_api_key"))
                gemini_model_var = ctk.StringVar(value=self.config_manager.get("gemini_model"))
                batch_size_var = ctk.StringVar(value=str(self.config_manager.get("batch_size")))
                use_vad_var = ctk.BooleanVar(value=self.config_manager.get("use_vad"))
                vad_threshold_var = ctk.DoubleVar(value=self.config_manager.get("vad_threshold"))
                vad_silence_duration_var = ctk.DoubleVar(value=self.config_manager.get("vad_silence_duration"))
                save_audio_var = ctk.BooleanVar(value=self.config_manager.get("save_audio_for_debug"))
                display_transcripts_var = ctk.BooleanVar(value=self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY))
                gemini_prompt_correction_var = ctk.StringVar(value=self.config_manager.get("gemini_prompt"))

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
                    sound_freq_to_apply = int(sound_frequency_var.get())
                    sound_duration_to_apply = float(sound_duration_var.get())
                    sound_volume_to_apply = float(sound_volume_var.get())
                    text_correction_enabled_to_apply = text_correction_enabled_var.get()
                    text_correction_service_to_apply = text_correction_service_var.get()
                    openrouter_api_key_to_apply = openrouter_api_key_var.get()
                    openrouter_model_to_apply = openrouter_model_var.get()
                    gemini_api_key_to_apply = gemini_api_key_var.get()
                    gemini_model_to_apply = gemini_model_var.get()
                    gemini_prompt_correction_to_apply = gemini_prompt_correction_textbox.get("1.0", "end-1c")
                    agentico_prompt_to_apply = agentico_prompt_textbox.get("1.0", "end-1c")
                    batch_size_to_apply = int(batch_size_var.get())
                    min_transcription_duration_to_apply = float(min_transcription_duration_var.get()) # Coleta o valor
                    use_vad_to_apply = use_vad_var.get()
                    vad_threshold_to_apply = float(vad_threshold_var.get())
                    vad_silence_duration_to_apply = float(vad_silence_duration_var.get())
                    save_audio_for_debug_to_apply = save_audio_var.get()
                    display_transcripts_to_apply = display_transcripts_var.get()

                    # Logic for converting UI to GPU index
                    selected_device_str = gpu_selection_var.get()
                    gpu_index_to_apply = -1 # Default to "Auto-select"
                    if "Force CPU" in selected_device_str:
                        gpu_index_to_apply = -1 # We use -1 for forced CPU
                    elif selected_device_str.startswith("GPU"):
                        try:
                            gpu_index_to_apply = int(selected_device_str.split(":")[0].replace("GPU", "").strip())
                        except (ValueError, IndexError):
                            logging.error(f"Could not parse GPU index from string: '{selected_device_str}'. Using auto-select.")
                            gpu_index_to_apply = -1

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
                        new_gemini_prompt=gemini_prompt_correction_to_apply,
                        new_prompt_agentico=agentico_prompt_to_apply,
                        new_agent_model=model_to_apply,
                        new_gemini_model_options=new_models_list,
                        new_batch_size=batch_size_to_apply,
                        new_gpu_index=gpu_index_to_apply,
                        new_hotkey_stability_service_enabled=hotkey_stability_service_enabled_to_apply, # Nova configuração unificada
                        new_min_transcription_duration=min_transcription_duration_to_apply,
                        new_save_audio_for_debug=save_audio_for_debug_to_apply,
                        new_use_vad=use_vad_to_apply,
                        new_vad_threshold=vad_threshold_to_apply,
                        new_vad_silence_duration=vad_silence_duration_to_apply,
                        new_display_transcripts_in_terminal=display_transcripts_to_apply
                    )
                    self._close_settings_window() # Call class method

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
                
                def detect_key_task_internal(key_var):
                    key_var.set("PRESS KEY...")
                    settings_win.update_idletasks()
                    # Pass the callback to AppCore to update the StringVar
                    self.core_instance_ref.set_key_detection_callback(lambda key: key_var.set(key))
                    self.core_instance_ref.start_key_detection_thread()

                detect_key_button = ctk.CTkButton(key_frame, text="Detect Key", command=lambda: detect_key_task_internal(detected_key_var))
                detect_key_button.pack(side="left", padx=5)

                # Agent Hotkey (Moved here)
                agent_key_frame = ctk.CTkFrame(general_frame)
                agent_key_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(agent_key_frame, text="Agent Hotkey:").pack(side="left", padx=(5, 10))
                agent_key_display = ctk.CTkLabel(agent_key_frame, textvariable=agent_key_var, fg_color="gray20", corner_radius=5, width=120)
                agent_key_display.pack(side="left", padx=5)
                detect_agent_key_button = ctk.CTkButton(agent_key_frame, text="Detect Key", command=lambda: detect_key_task_internal(agent_key_var))
                detect_agent_key_button.pack(side="left", padx=5)

                # Recording Mode
                mode_frame = ctk.CTkFrame(general_frame)
                mode_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(mode_frame, text="Recording Mode:").pack(side="left", padx=(5, 10))
                ctk.CTkRadioButton(mode_frame, text="Toggle", variable=mode_var, value="toggle").pack(side="left", padx=5)
                ctk.CTkRadioButton(mode_frame, text="Hold", variable=mode_var, value="hold").pack(side="left", padx=5)

                # Auto-Paste
                paste_frame = ctk.CTkFrame(general_frame)
                paste_frame.pack(fill="x", pady=5)
                ctk.CTkSwitch(paste_frame, text="Auto-Paste", variable=auto_paste_var).pack(side="left", padx=5)

                # Hotkey Stability Service
                stability_service_frame = ctk.CTkFrame(general_frame)
                stability_service_frame.pack(fill="x", pady=5)
                ctk.CTkSwitch(stability_service_frame, text="Enable Hotkey Stability Service", variable=hotkey_stability_service_enabled_var).pack(side="left", padx=5)

                # --- Sound Settings Section ---
                sound_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                sound_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(sound_frame, text="Sound Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")
                
                sound_enabled_frame = ctk.CTkFrame(sound_frame)
                sound_enabled_frame.pack(fill="x", pady=5)
                ctk.CTkSwitch(sound_enabled_frame, text="Enable Sounds", variable=sound_enabled_var).pack(side="left", padx=5)

                sound_details_frame = ctk.CTkFrame(sound_frame)
                sound_details_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(sound_details_frame, text="Frequency (Hz):").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(sound_details_frame, textvariable=sound_frequency_var, width=60).pack(side="left", padx=5)
                ctk.CTkLabel(sound_details_frame, text="Duration (s):").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(sound_details_frame, textvariable=sound_duration_var, width=60).pack(side="left", padx=5)
                ctk.CTkLabel(sound_details_frame, text="Volume:").pack(side="left", padx=(5, 10))
                ctk.CTkSlider(sound_details_frame, from_=0.0, to=1.0, variable=sound_volume_var).pack(side="left", padx=5, fill="x", expand=True)

                # --- Text Correction (AI Services) Section ---
                ai_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                ai_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(ai_frame, text="Text Correction (AI Services)", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                text_correction_frame = ctk.CTkFrame(ai_frame)
                text_correction_frame.pack(fill="x", pady=5)
                ctk.CTkSwitch(text_correction_frame, text="Enable Text Correction", variable=text_correction_enabled_var).pack(side="left", padx=5)

                service_frame = ctk.CTkFrame(ai_frame)
                service_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(service_frame, text="Service:").pack(side="left", padx=(5, 10))
                ctk.CTkOptionMenu(service_frame, variable=text_correction_service_var, values=[SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI]).pack(side="left", padx=5)

                # --- OpenRouter Settings ---
                openrouter_frame = ctk.CTkFrame(ai_frame)
                openrouter_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(openrouter_frame, text="OpenRouter API Key:").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(openrouter_frame, textvariable=openrouter_api_key_var, show="*", width=250).pack(side="left", padx=5)
                ctk.CTkLabel(openrouter_frame, text="OpenRouter Model:").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(openrouter_frame, textvariable=openrouter_model_var, width=200).pack(side="left", padx=5)

                # --- Gemini Settings ---
                gemini_frame = ctk.CTkFrame(ai_frame)
                gemini_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_frame, text="Gemini API Key:").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(gemini_frame, textvariable=gemini_api_key_var, show="*", width=250).pack(side="left", padx=5)
                ctk.CTkLabel(gemini_frame, text="Gemini Model:").pack(side="left", padx=(5, 10))
                ctk.CTkOptionMenu(gemini_frame, variable=gemini_model_var, values=self.config_manager.get("gemini_model_options", [])).pack(side="left", padx=5)
                
                # --- Gemini Prompt ---
                gemini_prompt_frame = ctk.CTkFrame(ai_frame)
                gemini_prompt_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(gemini_prompt_frame, text="Gemini Correction Prompt:").pack(anchor="w", pady=(5,0))
                gemini_prompt_correction_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=100, wrap="word")
                gemini_prompt_correction_textbox.pack(fill="x", expand=True, pady=5)
                gemini_prompt_correction_textbox.insert("1.0", self.config_manager.get("gemini_prompt"))

                ctk.CTkLabel(gemini_prompt_frame, text="Prompt do Modo Agêntico:").pack(anchor="w", pady=(5,0))
                agentico_prompt_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                agentico_prompt_textbox.pack(fill="x", expand=True, pady=5)
                agentico_prompt_textbox.insert("1.0", self.config_manager.get("prompt_agentico"))

                ctk.CTkLabel(gemini_prompt_frame, text="Gemini Models (one per line):").pack(anchor="w", pady=(5,0))
                gemini_models_textbox = ctk.CTkTextbox(gemini_prompt_frame, height=60, wrap="word")
                gemini_models_textbox.pack(fill="x", expand=True, pady=5)
                gemini_models_textbox.insert("1.0", "\n".join(self.config_manager.get("gemini_model_options", [])))

                # --- Transcription Settings (Advanced) Section ---
                transcription_frame = ctk.CTkFrame(scrollable_frame, fg_color="transparent")
                transcription_frame.pack(fill="x", padx=10, pady=5)
                ctk.CTkLabel(transcription_frame, text="Transcription Settings (Advanced)", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")

                device_frame = ctk.CTkFrame(transcription_frame)
                device_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(device_frame, text="Processing Device:").pack(side="left", padx=(5, 10))
                ctk.CTkOptionMenu(device_frame, variable=gpu_selection_var, values=available_devices).pack(side="left", padx=5)

                batch_size_frame = ctk.CTkFrame(transcription_frame)
                batch_size_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(batch_size_frame, text="Batch Size:").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(batch_size_frame, textvariable=batch_size_var, width=60).pack(side="left", padx=5)
    
                # New: Ignore Transcriptions Shorter Than
                min_transcription_duration_frame = ctk.CTkFrame(transcription_frame)
                min_transcription_duration_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(min_transcription_duration_frame, text="Ignore Transcriptions Shorter Than (sec):").pack(side="left", padx=(5, 10))
                min_transcription_duration_entry = ctk.CTkEntry(min_transcription_duration_frame, textvariable=min_transcription_duration_var, width=80)
                min_transcription_duration_entry.pack(side="left", padx=5)

                vad_enable_frame = ctk.CTkFrame(transcription_frame)
                vad_enable_frame.pack(fill="x", pady=5)
                ctk.CTkCheckBox(vad_enable_frame, text="Use VAD", variable=use_vad_var).pack(side="left", padx=5)

                vad_params_frame = ctk.CTkFrame(transcription_frame)
                vad_params_frame.pack(fill="x", pady=5)
                ctk.CTkLabel(vad_params_frame, text="VAD Threshold:").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(vad_params_frame, textvariable=vad_threshold_var, width=60).pack(side="left", padx=5)
                ctk.CTkLabel(vad_params_frame, text="Duração do silêncio (s):").pack(side="left", padx=(5, 10))
                ctk.CTkEntry(vad_params_frame, textvariable=vad_silence_duration_var, width=60).pack(side="left", padx=5)
                save_audio_frame = ctk.CTkFrame(transcription_frame)
                save_audio_frame.pack(fill="x", pady=5)
                ctk.CTkSwitch(save_audio_frame, text="Save Audio for Debug", variable=save_audio_var).pack(side="left", padx=5)

                display_transcripts_frame = ctk.CTkFrame(transcription_frame)
                display_transcripts_frame.pack(fill="x", pady=5)
                ctk.CTkSwitch(display_transcripts_frame, text="Display Transcript in Terminal", variable=display_transcripts_var).pack(side="left", padx=5)

                # --- Action Buttons ---
                button_frame = ctk.CTkFrame(settings_win) # Move outside scrollable_frame to keep fixed
                button_frame.pack(fill="x", padx=10, pady=(20, 10))
                
                apply_button = ctk.CTkButton(button_frame, text="Apply and Close", command=apply_settings)
                apply_button.pack(side="right", padx=5)
                
                close_button = ctk.CTkButton(button_frame, text="Cancel", command=self._close_settings_window, fg_color="gray50")
                close_button.pack(side="right", padx=5)

                force_reregister_button = ctk.CTkButton(button_frame, text="Force Hotkey Re-registration", command=self.core_instance_ref.force_reregister_hotkeys)
                force_reregister_button.pack(side="left", padx=5)

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
                enabled=(is_recording or is_idle)
            ),
            pystray.MenuItem(
                '⚙️ Settings',
                lambda: self.main_tk_root.after(0, self.run_settings_gui), # Call on main thread
                enabled=(not is_loading and not is_recording)
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
