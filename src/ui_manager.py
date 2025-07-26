import customtkinter as ctk
import tkinter.messagebox as messagebox
from tkinter import simpledialog
import logging
import threading
import time
import pystray
from PIL import Image, ImageDraw

from .utils.toast import ToastNotification
from .config_manager import (
    SETTINGS_WINDOW_GEOMETRY, SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI,
    GEMINI_MODEL_OPTIONS_CONFIG_KEY, DISPLAY_TRANSCRIPTS_KEY, DEFAULT_CONFIG,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY, GEMINI_PROMPT_CONFIG_KEY
)
from .utils.tooltip import Tooltip
import torch

def get_available_devices_for_ui():
    devices = ["Auto-select (Recommended)"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
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
        self.core_instance_ref = core_instance_ref
        self.tray_icon = None
        self.settings_window_instance = None
        self.settings_thread_running = False
        self.settings_window_lock = threading.Lock()
        self.live_window = None
        self.live_textbox = None
        self.ICON_COLORS = {
            "IDLE": ('green', 'white'), "LOADING_MODEL": ('gray', 'yellow'),
            "RECORDING": ('red', 'white'), "TRANSCRIBING": ('blue', 'white'),
            "ERROR_MODEL": ('black', 'red'), "ERROR_AUDIO": ('black', 'red'),
            "ERROR_TRANSCRIPTION": ('black', 'red'), "ERROR_SETTINGS": ('black', 'red'),
        }
        self.DEFAULT_ICON_COLOR = ('black', 'white')
        self.recording_timer_thread = None
        self.stop_recording_timer_event = threading.Event()

    def _show_live_transcription_window(self):
        if not self.config_manager.get("live_transcription_enabled"): return
        if self.live_window and self.live_window.winfo_exists(): return
        self.live_window = ctk.CTkToplevel(self.main_tk_root)
        self.live_window.overrideredirect(True)
        self.live_window.geometry("400x150+50+50")
        self.live_window.attributes("-alpha", 0.85)
        self.live_window.attributes("-topmost", True)
        self.live_textbox = ctk.CTkTextbox(self.live_window, wrap="word", activate_scrollbars=True)
        self.live_textbox.pack(fill="both", expand=True)
        self.live_textbox.insert("end", "Listening...")

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

    def update_live_transcription_threadsafe(self, text):
        self.main_tk_root.after(0, lambda: self._update_live_transcription(text))

    def create_image(self, width, height, color1, color2=None):
        image = Image.new('RGB', (width, height), color1)
        if color2:
            dc = ImageDraw.Draw(image)
            dc.rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=color2)
        return image

    def _recording_tooltip_updater(self):
        while not self.stop_recording_timer_event.is_set():
            start_time = getattr(self.core_instance_ref.audio_handler, "start_time", None)
            if start_time is None: break
            elapsed = time.time() - start_time
            self.tray_icon.title = f"Whisper Recorder (RECORDING - {self._format_elapsed(elapsed)})"
            time.sleep(1)

    def _format_elapsed(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def update_tray_icon(self, state):
        if self.tray_icon:
            color1, color2 = self.ICON_COLORS.get(state, self.DEFAULT_ICON_COLOR)
            self.tray_icon.icon = self.create_image(64, 64, color1, color2)
            tooltip = f"Whisper Recorder ({state})"
            if state == "RECORDING":
                if not self.recording_timer_thread or not self.recording_timer_thread.is_alive():
                    self.stop_recording_timer_event.clear()
                    self.recording_timer_thread = threading.Thread(target=self._recording_tooltip_updater, daemon=True, name="RecordingTooltipThread")
                    self.recording_timer_thread.start()
                start_time = getattr(self.core_instance_ref.audio_handler, "start_time", None)
                if start_time is not None:
                    tooltip = f"Whisper Recorder (RECORDING - {self._format_elapsed(time.time() - start_time)})"
            else:
                if self.recording_timer_thread and self.recording_timer_thread.is_alive():
                    self.stop_recording_timer_event.set()
                    self.recording_timer_thread.join(timeout=1)
                if state == "IDLE" and self.core_instance_ref:
                    tooltip += f" - Record: {self.core_instance_ref.record_key.upper()} - Agent: {self.core_instance_ref.agent_key.upper()}"
                elif state.startswith("ERROR"):
                    tooltip += " - Check Logs/Settings"
            self.tray_icon.title = tooltip
            self.tray_icon.update_menu()
            logging.debug(f"Tray icon updated for state: {state}")

    def _initialize_settings_vars(self):
        self.auto_paste_var = ctk.BooleanVar(value=self.config_manager.get("auto_paste"))
        self.mode_var = ctk.StringVar(value=self.config_manager.get("record_mode"))
        self.detected_key_var = ctk.StringVar(value=self.config_manager.get("record_key").upper())
        self.agent_key_var = ctk.StringVar(value=self.config_manager.get("agent_key").upper())
        self.hotkey_stability_service_enabled_var = ctk.BooleanVar(value=self.config_manager.get("hotkey_stability_service_enabled"))
        self.sound_enabled_var = ctk.BooleanVar(value=self.config_manager.get("sound_enabled"))
        self.sound_frequency_var = ctk.StringVar(value=str(self.config_manager.get("sound_frequency")))
        self.sound_duration_var = ctk.StringVar(value=str(self.config_manager.get("sound_duration")))
        self.sound_volume_var = ctk.DoubleVar(value=self.config_manager.get("sound_volume"))
        self.launch_at_startup_var = ctk.BooleanVar(value=self.config_manager.get("launch_at_startup"))
        self.text_correction_enabled_var = ctk.BooleanVar(value=self.config_manager.get("text_correction_enabled"))
        self.text_correction_service_var = ctk.StringVar(value=self.config_manager.get("text_correction_service"))
        self.openrouter_api_key_var = ctk.StringVar(value=self.config_manager.get("openrouter_api_key"))
        self.openrouter_model_var = ctk.StringVar(value=self.config_manager.get("openrouter_model"))
        self.gemini_api_key_var = ctk.StringVar(value=self.config_manager.get("gemini_api_key"))
        self.gemini_model_var = ctk.StringVar(value=self.config_manager.get("gemini_model"))
        self.agent_model_var = ctk.StringVar(value=self.config_manager.get("gemini_agent_model"))
        self.available_devices = get_available_devices_for_ui()
        current_device_selection = "Auto-select (Recommended)"
        if self.config_manager.get("gpu_index_specified"):
            gpu_index = self.config_manager.get("gpu_index")
            if gpu_index == -1:
                current_device_selection = "Force CPU"
            else:
                for dev in self.available_devices:
                    if dev.startswith(f"GPU {gpu_index}"):
                        current_device_selection = dev
                        break
        self.gpu_selection_var = ctk.StringVar(value=current_device_selection)
        self.batch_size_var = ctk.StringVar(value=str(self.config_manager.get("batch_size")))
        self.min_transcription_duration_var = ctk.DoubleVar(value=self.config_manager.get("min_transcription_duration"))
        self.use_vad_var = ctk.BooleanVar(value=self.config_manager.get("use_vad"))
        self.vad_threshold_var = ctk.DoubleVar(value=self.config_manager.get("vad_threshold"))
        self.vad_silence_duration_var = ctk.DoubleVar(value=self.config_manager.get("vad_silence_duration"))
        self.save_temp_recordings_var = ctk.BooleanVar(value=self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY))
        self.display_transcripts_var = ctk.BooleanVar(value=self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY))
        self.record_storage_mode_var = ctk.StringVar(value=self.config_manager.get("record_storage_mode", "auto"))
        self.max_memory_seconds_mode_var = ctk.StringVar(value=self.config_manager.get("max_memory_seconds_mode", "manual"))
        self.max_memory_seconds_var = ctk.DoubleVar(value=self.config_manager.get("max_memory_seconds"))
        self.live_transcription_enabled_var = ctk.BooleanVar(value=self.config_manager.get("live_transcription_enabled"))

    def _apply_settings(self):
        if self.core_instance_ref.is_any_operation_running():
            messagebox.showwarning("Apply Settings", "Cannot apply settings while an operation is in progress.", parent=self.settings_window_instance)
            return
        try:
            settings_to_apply = {
                "new_key": self.detected_key_var.get().lower() if self.detected_key_var.get() != "PRESS KEY..." else self.config_manager.get("record_key"),
                "new_mode": self.mode_var.get(),
                "new_auto_paste": self.auto_paste_var.get(),
                "new_agent_key": self.agent_key_var.get().lower() if self.agent_key_var.get() != "PRESS KEY..." else self.config_manager.get("agent_key"),
                "new_hotkey_stability_service_enabled": self.hotkey_stability_service_enabled_var.get(),
                "new_sound_enabled": self.sound_enabled_var.get(),
                "new_sound_frequency": int(self.sound_frequency_var.get()),
                "new_sound_duration": float(self.sound_duration_var.get()),
                "new_sound_volume": float(self.sound_volume_var.get()),
                "new_launch_at_startup": self.launch_at_startup_var.get(),
                "new_text_correction_enabled": self.text_correction_enabled_var.get(),
                "new_text_correction_service": self.text_correction_service_var.get(),
                "new_openrouter_api_key": self.openrouter_api_key_var.get(),
                "new_openrouter_model": self.openrouter_model_var.get(),
                "new_gemini_api_key": self.gemini_api_key_var.get(),
                "new_gemini_model": self.gemini_model_var.get(),
                "new_agent_model": self.agent_model_var.get(),
                "new_gemini_prompt": self.gemini_prompt_correction_textbox.get("1.0", "end-1c"),
                "prompt_agentico": self.agentico_prompt_textbox.get("1.0", "end-1c"),
                "new_gemini_model_options": [line.strip() for line in self.gemini_models_textbox.get("1.0", "end-1c").split('\n') if line.strip()],
                "new_batch_size": int(self.batch_size_var.get()),
                "new_min_transcription_duration": float(self.min_transcription_duration_var.get()),
                "new_use_vad": self.use_vad_var.get(),
                "new_vad_threshold": float(self.vad_threshold_var.get()),
                "new_vad_silence_duration": float(self.vad_silence_duration_var.get()),
                "new_save_temp_recordings": self.save_temp_recordings_var.get(),
                "new_display_transcripts_in_terminal": self.display_transcripts_var.get(),
                "new_record_storage_mode": self.record_storage_mode_var.get(),
                "new_max_memory_seconds_mode": self.max_memory_seconds_mode_var.get(),
                "new_max_memory_seconds": float(self.max_memory_seconds_var.get()),
                "new_live_transcription_enabled": self.live_transcription_enabled_var.get(),
            }
            selected_device_str = self.gpu_selection_var.get()
            if "Force CPU" in selected_device_str: settings_to_apply["new_gpu_index"] = -1
            elif selected_device_str.startswith("GPU"): settings_to_apply["new_gpu_index"] = int(selected_device_str.split(":")[0].replace("GPU", "").strip())
            else: settings_to_apply["new_gpu_index"] = -1 # Auto-select
            if not settings_to_apply["new_gemini_model_options"]:
                messagebox.showwarning("Invalid Value", "The model list cannot be empty.", parent=self.settings_window_instance)
                return
        except (ValueError, TypeError) as e:
            messagebox.showerror("Invalid Value", f"Please check your input values. Error: {e}", parent=self.settings_window_instance)
            return
        self.core_instance_ref.apply_settings_from_external(**settings_to_apply)
        self._close_settings_window()
        ToastNotification(self.main_tk_root, "Settings saved successfully!").show()

    def _detect_key_task(self, key_var):
        key_var.set("PRESS KEY...")
        self.settings_window_instance.update_idletasks()
        self.core_instance_ref.set_key_detection_callback(lambda key: key_var.set(key.upper()))
        self.core_instance_ref.start_key_detection_thread()

    def _create_general_tab(self, tab_view):
        general_frame = ctk.CTkScrollableFrame(tab_view, fg_color="transparent")
        general_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(general_frame, text="Hotkeys", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")
        key_frame = ctk.CTkFrame(general_frame)
        key_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(key_frame, text="Record Hotkey:").pack(side="left", padx=(5, 10))
        key_display = ctk.CTkLabel(key_frame, textvariable=self.detected_key_var, fg_color="gray20", corner_radius=5, width=120)
        key_display.pack(side="left", padx=5)
        Tooltip(key_display, "The key to start and stop recording.")
        ctk.CTkButton(key_frame, text="Detect Key", command=lambda: self._detect_key_task(self.detected_key_var)).pack(side="left", padx=5)
        agent_key_frame = ctk.CTkFrame(general_frame)
        agent_key_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(agent_key_frame, text="Agent Hotkey:").pack(side="left", padx=(5, 10))
        agent_key_display = ctk.CTkLabel(agent_key_frame, textvariable=self.agent_key_var, fg_color="gray20", corner_radius=5, width=120)
        agent_key_display.pack(side="left", padx=5)
        Tooltip(agent_key_display, "The key to trigger the agent prompt.")
        ctk.CTkButton(agent_key_frame, text="Detect Key", command=lambda: self._detect_key_task(self.agent_key_var)).pack(side="left", padx=5)
        ctk.CTkLabel(general_frame, text="Behavior", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10), anchor="w")
        mode_frame = ctk.CTkFrame(general_frame)
        mode_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(mode_frame, text="Recording Mode:").pack(side="left", padx=(5, 10))
        toggle_rb = ctk.CTkRadioButton(mode_frame, text="Toggle", variable=self.mode_var, value="toggle")
        toggle_rb.pack(side="left", padx=5)
        Tooltip(toggle_rb, "Press once to start, press again to stop.")
        hold_rb = ctk.CTkRadioButton(mode_frame, text="Hold", variable=self.mode_var, value="hold")
        hold_rb.pack(side="left", padx=5)
        Tooltip(hold_rb, "Record only while the key is held down.")
        switches_frame = ctk.CTkFrame(general_frame)
        switches_frame.pack(fill="x", pady=5)
        auto_paste_switch = ctk.CTkSwitch(switches_frame, text="Auto-Paste", variable=self.auto_paste_var)
        auto_paste_switch.pack(side="left", padx=5)
        Tooltip(auto_paste_switch, "Automatically paste the transcribed text into the last active window.")
        launch_startup_switch = ctk.CTkSwitch(switches_frame, text="Launch at Startup", variable=self.launch_at_startup_var)
        launch_startup_switch.pack(side="left", padx=5)
        Tooltip(launch_startup_switch, "Start the application automatically when Windows boots.")
        hotkey_stability_switch = ctk.CTkSwitch(switches_frame, text="Enable Hotkey Stability", variable=self.hotkey_stability_service_enabled_var)
        hotkey_stability_switch.pack(side="left", padx=5)
        Tooltip(hotkey_stability_switch, "Improves hotkey responsiveness, especially on Windows 11.")
        live_transcription_switch = ctk.CTkSwitch(switches_frame, text="Enable Live Transcription", variable=self.live_transcription_enabled_var)
        live_transcription_switch.pack(side="left", padx=5)
        Tooltip(live_transcription_switch, "Show a window with the live transcription.")
        ctk.CTkLabel(general_frame, text="Audio Feedback", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10), anchor="w")
        sound_enabled_frame = ctk.CTkFrame(general_frame)
        sound_enabled_frame.pack(fill="x", pady=5)
        sound_switch = ctk.CTkSwitch(sound_enabled_frame, text="Enable Sounds", variable=self.sound_enabled_var)
        sound_switch.pack(side="left", padx=5)
        Tooltip(sound_switch, "Play a sound when recording starts and stops.")
        sound_details_frame = ctk.CTkFrame(general_frame)
        sound_details_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(sound_details_frame, text="Frequency (Hz):").pack(side="left", padx=(5, 10))
        freq_entry = ctk.CTkEntry(sound_details_frame, textvariable=self.sound_frequency_var, width=60)
        freq_entry.pack(side="left", padx=5)
        Tooltip(freq_entry, "The frequency of the notification sound.")
        ctk.CTkLabel(sound_details_frame, text="Duration (s):").pack(side="left", padx=(5, 10))
        duration_entry = ctk.CTkEntry(sound_details_frame, textvariable=self.sound_duration_var, width=60)
        duration_entry.pack(side="left", padx=5)
        Tooltip(duration_entry, "The duration of the notification sound.")
        ctk.CTkLabel(sound_details_frame, text="Volume:").pack(side="left", padx=(5, 10))
        volume_slider = ctk.CTkSlider(sound_details_frame, from_=0.0, to=1.0, variable=self.sound_volume_var)
        volume_slider.pack(side="left", padx=5, fill="x", expand=True)
        Tooltip(volume_slider, "The volume of the notification sound.")

    def _create_ai_tab(self, tab_view):
        ai_frame = ctk.CTkScrollableFrame(tab_view, fg_color="transparent")
        ai_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(ai_frame, text="AI Text Correction", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")
        correction_switch = ctk.CTkSwitch(ai_frame, text="Enable Text Correction", variable=self.text_correction_enabled_var)
        correction_switch.pack(anchor="w", padx=5, pady=5)
        Tooltip(correction_switch, "Use an AI service to improve punctuation and grammar.")
        service_frame = ctk.CTkFrame(ai_frame)
        service_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(service_frame, text="Service:").pack(side="left", padx=(5, 10))
        service_menu = ctk.CTkOptionMenu(service_frame, variable=self.text_correction_service_var, values=[SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI])
        service_menu.pack(side="left", padx=5)
        Tooltip(service_menu, "The AI service to use for text correction.")
        ctk.CTkLabel(ai_frame, text="OpenRouter Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5), anchor="w")
        openrouter_key_entry = ctk.CTkEntry(ai_frame, textvariable=self.openrouter_api_key_var, placeholder_text="OpenRouter API Key", show="*")
        openrouter_key_entry.pack(fill="x", padx=5, pady=5)
        Tooltip(openrouter_key_entry, "Your API key for the OpenRouter service.")
        openrouter_model_entry = ctk.CTkEntry(ai_frame, textvariable=self.openrouter_model_var, placeholder_text="OpenRouter Model")
        openrouter_model_entry.pack(fill="x", padx=5, pady=5)
        Tooltip(openrouter_model_entry, "The model to use for OpenRouter.")
        ctk.CTkLabel(ai_frame, text="Google Gemini Settings", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5), anchor="w")
        gemini_key_entry = ctk.CTkEntry(ai_frame, textvariable=self.gemini_api_key_var, placeholder_text="Gemini API Key", show="*")
        gemini_key_entry.pack(fill="x", padx=5, pady=5)
        Tooltip(gemini_key_entry, "Your API key for the Gemini service.")
        gemini_model_frame = ctk.CTkFrame(ai_frame)
        gemini_model_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(gemini_model_frame, text="Correction Model:").pack(side="left", padx=(5,10))
        gemini_model_menu = ctk.CTkOptionMenu(gemini_model_frame, variable=self.gemini_model_var, values=self.config_manager.get("gemini_model_options", []))
        gemini_model_menu.pack(side="left", padx=5)
        Tooltip(gemini_model_menu, "The Gemini model to use for text correction.")
        gemini_agent_model_frame = ctk.CTkFrame(ai_frame)
        gemini_agent_model_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(gemini_agent_model_frame, text="Agent Model:").pack(side="left", padx=(5, 10))
        agent_model_menu = ctk.CTkOptionMenu(gemini_agent_model_frame, variable=self.agent_model_var, values=self.config_manager.get("gemini_model_options", []))
        agent_model_menu.pack(side="left", padx=5)
        Tooltip(agent_model_menu, "The Gemini model to use for agent mode.")
        ctk.CTkLabel(ai_frame, text="Gemini Correction Prompt:", anchor="w").pack(fill="x", padx=5, pady=(10,0))
        self.gemini_prompt_correction_textbox = ctk.CTkTextbox(ai_frame, height=80, wrap="word")
        self.gemini_prompt_correction_textbox.pack(fill="x", expand=True, padx=5, pady=5)
        self.gemini_prompt_correction_textbox.insert("1.0", self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY))
        Tooltip(self.gemini_prompt_correction_textbox, "The prompt to use for text correction with Gemini.")
        ctk.CTkLabel(ai_frame, text="Agent Mode Prompt:", anchor="w").pack(fill="x", padx=5, pady=(10,0))
        self.agentico_prompt_textbox = ctk.CTkTextbox(ai_frame, height=60, wrap="word")
        self.agentico_prompt_textbox.pack(fill="x", expand=True, padx=5, pady=5)
        self.agentico_prompt_textbox.insert("1.0", self.config_manager.get("prompt_agentico"))
        Tooltip(self.agentico_prompt_textbox, "The prompt to use for agent mode with Gemini.")
        ctk.CTkLabel(ai_frame, text="Available Gemini Models (one per line):", anchor="w").pack(fill="x", padx=5, pady=(10,0))
        self.gemini_models_textbox = ctk.CTkTextbox(ai_frame, height=60, wrap="word")
        self.gemini_models_textbox.pack(fill="x", expand=True, padx=5, pady=5)
        self.gemini_models_textbox.insert("1.0", "\n".join(self.config_manager.get("gemini_model_options", [])))
        Tooltip(self.gemini_models_textbox, "The list of Gemini models available in the dropdowns.")

    def _create_advanced_tab(self, tab_view):
        advanced_frame = ctk.CTkScrollableFrame(tab_view, fg_color="transparent")
        advanced_frame.pack(fill="both", expand=True, padx=10, pady=5)
        ctk.CTkLabel(advanced_frame, text="Performance", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10), anchor="w")
        device_frame = ctk.CTkFrame(advanced_frame)
        device_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(device_frame, text="Processing Device:").pack(side="left", padx=(5, 10))
        ctk.CTkOptionMenu(device_frame, variable=self.gpu_selection_var, values=self.available_devices).pack(side="left", padx=5)
        Tooltip(device_frame, "The hardware to use for transcription. 'Auto-select' is recommended.")
        batch_size_frame = ctk.CTkFrame(advanced_frame)
        batch_size_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(batch_size_frame, text="Batch Size:").pack(side="left", padx=(5, 10))
        batch_entry = ctk.CTkEntry(batch_size_frame, textvariable=self.batch_size_var, width=80)
        batch_entry.pack(side="left", padx=5)
        Tooltip(batch_entry, "The number of audio chunks to process in parallel. Higher values may improve speed on powerful GPUs.")
        ctk.CTkLabel(advanced_frame, text="Recording & VAD", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10), anchor="w")
        min_transcription_duration_frame = ctk.CTkFrame(advanced_frame)
        min_transcription_duration_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(min_transcription_duration_frame, text="Min. Recording Duration (s):").pack(side="left", padx=(5, 10))
        min_duration_entry = ctk.CTkEntry(min_transcription_duration_frame, textvariable=self.min_transcription_duration_var, width=80)
        min_duration_entry.pack(side="left", padx=5)
        Tooltip(min_duration_entry, "Recordings shorter than this will be discarded.")
        vad_enable_frame = ctk.CTkFrame(advanced_frame)
        vad_enable_frame.pack(fill="x", pady=5)
        vad_checkbox = ctk.CTkCheckBox(vad_enable_frame, text="Use VAD (Voice Activity Detection)", variable=self.use_vad_var)
        vad_checkbox.pack(side="left", padx=5)
        Tooltip(vad_checkbox, "Automatically trim silence from your recordings.")
        vad_params_frame = ctk.CTkFrame(advanced_frame)
        vad_params_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(vad_params_frame, text="VAD Threshold:").pack(side="left", padx=(5, 10))
        vad_threshold_entry = ctk.CTkEntry(vad_params_frame, textvariable=self.vad_threshold_var, width=80)
        vad_threshold_entry.pack(side="left", padx=5)
        Tooltip(vad_threshold_entry, "The sensitivity of the voice activity detection. Higher values are more sensitive.")
        ctk.CTkLabel(vad_params_frame, text="VAD Silence (s):").pack(side="left", padx=(5, 10))
        vad_silence_entry = ctk.CTkEntry(vad_params_frame, textvariable=self.vad_silence_duration_var, width=80)
        vad_silence_entry.pack(side="left", padx=5)
        Tooltip(vad_silence_entry, "The maximum duration of silence to keep in the recording.")
        ctk.CTkLabel(advanced_frame, text="Storage", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10), anchor="w")
        storage_mode_frame = ctk.CTkFrame(advanced_frame)
        storage_mode_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(storage_mode_frame, text="Record Storage Mode:").pack(side="left", padx=(5, 10))
        storage_menu = ctk.CTkOptionMenu(storage_mode_frame, variable=self.record_storage_mode_var, values=["auto", "memory", "disk"])
        storage_menu.pack(side="left", padx=5)
        Tooltip(storage_menu, "Choose where to store audio during recording. 'Auto' is recommended.")
        mem_time_frame = ctk.CTkFrame(advanced_frame)
        mem_time_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(mem_time_frame, text="Max Memory Time (s):").pack(side="left", padx=(5, 10))
        mem_time_entry = ctk.CTkEntry(mem_time_frame, textvariable=self.max_memory_seconds_var, width=80)
        mem_time_entry.pack(side="left", padx=5)
        Tooltip(mem_time_entry, "The maximum duration to hold a recording in memory in 'auto' mode.")
        mem_mode_menu = ctk.CTkOptionMenu(mem_time_frame, variable=self.max_memory_seconds_mode_var, values=["manual", "auto"], width=100)
        mem_mode_menu.pack(side="left", padx=5)
        Tooltip(mem_mode_menu, "Choose how the memory time is determined.")
        ctk.CTkLabel(advanced_frame, text="Debugging", font=ctk.CTkFont(weight="bold")).pack(pady=(15, 10), anchor="w")
        debug_switches_frame = ctk.CTkFrame(advanced_frame)
        debug_switches_frame.pack(fill="x", pady=5)
        save_temp_switch = ctk.CTkSwitch(debug_switches_frame, text="Save Temp Recordings", variable=self.save_temp_recordings_var)
        save_temp_switch.pack(side="left", padx=5)
        Tooltip(save_temp_switch, "Save the raw audio files for debugging.")
        display_transcript_switch = ctk.CTkSwitch(debug_switches_frame, text="Display Transcript in Terminal", variable=self.display_transcripts_var)
        display_transcript_switch.pack(side="left", padx=5)
        Tooltip(display_transcript_switch, "Print the final transcript to the console.")

    def run_settings_gui(self):
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
            settings_win.update_idletasks()
            window_width, window_height = 600, 750
            screen_width, screen_height = settings_win.winfo_screenwidth(), settings_win.winfo_screenheight()
            x_cordinate = int((screen_width / 2) - (window_width / 2))
            y_cordinate = int((screen_height / 2) - (window_height / 2))
            settings_win.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
            self._initialize_settings_vars()
            tab_view = ctk.CTkTabview(settings_win)
            tab_view.pack(fill="both", expand=True, padx=10, pady=10)
            general_tab = tab_view.add("General")
            ai_tab = tab_view.add("AI & Prompts")
            advanced_tab = tab_view.add("Advanced")
            self._create_general_tab(general_tab)
            self._create_ai_tab(ai_tab)
            self._create_advanced_tab(advanced_tab)
            status_bar = ctk.CTkFrame(settings_win, height=25)
            status_bar.pack(fill="x", side="bottom", padx=10, pady=(5,10))
            ctk.CTkLabel(status_bar, text=f"Status: {self.core_instance_ref.current_state}").pack(side="left", padx=10)
            button_frame = ctk.CTkFrame(settings_win)
            button_frame.pack(fill="x", side="bottom", padx=10, pady=10)
            ctk.CTkButton(button_frame, text="Apply and Close", command=self._apply_settings).pack(side="right", padx=5)
            ctk.CTkButton(button_frame, text="Cancel", command=self._close_settings_window, fg_color="gray50").pack(side="right", padx=5)
            ctk.CTkButton(button_frame, text="Restore Defaults", command=lambda: messagebox.askyesno("Restore Defaults", "Are you sure?", parent=settings_win)).pack(side="left", padx=5)
            ctk.CTkButton(button_frame, text="Force Hotkey Re-reg", command=self.core_instance_ref.force_reregister_hotkeys).pack(side="left", padx=5)
            settings_win.protocol("WM_DELETE_WINDOW", self._close_settings_window)
        except Exception as e:
            logging.error(f"Failed to create Toplevel for settings: {e}", exc_info=True)
            self.settings_thread_running = False

    def setup_tray_icon(self):
        initial_state = self.core_instance_ref.current_state
        color1, color2 = self.ICON_COLORS.get(initial_state, self.DEFAULT_ICON_COLOR)
        initial_image = self.create_image(64, 64, color1, color2)
        initial_tooltip = f"Whisper Recorder ({initial_state})"
        self.tray_icon = pystray.Icon("whisper_recorder", initial_image, initial_tooltip, menu=pystray.Menu(lambda: self.create_dynamic_menu()))
        self.core_instance_ref.set_state_update_callback(self.update_tray_icon)
        self.core_instance_ref.set_segment_callback(self.update_live_transcription_threadsafe)
        tray_thread = threading.Thread(target=lambda icon: icon.run(), args=(self.tray_icon,), daemon=True, name="PystrayThread")
        tray_thread.start()

    def _close_settings_window(self):
        with self.settings_window_lock:
            if self.settings_window_instance:
                self.settings_window_instance.destroy()
                self.settings_window_instance = None
            self.settings_thread_running = False

    def create_dynamic_menu(self):
        is_recording = self.core_instance_ref.current_state == "RECORDING"
        menu_items = [
            pystray.MenuItem('⏹️ Stop Recording' if is_recording else '▶️ Start Recording', lambda: self.core_instance_ref.toggle_recording(), default=True, enabled=lambda item: self.core_instance_ref.current_state in ['RECORDING', 'IDLE']),
            pystray.MenuItem('⚙️ Settings', lambda: self.main_tk_root.after(0, self.run_settings_gui), enabled=lambda item: self.core_instance_ref.current_state not in ['LOADING_MODEL', 'RECORDING']),
            pystray.MenuItem('Gemini Model', pystray.Menu(*[pystray.MenuItem(model, lambda icon, item: self.core_instance_ref.apply_settings_from_external(new_gemini_model=item.text), radio=True, checked=lambda item: self.config_manager.get('gemini_model') == item.text) for model in self.config_manager.get(GEMINI_MODEL_OPTIONS_CONFIG_KEY, [])])),
            pystray.MenuItem('Batch Size', pystray.Menu(
                pystray.MenuItem('Automático (VRAM)', lambda icon, item: self.core_instance_ref.update_setting('batch_size_mode', 'auto'), radio=True, checked=lambda item: self.config_manager.get('batch_size_mode') == 'auto'),
                pystray.MenuItem('Manual', lambda icon, item: self.core_instance_ref.update_setting('batch_size_mode', 'manual'), radio=True, checked=lambda item: self.config_manager.get('batch_size_mode') == 'manual'),
                pystray.MenuItem('Definir Batch Size Manual...', lambda icon, item: self.main_tk_root.after(0, self._prompt_for_manual_batch_size))
            )),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('❌ Exit', self.on_exit_app)
        ]
        return tuple(menu_items)

    def on_exit_app(self, *_):
        logging.info("Exit requested from tray icon.")
        if self.core_instance_ref: self.core_instance_ref.shutdown()
        if self.tray_icon: self.tray_icon.stop()
        self.main_tk_root.quit()

    def _prompt_for_manual_batch_size(self):
        current_manual_batch_size = self.config_manager.get("manual_batch_size", 8)
        new_batch_size_str = simpledialog.askstring("Definir Batch Size Manual", f"Insira o novo Batch Size manual (atual: {current_manual_batch_size}):", parent=self.settings_window_instance)
        if new_batch_size_str:
            try:
                new_batch_size = int(new_batch_size_str)
                if new_batch_size > 0:
                    self.core_instance_ref.update_setting('batch_size_mode', 'manual')
                    self.core_instance_ref.update_setting('manual_batch_size', new_batch_size)
                    logging.info(f"Batch Size manual definido para: {new_batch_size}")
                else:
                    messagebox.showerror("Erro de Entrada", "O Batch Size deve ser um número inteiro positivo.", parent=self.settings_window_instance)
            except ValueError:
                messagebox.showerror("Erro de Entrada", "Entrada inválida. Por favor, insira um número inteiro.", parent=self.settings_window_instance)
