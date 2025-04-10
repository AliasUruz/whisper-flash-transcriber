# -*- coding: utf-8 -*-
import sys
import os
import json
import threading
import time
# Keep tkinter for settings window, but remove ttk and messagebox from global scope
import tkinter as tk
# Import messagebox specifically when needed inside functions that use it
# import tkinter.messagebox as messagebox
import sounddevice as sd
import numpy as np
import wave
import pystray
from PIL import Image, ImageDraw
import torch
from transformers import pipeline
import pyautogui
import keyboard
import logging
from threading import RLock
import glob
import atexit
try:
    import pyperclip # Optional import
except ImportError:
    pyperclip = None

# Import OpenRouter API for text correction
try:
    from openrouter_api import OpenRouterAPI
except ImportError:
    OpenRouterAPI = None
    logging.warning("OpenRouterAPI module not found. Text correction will be disabled.")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

# --- Constantes ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "record_key": "F3",
    "record_mode": "toggle",
    "auto_paste": True,
    "min_record_duration": 0.5,
    "sound_enabled": True,
    "sound_frequency": 400,
    "sound_duration": 0.3,
    "sound_volume": 0.5,
    "reload_key": "F5",
    "openrouter_enabled": False,
    "openrouter_api_key": "",
    "openrouter_model": "deepseek/deepseek-chat-v3-0324:free"
}
HOTKEY_DEBOUNCE_INTERVAL = 0.3
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
MIN_RECORDING_DURATION_CONFIG_KEY = "min_record_duration"
# Sound configuration keys
SOUND_ENABLED_CONFIG_KEY = "sound_enabled"
SOUND_FREQUENCY_CONFIG_KEY = "sound_frequency"
SOUND_DURATION_CONFIG_KEY = "sound_duration"
SOUND_VOLUME_CONFIG_KEY = "sound_volume"
# Reload key configuration
RELOAD_KEY_CONFIG_KEY = "reload_key"
# OpenRouter API configuration
OPENROUTER_ENABLED_CONFIG_KEY = "openrouter_enabled"
OPENROUTER_API_KEY_CONFIG_KEY = "openrouter_api_key"
OPENROUTER_MODEL_CONFIG_KEY = "openrouter_model"
# Window size adjusted to fit all elements comfortably
SETTINGS_WINDOW_GEOMETRY = "380x650" # Further increased height for OpenRouter API settings
REREGISTER_INTERVAL_SECONDS = 60 # 1 minuto (ajustável aqui)

# --- Device Configuration (can stay global/module level) ---
device_str = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device_str == "cuda" else torch.float32
device_index = 0 if device_str == "cuda" else -1
logging.info(f"Using device: {'GPU (CUDA)' if device_index == 0 else 'CPU'}")

# --- Simplified States for Tray App ---
STATE_IDLE = "IDLE"
STATE_LOADING_MODEL = "LOADING_MODEL"
STATE_RECORDING = "RECORDING"
STATE_SAVING = "SAVING" # Intermediate state after recording stops
STATE_TRANSCRIBING = "TRANSCRIBING"
STATE_ERROR_MODEL = "ERROR_MODEL"
STATE_ERROR_AUDIO = "ERROR_AUDIO"
STATE_ERROR_TRANSCRIPTION = "ERROR_TRANSCRIPTION"
STATE_ERROR_SETTINGS = "ERROR_SETTINGS" # Keep for settings/hotkey issues


class WhisperCore: # Renamed from WhisperApp
    def __init__(self): # Removed root parameter
        # --- Locks ---
        self.recording_lock = RLock()
        self.transcription_lock = RLock()
        self.state_lock = RLock() # Lock for managing self.current_state
        self.keyboard_lock = RLock() # Novo lock para operações com o módulo keyboard

        # --- State Update Callback ---
        self.state_update_callback = None # Function to call when state changes (for tray icon)

        # --- Configuration ---
        self.config = {}
        self.record_key = DEFAULT_CONFIG["record_key"]
        self.record_mode = DEFAULT_CONFIG["record_mode"]
        self.auto_paste = DEFAULT_CONFIG["auto_paste"]
        self.min_record_duration = DEFAULT_CONFIG["min_record_duration"]
        # Sound configuration
        self.sound_enabled = DEFAULT_CONFIG["sound_enabled"]
        self.sound_frequency = DEFAULT_CONFIG["sound_frequency"]
        self.sound_duration = DEFAULT_CONFIG["sound_duration"]
        self.sound_volume = DEFAULT_CONFIG["sound_volume"]
        # OpenRouter API configuration
        self.openrouter_enabled = DEFAULT_CONFIG["openrouter_enabled"]
        self.openrouter_api_key = DEFAULT_CONFIG["openrouter_api_key"]
        self.openrouter_model = DEFAULT_CONFIG["openrouter_model"]
        self.openrouter_client = None
        self.sound_lock = RLock()  # Lock for sound playback

        # Reload key configuration
        self.reload_key = DEFAULT_CONFIG["reload_key"]

        # --- Recording State ---
        self.is_recording = False
        self.start_time = None
        self.recording_data = []
        self.audio_stream = None

        # --- Transcription State ---
        self.pipe = None
        self.full_transcription = "" # Keep track of last full transcription
        self.transcription_in_progress = False # Tracks if any transcription task is running

        # --- Hotkeys ---
        self.last_key_press_time = 0.0
        self.hotkey_press_handler = None
        self.hotkey_release_handler = None

        # --- Settings Window State ---
        self.settings_window_open = False # Keep track of settings window state

        # --- Periodic Re-registration ---
        self.reregister_timer_thread = None
        self.stop_reregister_event = threading.Event()

        # --- Application State ---
        self.current_state = STATE_LOADING_MODEL # Initial state
        self.shutting_down = False # <<< FIX: Flag to prevent double shutdown

        # --- Initialization ---
        self._load_config()
        # Initialize OpenRouter client after loading config
        self._init_openrouter_client()
        self._start_model_loading()
        # Run startup cleanup here after config is loaded but before model/hotkeys
        self._cleanup_old_audio_files_on_startup()


    # --- OpenRouter API Integration ---
    def _init_openrouter_client(self):
        """Initialize the OpenRouter API client if enabled and available."""
        if not self.openrouter_enabled or not self.openrouter_api_key:
            logging.info("OpenRouter API is disabled or no API key provided.")
            self.openrouter_client = None
            return

        if OpenRouterAPI is None:
            logging.warning("OpenRouterAPI module not available. Text correction will be disabled.")
            self.openrouter_client = None
            return

        try:
            logging.info(f"Initializing OpenRouter API client with model: {self.openrouter_model}")
            self.openrouter_client = OpenRouterAPI(
                api_key=self.openrouter_api_key,
                model_id=self.openrouter_model
            )
            logging.info("OpenRouter API client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing OpenRouter API client: {e}")
            self.openrouter_client = None

    def _correct_text_with_openrouter(self, text):
        """Correct the transcribed text using OpenRouter API."""
        if not self.openrouter_enabled or not self.openrouter_client or not text:
            return text

        try:
            logging.info("Sending text to OpenRouter API for correction...")
            corrected_text = self.openrouter_client.correct_text(text)
            logging.info("Text correction completed successfully.")
            return corrected_text
        except Exception as e:
            logging.error(f"Error correcting text with OpenRouter API: {e}")
            return text  # Return original text on error

    # --- Configuration ---
    def _load_config(self):
        """Loads configuration from JSON file or uses defaults."""
        # Start with a fresh copy of defaults
        cfg = DEFAULT_CONFIG.copy()
        config_source = "defaults"

        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Update the default config with values from the loaded file
                cfg.update(loaded_config)
                config_source = CONFIG_FILE
                logging.info(f"Configuration loaded from {CONFIG_FILE}.")
            except (json.JSONDecodeError, Exception) as e:
                logging.error(f"Error reading or decoding {CONFIG_FILE}: {e}. Using defaults.")
                cfg = DEFAULT_CONFIG.copy()
                config_source = "defaults (error loading file)"
        else:
            logging.info(f"{CONFIG_FILE} not found. Using defaults and creating file.")
            # No need to do anything, cfg already holds defaults

        # --- Apply and Validate ---
        self.config = cfg

        # Use .get() for safer access and apply type conversions/validation
        self.record_key = str(self.config.get("record_key", DEFAULT_CONFIG["record_key"])).lower()
        self.record_mode = str(self.config.get("record_mode", DEFAULT_CONFIG["record_mode"])).lower()
        self.auto_paste = bool(self.config.get("auto_paste", DEFAULT_CONFIG["auto_paste"]))

        # Mode validation
        if self.record_mode not in ["toggle", "press"]:
            logging.warning(f"Invalid record_mode '{self.record_mode}' in config. Falling back to '{DEFAULT_CONFIG['record_mode']}'.")
            self.record_mode = DEFAULT_CONFIG['record_mode']
            self.config['record_mode'] = self.record_mode

        # Min duration validation
        try:
            min_duration_val = float(self.config.get(MIN_RECORDING_DURATION_CONFIG_KEY, DEFAULT_CONFIG[MIN_RECORDING_DURATION_CONFIG_KEY]))
            if min_duration_val < 0: raise ValueError("Duration cannot be negative")
            self.min_record_duration = max(0.1, min_duration_val)
        except (ValueError, TypeError):
            default_min_duration = DEFAULT_CONFIG[MIN_RECORDING_DURATION_CONFIG_KEY]
            logging.warning(f"Invalid min_record_duration '{self.config.get(MIN_RECORDING_DURATION_CONFIG_KEY)}' in config. Falling back to '{default_min_duration}'.")
            self.min_record_duration = default_min_duration
            self.config[MIN_RECORDING_DURATION_CONFIG_KEY] = self.min_record_duration

        # Sound settings validation
        # Sound enabled
        try:
            self.sound_enabled = bool(self.config.get(SOUND_ENABLED_CONFIG_KEY, DEFAULT_CONFIG[SOUND_ENABLED_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.sound_enabled = DEFAULT_CONFIG[SOUND_ENABLED_CONFIG_KEY]
            self.config[SOUND_ENABLED_CONFIG_KEY] = self.sound_enabled

        # Sound frequency
        try:
            freq_val = int(self.config.get(SOUND_FREQUENCY_CONFIG_KEY, DEFAULT_CONFIG[SOUND_FREQUENCY_CONFIG_KEY]))
            if freq_val < 20 or freq_val > 20000: raise ValueError("Frequency must be between 20 and 20000 Hz")
            self.sound_frequency = freq_val
        except (ValueError, TypeError):
            self.sound_frequency = DEFAULT_CONFIG[SOUND_FREQUENCY_CONFIG_KEY]
            self.config[SOUND_FREQUENCY_CONFIG_KEY] = self.sound_frequency

        # Sound duration
        try:
            duration_val = float(self.config.get(SOUND_DURATION_CONFIG_KEY, DEFAULT_CONFIG[SOUND_DURATION_CONFIG_KEY]))
            if duration_val < 0.05 or duration_val > 2.0: raise ValueError("Duration must be between 0.05 and 2.0 seconds")
            self.sound_duration = duration_val
        except (ValueError, TypeError):
            self.sound_duration = DEFAULT_CONFIG[SOUND_DURATION_CONFIG_KEY]
            self.config[SOUND_DURATION_CONFIG_KEY] = self.sound_duration

        # Sound volume
        try:
            volume_val = float(self.config.get(SOUND_VOLUME_CONFIG_KEY, DEFAULT_CONFIG[SOUND_VOLUME_CONFIG_KEY]))
            if volume_val < 0.0 or volume_val > 1.0: raise ValueError("Volume must be between 0.0 and 1.0")
            self.sound_volume = volume_val
        except (ValueError, TypeError):
            self.sound_volume = DEFAULT_CONFIG[SOUND_VOLUME_CONFIG_KEY]
            self.config[SOUND_VOLUME_CONFIG_KEY] = self.sound_volume

        # Reload key validation
        try:
            reload_key = str(self.config.get(RELOAD_KEY_CONFIG_KEY, DEFAULT_CONFIG[RELOAD_KEY_CONFIG_KEY])).lower()
            self.reload_key = reload_key
        except (ValueError, TypeError):
            self.reload_key = DEFAULT_CONFIG[RELOAD_KEY_CONFIG_KEY]
            self.config[RELOAD_KEY_CONFIG_KEY] = self.reload_key

        # OpenRouter settings validation
        # OpenRouter enabled
        try:
            self.openrouter_enabled = bool(self.config.get(OPENROUTER_ENABLED_CONFIG_KEY, DEFAULT_CONFIG[OPENROUTER_ENABLED_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.openrouter_enabled = DEFAULT_CONFIG[OPENROUTER_ENABLED_CONFIG_KEY]
            self.config[OPENROUTER_ENABLED_CONFIG_KEY] = self.openrouter_enabled

        # OpenRouter API key
        try:
            self.openrouter_api_key = str(self.config.get(OPENROUTER_API_KEY_CONFIG_KEY, DEFAULT_CONFIG[OPENROUTER_API_KEY_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.openrouter_api_key = DEFAULT_CONFIG[OPENROUTER_API_KEY_CONFIG_KEY]
            self.config[OPENROUTER_API_KEY_CONFIG_KEY] = self.openrouter_api_key

        # OpenRouter model
        try:
            self.openrouter_model = str(self.config.get(OPENROUTER_MODEL_CONFIG_KEY, DEFAULT_CONFIG[OPENROUTER_MODEL_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.openrouter_model = DEFAULT_CONFIG[OPENROUTER_MODEL_CONFIG_KEY]
            self.config[OPENROUTER_MODEL_CONFIG_KEY] = self.openrouter_model

        logging.info(f"Config source: {config_source}. Applied: Key='{self.record_key}', Mode='{self.record_mode}', AutoPaste={self.auto_paste}, MinDuration={self.min_record_duration}s")
        logging.info(f"OpenRouter settings: Enabled={self.openrouter_enabled}, Model={self.openrouter_model}")

        # Save only if the source was defaults (file didn't exist or was invalid)
        if config_source.startswith("defaults"):
            self._save_config()

    def _save_config(self):
        """Saves the current configuration to the JSON file."""
        config_to_save = {
            "record_key": self.record_key,
            "record_mode": self.record_mode,
            "auto_paste": self.auto_paste,
            MIN_RECORDING_DURATION_CONFIG_KEY: self.min_record_duration,
            # Sound settings
            SOUND_ENABLED_CONFIG_KEY: self.sound_enabled,
            SOUND_FREQUENCY_CONFIG_KEY: self.sound_frequency,
            SOUND_DURATION_CONFIG_KEY: self.sound_duration,
            SOUND_VOLUME_CONFIG_KEY: self.sound_volume,
            # Reload key
            RELOAD_KEY_CONFIG_KEY: self.reload_key,
            # OpenRouter API settings
            OPENROUTER_ENABLED_CONFIG_KEY: self.openrouter_enabled,
            OPENROUTER_API_KEY_CONFIG_KEY: self.openrouter_api_key,
            OPENROUTER_MODEL_CONFIG_KEY: self.openrouter_model
        }
        self.config = config_to_save # Update in-memory config as well
        try:
            with open(CONFIG_FILE, "w", encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
            logging.info(f"Configuration saved to {CONFIG_FILE}")
            self._log_status(f"Config saved. Hotkey: {self.record_key.upper()} ({self.record_mode})")
        except Exception as e:
            logging.error(f"Error saving configuration to {CONFIG_FILE}: {e}")
            self._log_status(f"Error saving config: {e}", error=True)

    # --- Sound Generation and Playback ---
    def _play_sound(self, frequency=None, duration=None, volume=None, is_start=True):
        """Generates and plays a tone with the specified parameters.

        Args:
            frequency: Tone frequency in Hz. If None, uses the configured value.
            duration: Tone duration in seconds. If None, uses the configured value.
            volume: Volume level (0.0 to 1.0). If None, uses the configured value.
            is_start: If True, plays the start tone, otherwise plays the stop tone.
        """
        if not self.sound_enabled:
            logging.debug("Sound playback skipped (disabled in settings)")
            return

        # Use configured values if not specified
        freq = frequency if frequency is not None else self.sound_frequency
        dur = duration if duration is not None else self.sound_duration
        vol = volume if volume is not None else self.sound_volume

        # Adjust frequency for start/stop differentiation
        if not is_start:
            freq = int(freq * 0.8)  # Lower pitch for stop tone

        try:
            with self.sound_lock:  # Prevent concurrent sound playback
                # Generate a sine wave
                t = np.linspace(0, dur, int(dur * AUDIO_SAMPLE_RATE), False)
                tone = np.sin(2 * np.pi * freq * t) * vol

                # Apply fade in/out to avoid clicks
                fade_samples = int(0.01 * AUDIO_SAMPLE_RATE)  # 10ms fade
                if fade_samples * 2 < len(tone):
                    fade_in = np.linspace(0, 1, fade_samples)
                    fade_out = np.linspace(1, 0, fade_samples)
                    tone[:fade_samples] *= fade_in
                    tone[-fade_samples:] *= fade_out

                # Play the tone
                sd.play(tone, AUDIO_SAMPLE_RATE)
                sd.wait()  # Wait for playback to finish

                logging.debug(f"Played {'start' if is_start else 'stop'} tone: {freq}Hz, {dur}s, vol={vol}")
        except Exception as e:
            logging.error(f"Error playing sound: {e}")
            # Don't raise - sound playback is non-critical

    # --- State and Status Handling ---
    def set_state_update_callback(self, callback):
        """Sets the function to call when the internal state changes (e.g., update tray icon)."""
        self.state_update_callback = callback
        # Call immediately with current state if callback is set after init
        if callback:
             callback(self.current_state)

    def _set_state(self, new_state):
        """Sets the internal state and calls the update callback if state changes."""
        with self.state_lock:
            if self.current_state == new_state:
                logging.debug(f"State already {new_state}, not changing.")
                return # No change, do nothing

            self.current_state = new_state
            logging.info(f"State changed to: {new_state}")

            # Call the callback outside the lock to prevent deadlocks if callback interacts back
            callback_to_call = self.state_update_callback
            current_state_for_callback = new_state

        if callback_to_call:
            try:
                # pystray callbacks should be safe to call directly
                callback_to_call(current_state_for_callback)
            except Exception as e:
                logging.error(f"Error calling state update callback for state {current_state_for_callback}: {e}")


    def _log_status(self, text, error=False):
        """Logs status messages."""
        if error:
            logging.error(text)
        else:
            logging.info(text)
        # Tray tooltip update will be handled by the state change callback mechanism

    def _handle_transcription_result(self, text_to_display):
        """Handles the final transcription text: logs, copies, pastes."""
        logging.info("Handling final transcription result.")
        self.full_transcription = text_to_display # Store last result

        # Attempt clipboard copy using pyperclip, but make it optional
        if pyperclip:
            try:
                pyperclip.copy(text_to_display)
                logging.info("Transcription copied to clipboard using pyperclip.")
            except Exception as e:
                # Catch potential errors during copy (e.g., clipboard access issues)
                logging.error(f"Error copying to clipboard with pyperclip: {e}")
                self._log_status("Transcription complete. Error during clipboard copy.", error=True)
                # Continue with paste if enabled
        else:
            logging.warning("pyperclip not available. Skipping explicit clipboard copy.")

        try:
            if self.auto_paste:
                self._do_paste() # Try direct paste
            else:
                self._log_status("Transcription complete. Auto-paste disabled.")

        except Exception as e:
            logging.error(f"Error during transcription result handling (paste): {e}")
            self._log_status("Transcription complete. Error pasting.", error=True)


    def _do_paste(self):
        """Performs the paste action."""
        try:
            pyautogui.hotkey('ctrl', 'v')
            logging.info("Pasted transcription.")
            self._log_status("Transcription complete. Text pasted.") # Log only
        except Exception as e:
            logging.error(f"Error pasting: {e}")
            self._log_status("Transcription complete. Error pasting.", error=True)

    # --- Model Loading ---
    def _start_model_loading(self):
        """Starts loading the Whisper model in a background thread."""
        self._set_state(STATE_LOADING_MODEL)
        threading.Thread(target=self._load_model_task, daemon=True, name="ModelLoadThread").start()

    def _load_model_task(self):
        """Loads the Whisper model in a separate thread."""
        logging.info("Model loading thread started.")
        load_start_time = time.time()
        model_loaded_successfully = False
        error_message = "Unknown error during model load."

        try:
            logging.info("Attempting to load pipeline...")
            # Verificação inicial do CUDA
            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA device count: {torch.cuda.device_count()}")
                logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
                logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
                logging.info(f"CUDA version: {torch.version.cuda}")

            loaded_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch_dtype,
                device=device_str  # Força o uso do dispositivo especificado
            )

            # Verificação detalhada do dispositivo
            if torch.cuda.is_available():
                device = next(loaded_pipe.model.parameters()).device
                logging.info(f"Model initially loaded on: {device}")

                if device.type != 'cuda':
                    logging.error("Model not using CUDA despite availability!")
                    raise RuntimeError("Failed to load model on GPU")

                # Log detalhado da GPU
                if device.type == 'cuda':
                    logging.info(f"GPU details: {torch.cuda.get_device_name(device)}")
                    logging.info(f"CUDA capability: {torch.cuda.get_device_capability(device)}")
                    logging.info(f"GPU memory - Allocated: {torch.cuda.memory_allocated(device)/1024**2:.2f}MB")
                    logging.info(f"GPU memory - Reserved: {torch.cuda.memory_reserved(device)/1024**2:.2f}MB")
            if loaded_pipe:
                self.pipe = loaded_pipe # Assign to instance var
                logging.info("Pipeline loading successful.")
                model_loaded_successfully = True
            else:
                error_message = "Pipeline function returned None."
                logging.error(error_message)

        except ImportError as e:
             error_message = f"Model dependencies missing? {e}"
             logging.error(f"Pipeline load failed: {error_message}", exc_info=True)
        except torch.cuda.OutOfMemoryError as e:
             error_message = f"CUDA Out of Memory. {e}"
             logging.error(f"Pipeline load failed: {error_message}", exc_info=True)
        except Exception as e:
            error_message = f"Failed to load model: {e}"
            logging.error(f"Pipeline load failed: {error_message}", exc_info=True)

        load_end_time = time.time()
        logging.info(f"Model loading attempt finished in {load_end_time - load_start_time:.2f}s. Success: {model_loaded_successfully}")

        if model_loaded_successfully:
            self._on_model_loaded()
        else:
            self._on_model_load_failed(error_message)

    def _on_model_loaded(self):
        """Callback executed after successful model load."""
        logging.info("Model loaded successfully.")
        self._set_state(STATE_IDLE)
        self.register_hotkeys() # Register hotkeys now that model is ready
        logging.info("Hotkeys registered.")

        # Start the periodic re-registration thread
        if self.reregister_timer_thread is None:
            self.stop_reregister_event.clear() # Ensure event is clear
            self.reregister_timer_thread = threading.Thread(
                target=self._periodic_reregister_task,
                daemon=True, # Daemon thread exits when main program exits
                name="PeriodicReregisterThread"
            )
            self.reregister_timer_thread.start()
            logging.info("Periodic hotkey re-registration thread initiated.")

    def _on_model_load_failed(self, error_msg):
         """Handles model loading failure."""
         logging.error(f"Model load failed: {error_msg}")
         self._set_state(STATE_ERROR_MODEL)
         self._log_status(f"Error: Model failed to load. {error_msg}", error=True)


    # --- Hotkeys ---
    def _debounce_logic(self, callback):
        """Prevents rapid firing of hotkey callbacks."""
        current_time = time.time()
        if current_time - self.last_key_press_time < HOTKEY_DEBOUNCE_INTERVAL:
            logging.debug(f"Debounce: Ignoring event")
            return False
        self.last_key_press_time = current_time
        logging.debug(f"Debounce: Allowing event")
        # Run the callback in a separate thread to avoid blocking the hotkey listener
        threading.Thread(target=callback, daemon=True, name="HotkeyCallbackThread").start()
        return True

    def _handle_toggle_press(self):
        logging.info("Toggle key pressed (Callback Triggered)")
        self._debounce_logic(self.toggle_recording) # Call the instance method

    def _handle_press_start(self):
         logging.info(f"'{self.record_key}' key pressed (Press Mode Callback Triggered)")
         with self.recording_lock:
             if self.is_recording: return # Already recording, do nothing on subsequent presses
         self._debounce_logic(self.start_recording) # Call the instance method

    def _handle_press_release(self):
        logging.info(f"'{self.record_key}' key released (Press Mode Callback Triggered)")
        # No debounce needed on release, just stop if recording
        # Run stop in a thread to avoid blocking listener if stop takes time
        threading.Thread(target=self.stop_recording_if_needed, daemon=True, name="StopRecordingThread").start() # Call the instance method

    def register_hotkeys(self):
        """Clears old hooks and registers new ones based on current config."""
        self._cleanup_hotkeys() # Unhook previous first
        time.sleep(0.1) # Small delay

        if not self.record_key:
            logging.error("Cannot register hotkey: record_key is empty.")
            self._set_state(STATE_ERROR_SETTINGS)
            self._log_status("Error: No record key set!", error=True)
            return

        logging.info(f"Attempting to register hotkeys: Key='{self.record_key}', Mode='{self.record_mode}'")
        try:
            with self.keyboard_lock: # Protect keyboard operations
                if self.record_mode == "toggle":
                    self.hotkey_press_handler = keyboard.add_hotkey(
                        self.record_key,
                        self._handle_toggle_press, # Use the internal handler
                        suppress=True
                    )
                    logging.info(f"Toggle hotkey registered: '{self.record_key.upper()}' (suppress=True)")
                elif self.record_mode == "press":
                    self.hotkey_press_handler = keyboard.add_hotkey(
                        self.record_key,
                        self._handle_press_start, # Use the internal handler
                        suppress=True,
                        trigger_on_release=False
                    )
                    self.hotkey_release_handler = keyboard.add_hotkey(
                        self.record_key,
                        self._handle_press_release, # Use the internal handler
                        suppress=True,
                        trigger_on_release=True
                    )
                    logging.info(f"Press/release hotkeys registered: '{self.record_key.upper()}' (suppress=True)")
                else:
                     logging.error(f"Invalid record_mode: {self.record_mode}")
                     self._set_state(STATE_ERROR_SETTINGS)
                     self._log_status(f"Invalid record mode: {self.record_mode}", error=True)
                     return

                # Register the reload hotkey
                self._register_reload_hotkey()

                # Verificação adicional para garantir que suppress=True está funcionando
                if (self.hotkey_press_handler is None or
                    (self.record_mode == "press" and self.hotkey_release_handler is None)):
                    raise RuntimeError("Failed to register hotkey with suppress=True")

            # If no exception, assume success
            self._log_status(f"Global hotkey registered: {self.record_key.upper()} (mode: {self.record_mode})")
            logging.info("Hotkey registration successful.")
            # Ensure state reflects idle state correctly after registration
            if self.current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
                 self._set_state(STATE_IDLE)

        except (ValueError, ImportError, Exception) as e:
             log_msg = f"Hotkey registration failed: {e}"
             logging.error(log_msg, exc_info=True)
             status_msg = f"Error: Hotkey registration failed."
             if isinstance(e, ValueError): status_msg = f"Error: Invalid key name '{self.record_key}'"
             if isinstance(e, ImportError): status_msg = "Error: Hotkey library failed (Permissions?)"
             self._set_state(STATE_ERROR_SETTINGS)
             self._log_status(status_msg, error=True)
             self.hotkey_press_handler = None
             self.hotkey_release_handler = None

    def _cleanup_hotkeys(self):
        """Unhooks all keyboard listeners."""
        logging.debug("Attempting to unhook keyboard listeners...")
        with self.keyboard_lock: # Protect keyboard operations
            try:
                keyboard.unhook_all() # More robust
                logging.info("Keyboard hooks removed.")
            except Exception as e:
                logging.error(f"Error during keyboard hook cleanup: {e}")
            finally:
                 self.hotkey_press_handler = None
                 self.hotkey_release_handler = None

    def _register_reload_hotkey(self):
        """Registers the reload hotkey separately."""
        try:
            with self.keyboard_lock:
                # Unregister existing reload hotkey if any
                try:
                    keyboard.unhook_key(self.reload_key)
                except Exception:
                    pass  # Ignore errors if key wasn't registered

                # Register the reload hotkey
                keyboard.on_press_key(self.reload_key, self._reload_hotkey_handler)
                logging.info(f"Reload hotkey registered: {self.reload_key}")
                return True
        except Exception as e:
            logging.error(f"Error registering reload hotkey: {e}")
            self._log_status(f"Failed to register reload hotkey: {e}", error=True)
            return False

    def _reload_hotkey_handler(self, e):
        """Handler for the reload hotkey press."""
        # Avoid processing if key is being held down
        if e.event_type == keyboard.KEY_DOWN:
            # Debounce to prevent multiple rapid calls
            current_time = time.time()
            if hasattr(self, 'last_reload_time') and current_time - self.last_reload_time < HOTKEY_DEBOUNCE_INTERVAL:
                logging.debug(f"Ignoring reload hotkey press (debounce): {current_time - self.last_reload_time:.2f}s")
                return

            self.last_reload_time = current_time
            logging.info(f"Reload hotkey pressed: {self.reload_key}")

            # Play a sound to indicate reload is happening
            if self.sound_enabled:
                threading.Thread(target=self._play_sound, kwargs={"frequency": self.sound_frequency * 1.2, "duration": 0.15, "is_start": True}, daemon=True).start()
                time.sleep(0.2)  # Small delay between sounds
                threading.Thread(target=self._play_sound, kwargs={"frequency": self.sound_frequency * 1.2, "duration": 0.15, "is_start": True}, daemon=True).start()

            # Force re-register in a separate thread to avoid blocking
            threading.Thread(target=self.force_reregister_hotkeys, daemon=True, name="ReloadHotkeyThread").start()

    def _reload_keyboard_and_suppress(self):
        """Recarrega completamente a biblioteca Keyboard e garante suppress=True."""
        global keyboard  # Garante que estamos usando o módulo global

        # Protege contra recarregamentos durante gravações
        with self.keyboard_lock: # Use the lock for the whole operation
            max_attempts = 3
            attempt = 0
            last_error = None

            # Verificação inicial do módulo keyboard
            if 'keyboard' not in globals():
                logging.debug("Módulo keyboard não encontrado em globals() - importando...")
                import keyboard

            while attempt < max_attempts:
                attempt += 1
                try:
                    logging.info(f"Tentativa {attempt}/{max_attempts} de recarregamento do keyboard...")

                    # 1. Limpa todos os hooks existentes
                    logging.debug("Removendo todos os hooks do keyboard...")
                    keyboard.unhook_all()
                    self.hotkey_press_handler = None # Clear references
                    self.hotkey_release_handler = None

                    # 2. Reinicia o módulo keyboard
                    logging.debug("Recarregando módulo keyboard...")
                    import importlib
                    import sys
                    if 'keyboard' in sys.modules:
                        keyboard = importlib.reload(sys.modules['keyboard'])
                    else:
                        import keyboard
                        logging.warning("Módulo keyboard não estava em sys.modules - novo import")

                    # 3. Verificação robusta do módulo
                    if 'keyboard' not in sys.modules:
                        raise RuntimeError("Módulo keyboard não está em sys.modules após recarregamento")
                    if not hasattr(keyboard, 'add_hotkey'):
                        raise RuntimeError("Módulo keyboard não tem função add_hotkey")

                    # 4. Pausa mais longa para estabilização
                    wait_time = 1.0 + (attempt * 0.5)  # Tempo aumentado
                    logging.debug(f"Aguardando {wait_time:.1f}s para estabilização...")
                    time.sleep(wait_time) # Release lock briefly? No, keep it for consistency

                    # 5. Verifica se o módulo está funcional
                    logging.debug("Testando funcionalidade básica do keyboard...")
                    if not callable(keyboard.add_hotkey):
                        raise RuntimeError("Função add_hotkey não é chamável")

                    # 6. Registra as hotkeys novamente (still inside lock)
                    logging.info("Registrando hotkeys após recarregamento...")
                    self.register_hotkeys() # This will call _cleanup_hotkeys again, but it's safe

                    # 7. Verificação final
                    if self.record_mode == "toggle" and not self.hotkey_press_handler:
                        raise RuntimeError("Falha ao registrar hotkey toggle")
                    elif self.record_mode == "press" and not (self.hotkey_press_handler and self.hotkey_release_handler):
                        raise RuntimeError("Falha ao registrar hotkeys press/release")

                    logging.info("Keyboard recarregado com sucesso!")
                    return True # <<< Retorna True em caso de sucesso

                except Exception as e:
                    last_error = e
                    logging.error(f"Erro durante tentativa {attempt}: {str(e)}", exc_info=True)
                    if attempt < max_attempts:
                        retry_delay = 0.5 * attempt  # Delay aumentado
                        logging.info(f"Aguardando {retry_delay:.1f}s para nova tentativa...")
                        time.sleep(retry_delay) # Release lock briefly? No.

            # Todas as tentativas falharam
            error_msg = f"Falha ao recarregar keyboard após {max_attempts} tentativas. Último erro: {last_error}"
            logging.error(error_msg)
            self._log_status("ERRO: Falha crítica no teclado - reinicie o aplicativo", error=True)
            # raise RuntimeError(error_msg) from last_error # Não levanta exceção, apenas retorna False
            return False # <<< Retorna False em caso de falha

    def _periodic_reregister_task(self):
        """Thread task that periodically re-registers hotkeys if the app is idle or transcribing."""
        logging.info("Periodic hotkey re-registration thread started.")
        while not self.stop_reregister_event.wait(REREGISTER_INTERVAL_SECONDS):
            # wait() returns True if the event was set (stop requested), False on timeout
            # So the loop continues as long as wait() returns False (timeout occurred)

            with self.state_lock:
                current_state = self.current_state # Get state under lock

            if current_state in [STATE_IDLE, STATE_TRANSCRIBING]:
                logging.info(f"Periodic check: State is {current_state}. Attempting hotkey re-registration.")
                try:
                    # Run the reload in a separate thread to avoid blocking the timer?
                    # No, the reload itself needs the keyboard lock, better to do it here.
                    success = self._reload_keyboard_and_suppress() # Captura o resultado
                    if success:
                        logging.info("Periodic hotkey re-registration attempt finished successfully.")
                    else:
                        logging.warning("Periodic hotkey re-registration attempt failed.")
                        # Set error state if reload fails consistently?
                        self._set_state(STATE_ERROR_SETTINGS) # Indicate potential issue
                except Exception as e:
                    logging.error(f"Error during periodic hotkey re-registration: {e}", exc_info=True)
                    self._set_state(STATE_ERROR_SETTINGS) # Indicate potential issue
            else:
                logging.debug(f"Periodic check: State is {current_state}. Skipping hotkey re-registration.")

        logging.info("Periodic hotkey re-registration thread stopped.")

    def force_reregister_hotkeys(self):
        """Manually attempts to re-register hotkeys if state allows. Returns True on success, False on failure."""
        with self.state_lock:
            current_state = self.current_state

        # Allow retry from any state except active recording/saving/loading
        if current_state not in [STATE_RECORDING, STATE_SAVING, STATE_LOADING_MODEL]:
            logging.info(f"Manual trigger: State is {current_state}. Attempting hotkey re-registration.")
            try:
                success = self._reload_keyboard_and_suppress() # Chama a função de recarga
                if success:
                    logging.info("Manual hotkey re-registration successful.")
                    # If successful, try to move back to IDLE state if we were in error
                    if current_state.startswith("ERROR"):
                        self._set_state(STATE_IDLE)
                    self._log_status("Recarregamento do teclado/hotkey concluído.", error=False)
                    return True # Indicate success
                else:
                    logging.error("Manual hotkey re-registration failed.")
                    self._log_status("Falha ao recarregar teclado/hotkey.", error=True)
                    # Ensure state reflects error if reload failed
                    self._set_state(STATE_ERROR_SETTINGS)
                    return False # Indicate failure

            except Exception as e:
                logging.error(f"Exception during manual hotkey re-registration: {e}", exc_info=True)
                self._log_status(f"Erro ao recarregar hotkey: {e}", error=True)
                self._set_state(STATE_ERROR_SETTINGS) # Indicate an issue
                return False # Indicate failure
        else:
            logging.warning(f"Manual trigger: Cannot re-register hotkeys. Current state is {current_state}.")
            self._log_status(f"Não é possível recarregar agora (Estado: {current_state}).", error=True)
            return False # Indicate failure (state not suitable)

    # --- Audio Recording ---
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for the sounddevice InputStream."""
        if status:
            logging.warning(f"Audio callback status: {status}")
        with self.recording_lock:
            if self.is_recording:
                self.recording_data.append(indata.copy())

    def _record_audio_task(self):
        """Manages the audio input stream in a separate thread."""
        stream = None
        try:
            logging.info("Audio recording thread started.")
            with self.recording_lock:
                if not self.is_recording: # Check flag again just before stream start
                    logging.warning("Recording flag turned off before stream start.")
                    return

            stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.audio_stream = stream # Store reference if needed elsewhere
            stream.start()
            logging.info(f"Audio stream started.")

            while True:
                with self.recording_lock:
                    should_continue = self.is_recording
                if not should_continue:
                    break
                sd.sleep(100) # Yield CPU

            logging.info("Recording flag is off. Stopping audio stream.")

        except sd.PortAudioError as e:
             logging.error(f"PortAudio error during recording: {e}", exc_info=True)
             with self.recording_lock: self.is_recording = False # Ensure state consistency
             self._set_state(STATE_ERROR_AUDIO)
             self._log_status(f"Error: Audio device issue - {e}", error=True)
        except Exception as e:
            logging.error(f"Error in audio recording thread: {e}", exc_info=True)
            with self.recording_lock: self.is_recording = False
            self._set_state(STATE_ERROR_AUDIO)
            self._log_status(f"Error: Recording failed - {e}", error=True)
        finally:
            if stream is not None:
                try:
                    if stream.active: stream.stop()
                    stream.close()
                    logging.info("Audio stream stopped and closed.")
                except Exception as e:
                     logging.error(f"Error stopping/closing audio stream: {e}")
            self.audio_stream = None
            logging.info("Audio recording thread finished.")


    # --- Recording Control ---
    def start_recording(self):
        """Starts the audio recording process."""
        with self.recording_lock:
            if self.is_recording:
                logging.warning("Start recording called but already recording.")
                return
            # Check transcription state
            with self.transcription_lock:
                 if self.transcription_in_progress:
                     logging.warning("Cannot start recording: transcription is in progress.")
                     self._log_status("Cannot record: Transcription running.", error=True)
                     return
            # Check model state
            with self.state_lock:
                if self.pipe is None or self.current_state == STATE_LOADING_MODEL:
                    logging.warning("Cannot start recording: model not ready.")
                    self._log_status("Cannot record: Model not loaded.", error=True)
                    return
                if self.current_state.startswith("ERROR"):
                    logging.warning(f"Cannot start recording: App in error state ({self.current_state}).")
                    self._log_status(f"Cannot record: App in error state ({self.current_state}).", error=True)
                    return


            # Reset state
            self.is_recording = True
            self.start_time = time.time()
            self.recording_data.clear()
            logging.info(f"Recording started at {self.start_time:.2f}")

        self._set_state(STATE_RECORDING)

        # Play start sound in a separate thread to avoid blocking
        threading.Thread(target=self._play_sound, kwargs={"is_start": True}, daemon=True, name="StartSoundThread").start()

        # Start audio capture thread
        threading.Thread(target=self._record_audio_task, daemon=True, name="AudioRecordThread").start()

    def stop_recording(self):
        """Stops recording, checks duration, and triggers saving if sufficient."""
        logging.debug("Attempting to stop recording...")
        should_save = False
        recording_duration = 0.0
        audio_data_copy = None
        current_time = time.time()

        with self.recording_lock:
            if not self.is_recording:
                logging.warning("Stop recording called but not recording.")
                return # Exit if not actually recording

            # Calculate duration accurately (simplified without pause)
            st = self.start_time
            if st is not None:
                recording_duration = current_time - st
                recording_duration = max(0, recording_duration)
                logging.debug(f"Stop: currentTime={current_time:.2f}, startTime={st:.2f}, duration={recording_duration:.2f}")
            else:
                logging.error("Stop recording: is_recording=True but start_time is None!")
                recording_duration = 0.0

            # --- Minimum Duration Check ---
            if recording_duration < self.min_record_duration:
                logging.warning(f"Recording stopped (< {self.min_record_duration}s). Discarding.")
                self.is_recording = False
                self.recording_data.clear()
                self.start_time = None
                self._set_state(STATE_IDLE)
                self._log_status(f"Recording too short (<{self.min_record_duration}s). Discarded.", error=True)
                return # --- Stop execution here ---

            # --- Duration OK - Proceed to Save ---
            logging.info(f"Recording stopped signal sent. Duration: {recording_duration:.2f}s. Proceeding.")
            self.is_recording = False

            # Play stop sound in a separate thread
            threading.Thread(target=self._play_sound, kwargs={"is_start": False}, daemon=True, name="StopSoundThread").start()

            # Copy data BEFORE clearing buffer
            try:
                valid_data = [arr for arr in self.recording_data if isinstance(arr, np.ndarray) and arr.size > 0]
                if valid_data:
                    audio_data_copy = np.concatenate(valid_data, axis=0)
                else:
                    logging.warning("No valid audio data recorded to save.")
                    self._set_state(STATE_IDLE)
                    self._log_status("No audio recorded.", error=True)
                    self.recording_data.clear() # Clear invalid data
                    self.start_time = None # Reset start time
                    return
            except Exception as e:
                 logging.error(f"Failed to prepare audio data: {e}", exc_info=True)
                 self._set_state(STATE_ERROR_AUDIO)
                 self._log_status("Error processing audio data.", error=True)
                 self.recording_data.clear() # Clear potentially corrupt data
                 self.start_time = None # Reset start time
                 return

            self.recording_data.clear() # Clear buffer AFTER successful copy/concat
            self.start_time = None # Reset start time
            should_save = True

        # --- Outside lock ---
        if should_save and audio_data_copy is not None:
            # State set in the task now
            # self._set_state(STATE_SAVING)
            # Start save task in thread
            threading.Thread(target=self._save_and_transcribe_task, args=(audio_data_copy,), daemon=True, name="SaveTranscribeThread").start()
        elif not should_save:
             pass # Already handled the "too short" case inside the lock
        else: # Should not happen if should_save is True but audio_data_copy is None
             logging.error("Logic error: should_save=True but no audio data.")
             self._set_state(STATE_IDLE)


    def stop_recording_if_needed(self):
        """Stops recording only if it's currently active (used for press/release)."""
        with self.recording_lock:
            if not self.is_recording:
                logging.debug("Stop requested (release) but not recording.")
                return
        logging.info("Proceeding with stop request (release).")
        self.stop_recording() # stop_recording handles state and duration check


    def toggle_recording(self):
        """Toggles the recording state (start/stop)."""
        logging.debug("Toggle recording requested.")
        with self.recording_lock:
            rec = self.is_recording
        with self.transcription_lock:
            transcribing = self.transcription_in_progress # Check transcription state too

        if rec:
            logging.info("Toggle: Currently recording, calling stop...")
            self.stop_recording()
        elif transcribing:
             logging.warning("Toggle: Cannot start recording, transcription in progress.")
             self._log_status("Cannot record: Transcription running.", error=True)
        else:
            # Check model/error state before starting (redundant with start_recording check, but safe)
            with self.state_lock:
                if self.pipe is None or self.current_state == STATE_LOADING_MODEL:
                    logging.warning("Toggle: Cannot start recording: model not ready.")
                    self._log_status("Cannot record: Model not loaded.", error=True)
                    return
                if self.current_state.startswith("ERROR"):
                    logging.warning(f"Toggle: Cannot start recording: App in error state ({self.current_state}).")
                    self._log_status(f"Cannot record: App in error state ({self.current_state}).", error=True)
                    return
            logging.info("Toggle: Currently stopped, calling start...")
            self.start_recording()


    # --- Audio Saving and Transcription Task ---
    def _save_and_transcribe_task(self, audio_data):
        """Saves audio data to WAV and immediately starts transcription."""
        logging.info("Save and transcribe task started.")
        self._set_state(STATE_SAVING)

        timestamp = int(time.time())
        temp_filename = f"temp_recording_{timestamp}.wav"
        final_filename = f"recording_{timestamp}.wav"
        saved_successfully = False

        try:
            # --- Save Audio ---
            logging.info(f"Saving audio to {temp_filename}")

            # Converter para formato compatível com int16 antes de salvar
            if audio_data.dtype != np.int16:
                # Check max/min values before scaling to prevent clipping/overflow
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    logging.warning(f"Audio data exceeds expected range [-1.0, 1.0] (max abs: {max_val}). Clipping may occur.")
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data_int16 = (audio_data * (2**15 - 1)).astype(np.int16)
            else:
                audio_data_int16 = audio_data # Already int16

            # Salvar usando wave para garantir compatibilidade
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(AUDIO_SAMPLE_RATE)
                wf.writeframes(audio_data_int16.tobytes())

            # Verificar se o arquivo foi salvo corretamente
            if not os.path.exists(temp_filename) or os.path.getsize(temp_filename) == 0:
                raise ValueError("Arquivo WAV vazio ou não criado após gravação")

            os.rename(temp_filename, final_filename)
            logging.info(f"Audio saved as {final_filename} (size: {os.path.getsize(final_filename)} bytes)")
            saved_successfully = True

        except Exception as e:
            logging.error(f"Error saving audio: {e}", exc_info=True)
            self._set_state(STATE_ERROR_AUDIO) # Or a new ERROR_SAVE state?
            self._log_status(f"Error saving audio: {e}", error=True)
            if os.path.exists(temp_filename): self._delete_audio_file(temp_filename) # Cleanup temp
            # Stay in error state - DO NOT proceed to transcription

        # --- Trigger Transcription Task ONLY if save was successful ---
        if saved_successfully:
            self._set_state(STATE_TRANSCRIBING)
            # Run transcription in a new thread
            threading.Thread(target=self._transcribe_audio_task, args=(final_filename,), daemon=True, name="TranscriptionThread").start()
        else:
             logging.error("Skipping transcription because audio save failed.")
             # State should already be ERROR_AUDIO


    def _delete_audio_file(self, filename):
        """Safely attempts to delete an audio file."""
        if filename and os.path.exists(filename):
            try:
                os.remove(filename)
                logging.info(f"Deleted audio file: {filename}")
            except OSError as e:
                logging.warning(f"Could not delete audio file '{filename}': {e}")

    # --- Transcription Processing (Simplified) ---
    def _transcribe_audio_task(self, audio_filename):
        """Transcribes a single audio file using the Whisper pipeline."""
        start_process_time = time.time()
        logging.info(f"Transcription task started for {audio_filename}")
        text_result = None
        transcription_error = None

        # Verificar integridade do arquivo antes de transcrever
        try:
            with wave.open(audio_filename, 'rb') as wf:
                n_channels = wf.getnchannels()
                framerate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                logging.debug(f"WAV check: Channels={n_channels}, Rate={framerate}, Width={sampwidth}, Frames={n_frames}")
                if n_channels != AUDIO_CHANNELS:
                    raise ValueError(f"Invalid channels: {n_channels} (expected {AUDIO_CHANNELS})")
                if framerate != AUDIO_SAMPLE_RATE:
                    raise ValueError(f"Invalid sample rate: {framerate} (expected {AUDIO_SAMPLE_RATE})")
                if sampwidth != 2:  # 16-bit
                    raise ValueError(f"Invalid sample width: {sampwidth} (expected 2)")
                if n_frames == 0:
                    raise ValueError("WAV file has zero frames.")
        except wave.Error as e:
            logging.error(f"Invalid WAV file format for {audio_filename}: {e}")
            transcription_error = e
            text_result = f"[Transcription Error: Invalid WAV format]"
        except ValueError as e:
            logging.error(f"Invalid WAV file properties for {audio_filename}: {e}")
            transcription_error = e
            text_result = f"[Transcription Error: Invalid WAV properties]"
        except Exception as e:
            logging.error(f"Error opening/checking WAV file {audio_filename}: {e}", exc_info=True)
            transcription_error = e
            text_result = f"[Transcription Error: Cannot read WAV]"

        # If WAV check failed, set error state and return early
        if transcription_error:
            with self.transcription_lock:
                self.transcription_in_progress = False # Ensure flag is cleared
            self._set_state(STATE_ERROR_TRANSCRIPTION)
            self._log_status(f"Error: Invalid audio file - {transcription_error}", error=True)
            self._delete_audio_file(audio_filename)
            return

        # Ensure transcription_in_progress is set
        with self.transcription_lock:
            self.transcription_in_progress = True

        try:
            if self.pipe is None:
                raise RuntimeError("Transcription pipeline unavailable.")

            logging.debug(f"Calling pipeline for: {audio_filename}")

            # --- Transformers Warnings Handling ---
            # Warning 1: 'inputs' is deprecated -> 'input_features'.
            #   The pipeline handles this internally when given a filename. No action needed here.
            # Warning 2: Multilingual default changed to detection.
            #   Current behavior is detection. If translation to English is ALWAYS desired,
            #   add generate_kwargs={"language": "en"} below.
            # Warning 3: Attention mask not set.
            #   The pipeline *should* handle this. If issues arise, investigate manual processing.
            # --- End Warnings Handling ---

            # Example if English translation was always needed:
            # result = self.pipe(audio_filename, chunk_length_s=30, batch_size=16, return_timestamps=False, generate_kwargs={"language": "en"})
            # Default behavior (language detection):
            result = self.pipe(audio_filename, chunk_length_s=30, batch_size=16, return_timestamps=False)

            logging.debug(f"Pipeline raw result: {result}")

            if result and "text" in result:
                text_result = result["text"].strip()
                if not text_result:
                    logging.warning(f"Empty transcription for {audio_filename}")
                    text_result = "[No speech detected]"
                else:
                    logging.info(f"Transcription successful for {audio_filename}.")
            else:
                logging.error(f"Unexpected pipeline result format: {result}")
                text_result = "[Transcription failed: Bad format]"
                transcription_error = RuntimeError("Unexpected result format")

        except RuntimeError as e:
            # Catch specific runtime errors like OOM
            logging.error(f"Runtime error during transcription for {audio_filename}: {e}", exc_info=True)
            transcription_error = e
            text_result = f"[Transcription Error: {e}]"
            if "out of memory" in str(e).lower():
                 self._set_state(STATE_ERROR_MODEL) # Indicate potential model/GPU issue
            else:
                 self._set_state(STATE_ERROR_TRANSCRIPTION)
        except Exception as e:
            logging.error(f"Error during transcription for {audio_filename}: {e}", exc_info=True)
            transcription_error = e
            text_result = f"[Transcription Error: {e}]"
            self._set_state(STATE_ERROR_TRANSCRIPTION)


        finally:
            end_process_time = time.time()
            logging.info(f"Transcription task for {audio_filename} finished in {end_process_time - start_process_time:.2f}s.")

            # --- Handle Result ---
            with self.transcription_lock:
                self.transcription_in_progress = False # Release flag

                if transcription_error:
                    # State might have been set already in except blocks
                    if not self.current_state.startswith("ERROR"):
                         self._set_state(STATE_ERROR_TRANSCRIPTION)
                    self._log_status(f"Error transcribing {os.path.basename(audio_filename)}: {transcription_error}", error=True)
                    # Stay in error state
                elif text_result and text_result != "[No speech detected]":
                    # Apply OpenRouter text correction if enabled
                    if self.openrouter_enabled and self.openrouter_client:
                        try:
                            logging.info("Applying OpenRouter text correction...")
                            corrected_text = self._correct_text_with_openrouter(text_result)
                            if corrected_text != text_result:
                                logging.info("Text was corrected by OpenRouter")
                                text_result = corrected_text
                        except Exception as e:
                            logging.error(f"Error during OpenRouter text correction: {e}")
                            # Continue with original text on error

                    self._handle_transcription_result(text_result) # Handle copy/paste
                    self._set_state(STATE_IDLE) # Back to idle after success
                else: # No text or "[No speech detected]"
                     logging.warning(f"Processed {audio_filename} with no significant text.")
                     self._log_status("Transcription finished: No speech detected.", error=False) # Log as info
                     self._set_state(STATE_IDLE) # Back to idle

            # --- Cleanup ---
            self._delete_audio_file(audio_filename) # Delete file after processing


    # --- Settings Application Logic (Called from Settings Thread) ---
    def apply_settings_from_external(self, new_key, new_mode, new_auto_paste,
                                   new_sound_enabled=None, new_sound_frequency=None,
                                   new_sound_duration=None, new_sound_volume=None,
                                   new_reload_key=None, new_openrouter_enabled=None,
                                   new_openrouter_api_key=None, new_openrouter_model=None):
        """Applies settings passed from the external settings window/thread."""
        logging.info("Applying new configuration from external source.")
        key_changed = False
        mode_changed = False
        paste_changed = False
        sound_changed = False
        config_needs_saving = False

        # Apply key
        if new_key is not None and new_key.lower() != self.record_key:
            self.record_key = new_key.lower()
            key_changed = True
            config_needs_saving = True
            logging.info(f"Record key changed to: '{self.record_key}'")

        # Apply mode
        new_mode_str = str(new_mode).lower() # Ensure string
        if new_mode_str in ["toggle", "press"] and new_mode_str != self.record_mode:
            self.record_mode = new_mode_str
            mode_changed = True
            config_needs_saving = True
            logging.info(f"Record mode changed to: '{self.record_mode}'")

        # Apply auto-paste
        auto_paste_bool = bool(new_auto_paste)
        if auto_paste_bool != self.auto_paste:
            self.auto_paste = auto_paste_bool
            paste_changed = True
            config_needs_saving = True
            logging.info(f"Auto paste changed to: {self.auto_paste}")

        # Apply sound settings
        if new_sound_enabled is not None:
            sound_enabled_bool = bool(new_sound_enabled)
            if sound_enabled_bool != self.sound_enabled:
                self.sound_enabled = sound_enabled_bool
                sound_changed = True
                config_needs_saving = True
                logging.info(f"Sound enabled changed to: {self.sound_enabled}")

        if new_sound_frequency is not None:
            try:
                freq_val = int(new_sound_frequency)
                if 20 <= freq_val <= 20000 and freq_val != self.sound_frequency:
                    self.sound_frequency = freq_val
                    sound_changed = True
                    config_needs_saving = True
                    logging.info(f"Sound frequency changed to: {self.sound_frequency} Hz")
            except (ValueError, TypeError):
                logging.warning(f"Invalid sound frequency value: {new_sound_frequency}")

        if new_sound_duration is not None:
            try:
                dur_val = float(new_sound_duration)
                if 0.05 <= dur_val <= 2.0 and dur_val != self.sound_duration:
                    self.sound_duration = dur_val
                    sound_changed = True
                    config_needs_saving = True
                    logging.info(f"Sound duration changed to: {self.sound_duration} seconds")
            except (ValueError, TypeError):
                logging.warning(f"Invalid sound duration value: {new_sound_duration}")

        if new_sound_volume is not None:
            try:
                vol_val = float(new_sound_volume)
                if 0.0 <= vol_val <= 1.0 and vol_val != self.sound_volume:
                    self.sound_volume = vol_val
                    sound_changed = True
                    config_needs_saving = True
                    logging.info(f"Sound volume changed to: {self.sound_volume}")
            except (ValueError, TypeError):
                logging.warning(f"Invalid sound volume value: {new_sound_volume}")

        # Apply reload key setting
        if new_reload_key is not None:
            reload_key_str = str(new_reload_key).lower()
            if reload_key_str != self.reload_key:
                self.reload_key = reload_key_str
                config_needs_saving = True
                logging.info(f"Reload key changed to: {self.reload_key.upper()}")

                # Re-register the reload key hotkey
                self._register_reload_hotkey()

        # Apply OpenRouter settings if provided
        openrouter_changed = False
        if new_openrouter_enabled is not None:
            openrouter_enabled_bool = bool(new_openrouter_enabled)
            if openrouter_enabled_bool != self.openrouter_enabled:
                self.openrouter_enabled = openrouter_enabled_bool
                openrouter_changed = True
                config_needs_saving = True
                logging.info(f"OpenRouter enabled changed to: {self.openrouter_enabled}")

        if new_openrouter_api_key is not None:
            api_key_str = str(new_openrouter_api_key)
            if api_key_str != self.openrouter_api_key:
                self.openrouter_api_key = api_key_str
                openrouter_changed = True
                config_needs_saving = True
                logging.info("OpenRouter API key updated")

        if new_openrouter_model is not None:
            model_str = str(new_openrouter_model)
            if model_str != self.openrouter_model:
                self.openrouter_model = model_str
                openrouter_changed = True
                config_needs_saving = True
                logging.info(f"OpenRouter model changed to: {self.openrouter_model}")

        # Reinitialize OpenRouter client if settings changed
        if openrouter_changed:
            self._init_openrouter_client()

        # Save config only if something actually changed
        if config_needs_saving:
            logging.info(f"Settings applied: Key='{self.record_key.upper()}', Mode='{self.record_mode}', AutoPaste={self.auto_paste}, Sound={self.sound_enabled}, ReloadKey='{self.reload_key.upper()}'")
            self._save_config() # File I/O is generally safe

            # Re-register hotkeys only if key or mode changed
            if key_changed or mode_changed:
                logging.info("Hotkey config changed. Re-registering.")
                # Use the simpler register_hotkeys first.
                self.register_hotkeys()
            else:
                logging.info("Hotkey config unchanged, but other settings might have.")
                # Update status log even if hotkey didn't change
                self._log_status(f"Config updated. Hotkey: {self.record_key.upper()} ({self.record_mode})")
        else:
             logging.info("No settings changed.")


    # --- Cleanup ---
    def _cleanup_old_audio_files_on_startup(self):
        """Removes leftover .wav files from previous runs on startup."""
        removed_count = 0
        logging.info("Running startup audio file cleanup...")
        try:
            # Target both temp and final recording files leftover from previous sessions
            files_to_check = glob.glob("temp_recording_*.wav") + glob.glob("recording_*.wav")

            for f in files_to_check:
                self._delete_audio_file(f) # Use helper for deletion
                removed_count += 1

            if removed_count > 0:
                 logging.info(f"Cleanup (startup): {removed_count} old audio file(s) removed.")
            else:
                 logging.debug("Cleanup (startup): No old audio files found.")

        except Exception as e:
            logging.error(f"Error during startup audio file cleanup: {e}")

    def shutdown(self):
        """Handles application closing sequence initiated by tray exit."""
        # <<< FIX: Prevent double execution >>>
        if self.shutting_down:
            logging.debug("Shutdown already in progress, ignoring duplicate request.")
            return
        self.shutting_down = True
        # <<< End Fix >>>

        logging.info("Shutdown sequence initiated.")

        # 1. Signal the periodic re-register thread to stop
        self.stop_reregister_event.set()

        # 2. Stop recording if active (don't save)
        with self.recording_lock: # Ensure lock is used
            if self.is_recording:
                logging.warning("Recording active during shutdown. Forcing stop...")
                self.is_recording = False # Signal recording thread to stop
                # Try to close stream gracefully if it exists
                stream = self.audio_stream
                if stream and stream.active:
                     try:
                         stream.stop()
                         stream.close()
                         logging.info("Audio stream stopped and closed during shutdown.")
                     except Exception as e: logging.error(f"Error stopping stream on close: {e}")
                self.recording_data.clear()
                self.audio_stream = None # Clear reference

        # 3. Unhook hotkeys (also done by atexit, but good practice)
        self._cleanup_hotkeys()

        # 4. Stop any running transcription? Difficult to interrupt cleanly. Log instead.
        with self.transcription_lock: # Ensure lock is used
            if self.transcription_in_progress:
                logging.warning("Shutting down while transcription is in progress. Transcription may not complete.")
                # We can't easily kill the transformers pipeline thread.

        # 5. Wait briefly for the timer thread to exit (optional but good practice)
        if self.reregister_timer_thread and self.reregister_timer_thread.is_alive():
            logging.debug("Waiting for periodic re-register thread to stop...")
            self.reregister_timer_thread.join(timeout=1.5) # Wait slightly longer
            if self.reregister_timer_thread.is_alive():
                logging.warning("Periodic re-register thread did not stop gracefully.")

        logging.info("Core shutdown sequence complete.")

# --- End of WhisperCore Class ---


# --- Icon Creation ---
def create_image(width, height, color1, color2=None):
    """Creates a simple PIL image for the tray icon."""
    image = Image.new('RGB', (width, height), color1)
    if color2: # Optional second color for visual state indication (e.g., border)
        dc = ImageDraw.Draw(image)
        dc.rectangle(
            (width // 4, height // 4, width * 3 // 4, height * 3 // 4),
            fill=color2)
    return image

# --- Global Variables ---
# Need a way for callbacks to access the core instance and icon
core_instance = None
tray_icon = None

# --- Tray Icon State Mapping ---
ICON_COLORS = {
    STATE_IDLE: ('green', 'white'),
    STATE_LOADING_MODEL: ('gray', 'yellow'),
    STATE_RECORDING: ('red', 'white'),
    STATE_SAVING: ('orange', 'white'),
    STATE_TRANSCRIBING: ('blue', 'white'),
    STATE_ERROR_MODEL: ('black', 'red'),
    STATE_ERROR_AUDIO: ('black', 'red'),
    STATE_ERROR_TRANSCRIPTION: ('black', 'red'),
    STATE_ERROR_SETTINGS: ('black', 'red'),
}
DEFAULT_ICON_COLOR = ('black', 'white') # Fallback

# --- Tray Callback Functions ---
def update_tray_icon(state):
    """Callback function to update the tray icon based on core state."""
    global tray_icon
    if tray_icon:
        color1, color2 = ICON_COLORS.get(state, DEFAULT_ICON_COLOR)
        icon_image = create_image(64, 64, color1, color2)
        tray_icon.icon = icon_image
        tooltip = f"Whisper Recorder ({state})"
        if state == STATE_IDLE and core_instance:
             tooltip += f" - Record: {core_instance.record_key.upper()} - Reload: {core_instance.reload_key.upper()}"
        elif state.startswith("ERROR") and core_instance:
             tooltip += f" - Check Logs/Settings"

        tray_icon.title = tooltip
        tray_icon.update_menu() # Update the menu based on the new state
        logging.debug(f"Tray icon updated for state: {state}")

# --- Settings Window (Run in Separate Thread) ---
settings_window_instance = None # Track if window is already open

def run_settings_gui():
    """Runs the Tkinter settings window GUI in its own mainloop."""
    global core_instance, settings_window_instance # Need globals here too
    # Import messagebox locally where needed
    import tkinter.messagebox as messagebox

    # Create a temporary hidden root for this instance of the settings window
    temp_tk_root = tk.Tk()
    temp_tk_root.withdraw()

    # Ensure the window instance tracker is cleared if the root somehow fails
    def on_temp_root_close():
        global settings_window_instance
        logging.debug("Temporary Tk root for settings is closing.")
        settings_window_instance = None
        if temp_tk_root and temp_tk_root.winfo_exists():
            temp_tk_root.destroy() # Ensure it's destroyed
    temp_tk_root.protocol("WM_DELETE_WINDOW", on_temp_root_close)


    # Create Toplevel as child of the temporary root
    try:
        settings_win = tk.Toplevel(temp_tk_root)
        settings_window_instance = settings_win # Store reference
    except Exception as e:
        logging.error(f"Failed to create Toplevel for settings: {e}", exc_info=True)
        on_temp_root_close() # Clean up root if Toplevel fails
        return

    # --- Configure the Toplevel window (INDENTED under the try) ---
    settings_win.title("Settings")
    settings_win.configure(bg="#2e2e2e")
    settings_win.resizable(False, False)
    settings_win.attributes("-topmost", True) # Request to stay on top

    # --- Calculate Center Position ---
    settings_win.update_idletasks() # Update geometry info
    # Use fixed size directly for calculation and setting geometry
    window_width = int(SETTINGS_WINDOW_GEOMETRY.split('x')[0]) # Extract width
    window_height = int(SETTINGS_WINDOW_GEOMETRY.split('x')[1]) # Extract height

    screen_width = settings_win.winfo_screenwidth()
    screen_height = settings_win.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    logging.info(f"Centering settings window at {x_cordinate}, {y_cordinate}")
    # Define um tamanho fixo (ajustado) para a janela de configurações
    settings_win.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")


    # --- Variables ---
    # Use tk variables to link UI elements to data
    auto_paste_var = tk.BooleanVar(value=core_instance.auto_paste)
    mode_var = tk.StringVar(value=core_instance.record_mode)
    # For hotkey detection
    detected_key_var = tk.StringVar(value=core_instance.record_key.upper()) # Displays current/detected key
    new_record_key_temp = None # Stores the newly detected key temporarily (lowercase)

    # For reload hotkey detection
    reload_key_var = tk.StringVar(value=core_instance.reload_key.upper()) # Displays current/detected reload key
    new_reload_key_temp = None # Stores the newly detected reload key temporarily (lowercase)

    # Sound settings variables
    sound_enabled_var = tk.BooleanVar(value=core_instance.sound_enabled)
    sound_frequency_var = tk.IntVar(value=core_instance.sound_frequency)
    sound_duration_var = tk.DoubleVar(value=core_instance.sound_duration)
    sound_volume_var = tk.DoubleVar(value=core_instance.sound_volume)

    # OpenRouter API settings variables
    openrouter_enabled_var = tk.BooleanVar(value=core_instance.openrouter_enabled)
    openrouter_api_key_var = tk.StringVar(value=core_instance.openrouter_api_key)
    openrouter_model_var = tk.StringVar(value=core_instance.openrouter_model)

    # --- Functions ---
    detect_key_thread = None # Keep track of the detection thread

    def detect_key_task_internal():
        """Internal task to detect key press, runs in a thread."""
        nonlocal new_record_key_temp
        detected_key_str = "ERROR"
        new_record_key_temp = None
        logging.info("Detect key task started (in detect thread).")

        # Schedule button update on the Tkinter thread using temp_tk_root.after
        # Check if window still exists before scheduling
        if settings_window_instance and settings_window_instance.winfo_exists():
            temp_tk_root.after(0, lambda: detect_key_button.config(text="PRESS KEY...", state=tk.DISABLED))
        else:
            logging.warning("Settings window closed before starting key detection UI update.")
            return # Don't proceed if window is gone

        original_hotkey_press = None
        original_hotkey_release = None

        # Interaction with core_instance (hotkeys) from detect thread - potential risk
        try:
            if core_instance:
                logging.info("Unhooking global hotkeys for detection (from detect thread)...")
                # Store original handlers before unhooking
                with core_instance.keyboard_lock:
                    original_hotkey_press = core_instance.hotkey_press_handler
                    original_hotkey_release = core_instance.hotkey_release_handler
                core_instance._cleanup_hotkeys() # Unhook global hotkeys
            time.sleep(0.1) # Give time for unhooking
            logging.info("Waiting for key event...")
            # Use read_hotkey for better complex key detection
            detected_combination = keyboard.read_hotkey(suppress=False) # Don't suppress here, just read
            logging.info(f"Hotkey combination read: {detected_combination}")

            # Re-hook immediately after reading to minimize disruption
            logging.info("Re-registering original hotkeys immediately after detection read...")
            if core_instance:
                 # Re-register using the stored config, not the potentially old handlers
                 core_instance.register_hotkeys()

            # Process the detected key
            if detected_combination:
                 # Check if it's just a modifier
                 # Note: read_hotkey() returns the canonical name directly
                 if detected_combination.lower() in keyboard.all_modifiers:
                     logging.warning(f"Ignoring modifier key press: {detected_combination}.")
                     detected_key_str = "MODIFIER - TRY AGAIN"
                 else:
                     new_record_key_temp = detected_combination.lower() # Store lowercase canonical name
                     detected_key_str = new_record_key_temp.upper() # Display uppercase
            else:
                detected_key_str = "DETECTION FAILED" # Handle unexpected event types

        except Exception as e:
            logging.error(f"Error detecting key: {e}", exc_info=True)
            detected_key_str = "ERROR"
            new_record_key_temp = None
            # Ensure hotkeys are re-registered even on error
            if core_instance:
                logging.error("Re-registering original hotkeys after detection error...")
                core_instance.register_hotkeys()
        finally:
            # Schedule UI update back on the Tkinter thread
            if settings_window_instance and settings_window_instance.winfo_exists():
                 temp_tk_root.after(0, lambda: update_detection_ui(detected_key_str))
            logging.info("Detect key task finished (in detect thread).")

    def update_detection_ui(key_text, is_reload_key=False):
        """Updates the key label and button state (runs in Tkinter thread via after())."""
        # Check if window still exists before updating
        if settings_window_instance and settings_window_instance.winfo_exists():
            if is_reload_key:
                reload_key_var.set(key_text)
                # Ensure button exists before configuring
                if 'detect_reload_key_button' in locals() and isinstance(detect_reload_key_button, tk.Button) and detect_reload_key_button.winfo_exists():
                    detect_reload_key_button.config(text="Detect Key", state=tk.NORMAL)
                elif 'detect_reload_key_button' in globals() and isinstance(detect_reload_key_button, tk.Button) and detect_reload_key_button.winfo_exists():
                    detect_reload_key_button.config(text="Detect Key", state=tk.NORMAL)
                else:
                    logging.warning("detect_reload_key_button not found or not a valid widget during UI update.")
            else:
                detected_key_var.set(key_text)
                # Ensure button exists before configuring
                if 'detect_key_button' in locals() and isinstance(detect_key_button, tk.Button) and detect_key_button.winfo_exists():
                    detect_key_button.config(text="Detect Key", state=tk.NORMAL)
                elif 'detect_key_button' in globals() and isinstance(detect_key_button, tk.Button) and detect_key_button.winfo_exists():
                    detect_key_button.config(text="Detect Key", state=tk.NORMAL)
                else:
                    logging.warning("detect_key_button not found or not a valid widget during UI update.")
        else:
            logging.warning("Settings window closed before UI update for key detection.")

    def start_detect_key():
        """Starts the key detection thread."""
        nonlocal detect_key_thread
        # This function runs in the Tkinter thread (called by button command)
        if detect_key_thread and detect_key_thread.is_alive():
            logging.warning("Key detection thread already running.")
            return

        if settings_window_instance and settings_window_instance.winfo_exists():
             detected_key_var.set("PRESS KEY...")
        # Start the detection task in its own thread
        detect_key_thread = threading.Thread(target=detect_key_task_internal, daemon=True, name="DetectKeyThread")
        detect_key_thread.start()

    def detect_reload_key_task_internal():
        """Internal task to detect reload key press, runs in a thread."""
        nonlocal new_reload_key_temp
        detected_key_str = "ERROR"
        new_reload_key_temp = None
        logging.info("Detect reload key task started (in detect thread).")

        # Schedule button update on the Tkinter thread using temp_tk_root.after
        # Check if window still exists before scheduling
        if settings_window_instance and settings_window_instance.winfo_exists():
            temp_tk_root.after(0, lambda: detect_reload_key_button.config(text="PRESS KEY...", state=tk.DISABLED))
        else:
            logging.warning("Settings window closed before starting reload key detection UI update.")
            return # Don't proceed if window is gone

        # Interaction with core_instance (hotkeys) from detect thread - potential risk
        try:
            if core_instance:
                logging.info("Unhooking global hotkeys for reload key detection...")
                # Store original handlers before unhooking
                with core_instance.keyboard_lock:
                    core_instance._cleanup_hotkeys() # Unhook global hotkeys
            time.sleep(0.1) # Give time for unhooking
            logging.info("Waiting for reload key event...")
            # Use read_hotkey for better complex key detection
            detected_combination = keyboard.read_hotkey(suppress=False) # Don't suppress here, just read
            logging.info(f"Reload hotkey combination read: {detected_combination}")

            # Re-hook immediately after reading to minimize disruption
            logging.info("Re-registering original hotkeys immediately after detection read...")
            if core_instance:
                 # Re-register using the stored config, not the potentially old handlers
                 core_instance.register_hotkeys()

            # Process the detected key
            detected_key_str = detected_combination.upper()
            new_reload_key_temp = detected_combination.lower() # Store for later use in apply_settings
            logging.info(f"Detected reload key: {detected_key_str} (stored as: {new_reload_key_temp})")

        except Exception as e:
            logging.error(f"Error detecting reload key: {e}", exc_info=True)
            detected_key_str = "ERROR"
            new_reload_key_temp = None
            # Ensure hotkeys are re-registered even on error
            if core_instance:
                logging.error("Re-registering original hotkeys after reload key detection error...")
                core_instance.register_hotkeys()
        finally:
            # Schedule UI update back on the Tkinter thread
            if settings_window_instance and settings_window_instance.winfo_exists():
                 temp_tk_root.after(0, lambda: update_detection_ui(detected_key_str, is_reload_key=True))
            logging.info("Detect reload key task finished (in detect thread).")

    def start_detect_reload_key():
        """Starts the reload key detection thread."""
        nonlocal detect_key_thread
        # This function runs in the Tkinter thread (called by button command)
        if detect_key_thread and detect_key_thread.is_alive():
            logging.warning("Key detection thread already running.")
            return

        if settings_window_instance and settings_window_instance.winfo_exists():
             reload_key_var.set("PRESS KEY...")
        # Start the detection task in its own thread
        detect_key_thread = threading.Thread(target=detect_reload_key_task_internal, daemon=True, name="DetectReloadKeyThread")
        detect_key_thread.start()

    def apply_settings():
        """Applies the selected settings by calling the core instance method."""
        # This function runs in the Tkinter thread
        nonlocal new_record_key_temp, new_reload_key_temp
        logging.info("Apply settings clicked (in Tkinter thread).")

        # Basic validation before applying (can stay here)
        if core_instance.pipe is None:
            messagebox.showwarning("Apply Settings", "Model not loaded yet. Cannot apply.", parent=settings_win)
            return
        # Check recording state via core_instance (reading state should be safe)
        with core_instance.recording_lock:
            if core_instance.is_recording:
                messagebox.showwarning("Apply Settings", "Cannot apply while recording.", parent=settings_win)
                return
        with core_instance.transcription_lock:
            if core_instance.transcription_in_progress:
                messagebox.showwarning("Apply Settings", "Cannot apply while transcribing.", parent=settings_win)
                return

        key_to_apply = new_record_key_temp # Use the detected key if one was detected
        mode_to_apply = mode_var.get()
        auto_paste_to_apply = auto_paste_var.get()
        reload_key_to_apply = new_reload_key_temp # Use the detected reload key if one was detected

        # Get sound settings
        sound_enabled_to_apply = sound_enabled_var.get()

        # Validate sound settings before applying
        try:
            sound_freq_to_apply = int(sound_frequency_var.get())
            if not (20 <= sound_freq_to_apply <= 20000):
                messagebox.showwarning("Invalid Value", "Frequency must be between 20 and 20000 Hz", parent=settings_win)
                return
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Value", "Frequency must be a number", parent=settings_win)
            return

        try:
            sound_duration_to_apply = float(sound_duration_var.get())
            if not (0.05 <= sound_duration_to_apply <= 2.0):
                messagebox.showwarning("Invalid Value", "Duration must be between 0.05 and 2.0 seconds", parent=settings_win)
                return
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Value", "Duration must be a number", parent=settings_win)
            return

        try:
            sound_volume_to_apply = float(sound_volume_var.get())
            if not (0.0 <= sound_volume_to_apply <= 1.0):
                messagebox.showwarning("Invalid Value", "Volume must be between 0.0 and 1.0", parent=settings_win)
                return
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Value", "Volume must be a number", parent=settings_win)
            return

        # Call the core instance method to handle the actual application and re-registration
        # This call crosses the thread boundary.
        try:
             if hasattr(core_instance, 'apply_settings_from_external'):
                 core_instance.apply_settings_from_external(
                     new_key=key_to_apply,
                     new_mode=mode_to_apply,
                     new_auto_paste=auto_paste_to_apply,
                     new_sound_enabled=sound_enabled_to_apply,
                     new_sound_frequency=sound_freq_to_apply,
                     new_sound_duration=sound_duration_to_apply,
                     new_sound_volume=sound_volume_to_apply,
                     new_reload_key=reload_key_to_apply,
                     new_openrouter_enabled=openrouter_enabled_var.get(),
                     new_openrouter_api_key=openrouter_api_key_var.get(),
                     new_openrouter_model=openrouter_model_var.get()
                 )
             else:
                 logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
                 messagebox.showerror("Internal Error", "Cannot apply settings: Core method missing.", parent=settings_win)
                 return # Don't close window

        except Exception as e:
             logging.error(f"Error calling apply_settings_from_external from settings thread: {e}", exc_info=True)
             messagebox.showerror("Error", f"Failed to apply settings:\n{e}", parent=settings_win)
             # Don't close the window on error, let user try again or cancel
             return # Don't close window

        # Clear temp key only after successful application attempt
        new_record_key_temp = None

        # Close the window after applying successfully
        close_settings()

    # --- REMOVED: Manual Re-register Button Logic ---
    # def on_manual_reregister_click(): ... (Function removed)

    def close_settings():
        """Closes the settings window and its temporary root (runs in Tkinter thread)."""
        global settings_window_instance
        logging.info("Settings window closing sequence started (in Tkinter thread).")

        # Destroy the Toplevel window first
        if settings_window_instance and settings_window_instance.winfo_exists():
             logging.debug("Destroying settings Toplevel window...")
             try:
                 settings_win.destroy() # Use the local variable 'settings_win' from outer scope
             except tk.TclError as e:
                 logging.warning(f"TclError destroying settings window (might be already gone): {e}")
        else:
             logging.warning("Tried to close settings window, but instance or window no longer exists.")

        settings_window_instance = None # Clear the global reference

        # Now destroy the temporary root, which should terminate its mainloop
        if temp_tk_root and temp_tk_root.winfo_exists():
            logging.debug("Destroying temporary Tk root for settings window...")
            try:
                temp_tk_root.destroy() # This stops the mainloop of this thread
            except tk.TclError as e:
                logging.warning(f"TclError destroying temp root (might be already gone): {e}")

        logging.info("Settings window closing sequence finished (in Tkinter thread).")

    # --- UI Construction ---
    settings_win.protocol("WM_DELETE_WINDOW", close_settings) # Handle [X] button

    # Create a main frame to hold everything
    main_frame = tk.Frame(settings_win, bg="#2e2e2e")
    main_frame.pack(fill="both", expand=True)

    # Create a frame for settings content with padding
    settings_frame = tk.Frame(main_frame, bg="#2e2e2e")
    settings_frame.pack(fill="both", expand=True, padx=15, pady=(15, 0))

    # Create a frame for buttons at the bottom
    button_frame = tk.Frame(main_frame, bg="#2e2e2e")
    button_frame.pack(side="bottom", fill="x", padx=15, pady=10)

    # --- Auto Paste ---
    auto_frame = tk.Frame(settings_frame, bg="#2e2e2e")
    auto_frame.pack(pady=(0, 20), fill="x", anchor="w")
    tk.Label(auto_frame, text="Auto Paste Text:", bg="#2e2e2e", fg="white").pack(side="left", padx=(0, 15))
    tk.Checkbutton(auto_frame, variable=auto_paste_var, text="Enabled", bg="#2e2e2e", fg="white", selectcolor="#1e1e1e", activebackground="#626262", activeforeground="white", anchor="w").pack(side="left")

    # --- Record Hotkey ---
    key_frame = tk.Frame(settings_frame, bg="#2e2e2e")
    key_frame.pack(pady=15, fill="x", anchor="w")
    tk.Label(key_frame, text="Record Hotkey:", bg="#2e2e2e", fg="white").pack(side="left", padx=(0, 15))
    # Label to display the currently detected/set key
    key_display_label = tk.Label(key_frame, textvariable=detected_key_var, font=("Helvetica", 10, "bold"), width=18, anchor="w", bg="#3c3c3c", fg="#a0ffa0", relief=tk.FLAT, padx=5)
    key_display_label.pack(side="left", padx=5)
    # Button to trigger detection
    detect_key_button = tk.Button(key_frame, text="Detect Key", command=start_detect_key, width=12,
                                   bg="#4d4d4d", fg="white", activebackground="#626262", activeforeground="white", relief=tk.FLAT)
    detect_key_button.pack(side="left", padx=5)

    # --- Reload Hotkey ---
    reload_key_frame = tk.Frame(settings_frame, bg="#2e2e2e")
    reload_key_frame.pack(pady=15, fill="x", anchor="w")
    tk.Label(reload_key_frame, text="Reload Hotkey:", bg="#2e2e2e", fg="white").pack(side="left", padx=(0, 15))
    # Label to display the currently detected/set reload key
    reload_key_display_label = tk.Label(reload_key_frame, textvariable=reload_key_var, font=("Helvetica", 10, "bold"), width=18, anchor="w", bg="#3c3c3c", fg="#a0ffa0", relief=tk.FLAT, padx=5)
    reload_key_display_label.pack(side="left", padx=5)
    # Button to trigger detection
    detect_reload_key_button = tk.Button(reload_key_frame, text="Detect Key", command=start_detect_reload_key, width=12,
                                   bg="#4d4d4d", fg="white", activebackground="#626262", activeforeground="white", relief=tk.FLAT)
    detect_reload_key_button.pack(side="left", padx=5)

    # --- Record Mode ---
    mode_frame = tk.Frame(settings_frame, bg="#2e2e2e")
    mode_frame.pack(pady=15, fill="x", anchor="w")
    tk.Label(mode_frame, text="Record Mode:", bg="#2e2e2e", fg="white").pack(side="left", padx=(0, 15))
    tk.Radiobutton(mode_frame, text="Toggle", variable=mode_var, value="toggle", bg="#2e2e2e", fg="white", selectcolor="#1e1e1e", activebackground="#626262", activeforeground="white").pack(side="left", padx=5)
    tk.Radiobutton(mode_frame, text="Press/Hold", variable=mode_var, value="press", bg="#2e2e2e", fg="white", selectcolor="#1e1e1e", activebackground="#626262", activeforeground="white").pack(side="left", padx=5)

    # --- Sound Settings Section ---
    sound_section_frame = tk.Frame(settings_frame, bg="#2e2e2e")
    sound_section_frame.pack(pady=15, fill="x", anchor="w")
    tk.Label(sound_section_frame, text="Sound Settings", bg="#2e2e2e", fg="white", font=("Helvetica", 10, "bold")).pack(anchor="w")

    # Sound Enable Checkbox
    sound_enable_frame = tk.Frame(sound_section_frame, bg="#2e2e2e")
    sound_enable_frame.pack(pady=(5, 5), fill="x")
    tk.Checkbutton(sound_enable_frame, variable=sound_enabled_var, text="Enable Sound Feedback",
                   bg="#2e2e2e", fg="white", selectcolor="#1e1e1e",
                   activebackground="#626262", activeforeground="white").pack(side="left")

    # Sound Volume Slider
    volume_frame = tk.Frame(sound_section_frame, bg="#2e2e2e")
    volume_frame.pack(pady=5, fill="x")
    tk.Label(volume_frame, text="Volume:", bg="#2e2e2e", fg="white", width=10, anchor="w").pack(side="left")
    volume_slider = tk.Scale(volume_frame, variable=sound_volume_var, from_=0.0, to=1.0, resolution=0.01,
                            orient="horizontal", bg="#3c3c3c", fg="white", troughcolor="#555555",
                            highlightbackground="#2e2e2e", activebackground="#4d4d4d")
    volume_slider.pack(side="left", fill="x", expand=True, padx=(5, 0))

    # Sound Frequency Entry
    freq_frame = tk.Frame(sound_section_frame, bg="#2e2e2e")
    freq_frame.pack(pady=5, fill="x")
    tk.Label(freq_frame, text="Frequency (Hz):", bg="#2e2e2e", fg="white", width=10, anchor="w").pack(side="left")
    freq_entry = tk.Entry(freq_frame, textvariable=sound_frequency_var, width=8, bg="#3c3c3c", fg="white", insertbackground="white")
    freq_entry.pack(side="left", padx=(5, 0))

    # Sound Duration Entry
    duration_frame = tk.Frame(sound_section_frame, bg="#2e2e2e")
    duration_frame.pack(pady=5, fill="x")
    tk.Label(duration_frame, text="Duration (s):", bg="#2e2e2e", fg="white", width=10, anchor="w").pack(side="left")
    duration_entry = tk.Entry(duration_frame, textvariable=sound_duration_var, width=8, bg="#3c3c3c", fg="white", insertbackground="white")
    duration_entry.pack(side="left", padx=(5, 0))

    # Test Sound Button
    def play_test_sound():
        if not core_instance:
            messagebox.showerror("Error", "Core instance not available", parent=settings_win)
            return

        # Get current values from UI
        enabled = sound_enabled_var.get()
        if not enabled:
            messagebox.showinfo("Sound Test", "Sound is disabled. Enable it first to test.", parent=settings_win)
            return

        try:
            freq = int(sound_frequency_var.get())
            duration = float(sound_duration_var.get())
            volume = float(sound_volume_var.get())

            # Validate ranges
            if not (20 <= freq <= 20000):
                messagebox.showwarning("Invalid Value", "Frequency must be between 20 and 20000 Hz", parent=settings_win)
                return
            if not (0.05 <= duration <= 2.0):
                messagebox.showwarning("Invalid Value", "Duration must be between 0.05 and 2.0 seconds", parent=settings_win)
                return
            if not (0.0 <= volume <= 1.0):
                messagebox.showwarning("Invalid Value", "Volume must be between 0.0 and 1.0", parent=settings_win)
                return

            # Play test sound in a separate thread
            threading.Thread(
                target=core_instance._play_sound,
                kwargs={"frequency": freq, "duration": duration, "volume": volume, "is_start": True},
                daemon=True,
                name="TestSoundThread"
            ).start()
        except (ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Invalid sound settings: {e}", parent=settings_win)

    test_sound_button = tk.Button(sound_section_frame, text="Test Sound", command=play_test_sound,
                                 bg="#4d4d4d", fg="white", activebackground="#626262",
                                 activeforeground="white", relief=tk.FLAT)
    test_sound_button.pack(pady=(5, 0), anchor="w")

    # --- OpenRouter API Settings Section ---
    openrouter_section_frame = tk.Frame(settings_frame, bg="#2e2e2e")
    openrouter_section_frame.pack(pady=15, fill="x", anchor="w")
    tk.Label(openrouter_section_frame, text="OpenRouter API Settings", bg="#2e2e2e", fg="white", font=("Helvetica", 10, "bold")).pack(anchor="w")

    # OpenRouter Enable Checkbox
    openrouter_enable_frame = tk.Frame(openrouter_section_frame, bg="#2e2e2e")
    openrouter_enable_frame.pack(pady=(5, 5), fill="x")
    tk.Checkbutton(openrouter_enable_frame, variable=openrouter_enabled_var, text="Enable Text Correction with OpenRouter",
                   bg="#2e2e2e", fg="white", selectcolor="#1e1e1e",
                   activebackground="#626262", activeforeground="white").pack(side="left")

    # OpenRouter API Key
    api_key_frame = tk.Frame(openrouter_section_frame, bg="#2e2e2e")
    api_key_frame.pack(pady=(5, 5), fill="x")
    tk.Label(api_key_frame, text="API Key:", bg="#2e2e2e", fg="white", width=10, anchor="w").pack(side="left")
    api_key_entry = tk.Entry(api_key_frame, textvariable=openrouter_api_key_var, bg="#3c3c3c", fg="white", insertbackground="white", relief=tk.FLAT)
    api_key_entry.pack(side="left", fill="x", expand=True, padx=5)

    # OpenRouter Model
    model_frame = tk.Frame(openrouter_section_frame, bg="#2e2e2e")
    model_frame.pack(pady=(5, 5), fill="x")
    tk.Label(model_frame, text="Model:", bg="#2e2e2e", fg="white", width=10, anchor="w").pack(side="left")
    model_entry = tk.Entry(model_frame, textvariable=openrouter_model_var, bg="#3c3c3c", fg="white", insertbackground="white", relief=tk.FLAT)
    model_entry.pack(side="left", fill="x", expand=True, padx=5)

    # --- REMOVED: Manual Re-register Frame ---
    # reregister_frame = tk.Frame(settings_frame, bg="#2e2e2e") ... (Frame removed)

    # --- Action Buttons ---
    apply_button = tk.Button(button_frame, text="Apply", command=apply_settings, width=10,
                              bg="#5cb85c", fg="white", activebackground="#4cae4c", activeforeground="white", relief=tk.FLAT)
    apply_button.pack(side="right", padx=5)
    cancel_button = tk.Button(button_frame, text="Cancel", command=close_settings, width=10,
                               bg="#d9534f", fg="white", activebackground="#c9302c", activeforeground="white", relief=tk.FLAT)
    cancel_button.pack(side="right", padx=5)

    # --- Force UI Update before Making Visible ---
    settings_win.update_idletasks()
    settings_win.update()

    # --- Make Visible and Start Mainloop ---
    settings_win.transient(temp_tk_root) # Associate with hidden temp root
    settings_win.deiconify() # Ensure window is not iconified/minimized
    settings_win.lift() # Bring window to the front
    settings_win.focus_force() # Attempt to force keyboard focus

    logging.info("Starting mainloop for settings window thread...")
    try:
        temp_tk_root.mainloop() # Start the event loop FOR THIS THREAD
    except Exception as e:
        logging.error(f"Error during settings window mainloop: {e}", exc_info=True)
    finally:
        logging.info("Settings window thread mainloop finished.")
        # Ensure cleanup happens even if mainloop crashes (close_settings handles this now)


# --- Callbacks for pystray Menu Items ---
def on_start_recording_menu_click(icon=None, item=None):
    """Starts recording from tray menu."""
    global core_instance
    logging.info(f"Start recording menu click - core_instance: {core_instance is not None}")
    if core_instance:
        # Check state before starting
        with core_instance.recording_lock:
            is_rec = core_instance.is_recording
        if not is_rec:
            logging.info("Starting recording from menu...")
            # Run in thread to avoid blocking pystray
            threading.Thread(target=core_instance.start_recording, daemon=True, name="StartRecordingFromMenu").start()
        else:
            logging.warning("Cannot start recording from menu - already recording")
    else:
        logging.warning("Cannot start recording from menu - core_instance missing")

def on_stop_recording_menu_click(icon=None, item=None):
    """Stops recording from tray menu."""
    global core_instance
    logging.info(f"Stop recording menu click - core_instance: {core_instance is not None}")
    if core_instance:
        # Check state before stopping
        with core_instance.recording_lock:
            is_rec = core_instance.is_recording
        if is_rec:
            logging.info("Stopping recording from menu...")
            # Run in thread to avoid blocking pystray
            threading.Thread(target=core_instance.stop_recording, daemon=True, name="StopRecordingFromMenu").start()
        else:
            logging.warning("Cannot stop recording from menu - not recording")
    else:
        logging.warning("Cannot stop recording from menu - core_instance missing")

# --- Helper function to run toggle in a thread ---
def on_toggle_recording_menu_click(icon=None, item=None):
    """Toggles recording from tray menu (used for default action)."""
    global core_instance
    logging.info(f"Toggle recording menu click - core_instance: {core_instance is not None}")
    if core_instance:
        if hasattr(core_instance, 'toggle_recording'):
            # Run in a thread to avoid blocking pystray
            logging.info("Toggle recording requested via left-click/default action.")
            threading.Thread(target=core_instance.toggle_recording, daemon=True, name="ToggleRecordingFromMenu").start()
        else:
            logging.critical("CRITICAL: toggle_recording method not found on core_instance!")
            update_tray_icon(STATE_ERROR_SETTINGS) # Indicate an internal error
    else:
        logging.warning("Toggle recording requested, but core_instance is None.")


def on_settings_menu_click(icon=None, item=None):
    """Starts the settings GUI in a separate thread."""
    global settings_window_instance
    if settings_window_instance and settings_window_instance.winfo_exists():
        logging.warning("Settings window thread likely already running. Focusing existing window.")
        try:
            settings_window_instance.lift()
            settings_window_instance.focus_force()
        except Exception as e:
            logging.warning(f"Could not focus existing settings window: {e}")
        return

    logging.info("Starting settings window thread...")
    settings_thread = threading.Thread(target=run_settings_gui, daemon=True, name="SettingsGUIThread")
    settings_thread.start()

# --- NEW: Callback for Force Re-register Menu Item ---
def on_force_reregister_menu_click(icon=None, item=None):
    """Forces a reload of the keyboard library and hotkey re-registration."""
    global core_instance
    logging.info("Force keyboard/hotkey reload requested from tray menu.")
    if core_instance:
        if hasattr(core_instance, 'force_reregister_hotkeys'):
            # Run the core logic. It logs success/failure internally.
            # Consider running in a thread if it blocks pystray for too long,
            # but let's try direct call first for simplicity.
            success = core_instance.force_reregister_hotkeys()
            # Optional: Show a messagebox (requires importing tkinter.messagebox here)
            # if success:
            #     messagebox.showinfo("Hotkey Reload", "Keyboard/Hotkey reload successful.")
            # else:
            #     messagebox.showerror("Hotkey Reload", "Failed to reload keyboard/hotkey. Check logs.")
        else:
            logging.critical("CRITICAL: force_reregister_hotkeys method not found on core_instance!")
            update_tray_icon(STATE_ERROR_SETTINGS)
    else:
        logging.warning("Force reload requested, but core_instance is None.")


# --- Dynamic Menu Creation ---
def create_dynamic_menu(icon):
    """Creates the tray icon menu dynamically based on recording state."""
    global core_instance # Ensure core_instance is accessible

    # Default menu if core_instance is not ready
    if not core_instance:
        logging.warning("create_dynamic_menu called but core_instance is None.")
        return (pystray.MenuItem('Carregando...', None, enabled=False),)

    # Determine state safely
    try:
        with core_instance.state_lock:
            current_state = core_instance.current_state
    except AttributeError:
        logging.error("Error accessing core_instance state in create_dynamic_menu.")
        current_state = STATE_LOADING_MODEL # Fallback state

    is_recording = current_state == STATE_RECORDING
    is_idle = current_state == STATE_IDLE
    is_loading = current_state == STATE_LOADING_MODEL
    is_transcribing = current_state == STATE_TRANSCRIBING
    is_saving = current_state == STATE_SAVING
    is_error = current_state.startswith("ERROR")

    default_action_text = 'Carregando...'
    default_action_callback = None
    default_enabled = False # Disable by default

    if is_recording:
        default_action_text = '⏹️ Parar Gravação'
        default_action_callback = on_toggle_recording_menu_click
        default_enabled = True
    elif is_idle:
        default_action_text = '▶️ Iniciar Gravação'
        default_action_callback = on_toggle_recording_menu_click
        default_enabled = True
    elif is_loading:
        default_action_text = 'Carregando Modelo...'
    elif is_transcribing:
        default_action_text = 'Transcrevendo...'
    elif is_saving:
        default_action_text = 'Salvando...'
    elif is_error:
        default_action_text = 'Erro (Ver Logs/⚙️)'
    # Add other states if needed

    # Determine if force reload should be enabled
    can_force_reload = current_state not in [STATE_RECORDING, STATE_SAVING, STATE_LOADING_MODEL]

    # Build the menu tuple
    menu_items = [
        pystray.MenuItem(
            default_action_text,
            default_action_callback,
            default=True,
            enabled=default_enabled
        ),
        # --- Items below only appear on right-click ---
        pystray.MenuItem(
            '⚙️ Configurações',
            on_settings_menu_click,
            enabled=(not is_loading and not is_recording) # Allow settings unless loading/recording
        ),
        # --- NEW: Force Reload Item ---
        pystray.MenuItem(
            '🔄 Recarregar Teclado/Hotkey',
            on_force_reregister_menu_click,
            enabled=can_force_reload # Enable based on state
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem('❌ Sair', on_exit_app)
    ]

    return tuple(menu_items) # Return as a tuple

# --- Tray Exit Function ---
def on_exit_app(icon=None, item=None):
    """Callback to cleanly exit the application."""
    global core_instance, tray_icon
    logging.info("Exit requested from tray icon.")
    if core_instance:
        if hasattr(core_instance, 'shutdown'):
            # Call the shutdown method which now has the idempotency check
            core_instance.shutdown()
        else:
            logging.critical("CRITICAL: shutdown method not found on core_instance!")
            # Attempt basic cleanup anyway if method missing
            if not core_instance.shutting_down: # Basic check even if method missing
                core_instance.shutting_down = True
                core_instance._cleanup_hotkeys()
                core_instance.stop_reregister_event.set()
    if tray_icon:
        tray_icon.stop()
    # No need to destroy settings_tk_root here

# --- Main Execution ---
if __name__ == "__main__":
    # Register a simple atexit handler for logging, the main cleanup is in on_exit_app
    atexit.register(lambda: logging.info("atexit handler reached. Main shutdown should have occurred via on_exit_app."))

    try:
        # Initialize core logic
        core_instance = WhisperCore()

        # Setup initial icon state (Loading)
        initial_state = core_instance.current_state
        color1, color2 = ICON_COLORS.get(initial_state, DEFAULT_ICON_COLOR)
        initial_image = create_image(64, 64, color1, color2)
        initial_tooltip = f"Whisper Recorder ({initial_state})"

        # Create the icon object using the dynamic menu function
        tray_icon = pystray.Icon(
            "whisper_recorder",
            initial_image,
            initial_tooltip,
            menu=pystray.Menu(lambda: create_dynamic_menu(None))
        )

        # Set the callback in the core instance AFTER the icon is created
        core_instance.set_state_update_callback(update_tray_icon)

        # --- REMOVED: Redundant atexit registration of on_exit_app ---
        # atexit.register(on_exit_app, None, None) # Removed, on_exit_app handles tray stop now

        logging.info("Starting pystray icon run loop.")
        # Run the icon loop (this blocks)
        tray_icon.run()

    except Exception as e:
        logging.critical(f"Unhandled exception during application startup or main loop: {e}", exc_info=True)
        try:
            # Attempt to show a simple message box without relying on the core instance
            import tkinter as tk_err # Import locally
            import tkinter.messagebox as mb_err # Import locally
            err_root = tk_err.Tk()
            err_root.withdraw()
            mb_err.showerror("Fatal Error",
                             f"Unexpected error:\n\n{e}\n\nCheck logs for details.\nThe application will now exit.", parent=None)
            err_root.destroy()
        except Exception as tk_e:
            logging.error(f"Could not display fatal error messagebox: {tk_e}")
        # Ensure cleanup is attempted even on fatal error
        if core_instance and not core_instance.shutting_down:
            on_exit_app() # Attempt cleanup if not already shutting down
        sys.exit(1)

    finally:
        logging.info("Application shutting down or finished.")
        # atexit handlers run automatically after this point if sys.exit wasn't called.