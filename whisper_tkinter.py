# -*- coding: utf-8 -*-
import sys
import os
import json
import threading
import time
# Import tkinter apenas quando necessário dentro das funções
import tkinter.messagebox as messagebox
import sounddevice as sd
import numpy as np
import wave
import pystray
from PIL import Image, ImageDraw
import torch
from transformers import pipeline
import pyautogui
import soundfile as sf # Importar soundfile para carregar áudio
# Bibliotecas keyboard e pynput removidas completamente
# Usando apenas Win32HotkeyManager para gerenciamento de hotkeys
import logging
from threading import RLock
import glob
import atexit
try:
    import pyperclip # Optional import
except ImportError:
    pyperclip = None

# Import KeyboardHotkeyManager para gerenciamento de hotkeys usando a biblioteca keyboard
from keyboard_hotkey_manager import KeyboardHotkeyManager
# DirectHotkeyManager foi removido devido a problemas de compatibilidade com o Windows
# from direct_hotkey_manager import DirectHotkeyManager

# Import APIs for text correction
try:
    from openrouter_api import OpenRouterAPI
except ImportError:
    OpenRouterAPI = None
    logging.warning("OpenRouterAPI module not found. OpenRouter text correction will be disabled.")

try:
    from gemini_api import GeminiAPI
except ImportError:
    GeminiAPI = None
    logging.warning("GeminiAPI module not found. Gemini text correction will be disabled.")

# --- Logging Configuration ---
# Configuração para garantir que os logs usem UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s', encoding='utf-8')

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
    "keyboard_library": "win32",  # Apenas a opção 'win32' é suportada agora
    "text_correction_enabled": False,
    "text_correction_service": "none",
    "openrouter_api_key": "",
    "openrouter_model": "deepseek/deepseek-chat-v3-0324:free",
    "gemini_api_key": "",
    "gemini_model": "gemini-2.0-flash-001",
    "gemini_mode": "correction",
    "gemini_general_prompt": "Based on the following text, generate a short response: {text}",
    "gemini_prompt": """You are a speech-to-text correction specialist. Your task is to refine the following transcribed speech.

Key instructions:
- Remove self-corrections (when I say something wrong and then correct myself)
- Focus on removing speech-specific redundancies (repeated words, filler phrases, false starts)
- Make the text MUCH MORE FLUID AND COHERENT
- Remove possible errors in speech
- Maintain the speaker's emotional tone
- Keep the text as fluid as possible (IMPORTANT!)
- Preserve all language transitions (Portuguese/Spanish/English) exactly as they occur
- Connect related thoughts that may be fragmented in the transcription
- Maintain the core message and meaning, but fix speech errors and disfluencies

Return only the improved text without explanations.

Transcribed speech: {text}""",
    "batch_size": 16,
    "gpu_index": 0
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
# Batch size and GPU index configuration keys
BATCH_SIZE_CONFIG_KEY = "batch_size"
GPU_INDEX_CONFIG_KEY = "gpu_index"
# Reload key configuration
RELOAD_KEY_CONFIG_KEY = "reload_key"
# Keyboard library configuration - Usando apenas Win32 API
KEYBOARD_LIBRARY_CONFIG_KEY = "keyboard_library"
# Apenas uma opção de biblioteca de teclado agora
KEYBOARD_LIB_WIN32 = "win32"        # Opção usando apenas Win32 API
# Text correction configuration
TEXT_CORRECTION_ENABLED_CONFIG_KEY = "text_correction_enabled"
TEXT_CORRECTION_SERVICE_CONFIG_KEY = "text_correction_service"
# Service options
SERVICE_NONE = "none"
SERVICE_OPENROUTER = "openrouter"
SERVICE_GEMINI = "gemini"

# OpenRouter API configuration
OPENROUTER_API_KEY_CONFIG_KEY = "openrouter_api_key"
OPENROUTER_MODEL_CONFIG_KEY = "openrouter_model"
# Gemini API configuration
GEMINI_API_KEY_CONFIG_KEY = "gemini_api_key"
GEMINI_MODEL_CONFIG_KEY = "gemini_model"
# Window size adjusted to fit all elements comfortably
SETTINGS_WINDOW_GEOMETRY = "550x700" # Increased width and height for scrollable content
REREGISTER_INTERVAL_SECONDS = 60 # 1 minuto (ajustável aqui)
# Constantes para monitoramento de saúde das bibliotecas de hotkeys
MAX_HOTKEY_FAILURES = 3 # Número máximo de falhas consecutivas antes de tentar alternar para outra biblioteca
HOTKEY_HEALTH_CHECK_INTERVAL = 10 # Intervalo em segundos para verificar a saúde das hotkeys

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
        self.hotkey_lock = RLock() # Lock para sincronizar acesso a recursos de hotkey
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
        # Text correction configuration
        self.text_correction_enabled = DEFAULT_CONFIG[TEXT_CORRECTION_ENABLED_CONFIG_KEY]
        self.text_correction_service = DEFAULT_CONFIG[TEXT_CORRECTION_SERVICE_CONFIG_KEY]

        # Batch size and GPU index
        self.batch_size = DEFAULT_CONFIG[BATCH_SIZE_CONFIG_KEY]
        self.gpu_index = DEFAULT_CONFIG[GPU_INDEX_CONFIG_KEY]
        self.batch_size_specified = False
        self.gpu_index_specified = False

        # OpenRouter API configuration
        self.openrouter_api_key = DEFAULT_CONFIG[OPENROUTER_API_KEY_CONFIG_KEY]
        self.openrouter_model = DEFAULT_CONFIG[OPENROUTER_MODEL_CONFIG_KEY]
        self.openrouter_client = None

        # Gemini API configuration
        self.gemini_api_key = DEFAULT_CONFIG[GEMINI_API_KEY_CONFIG_KEY]
        self.gemini_model = DEFAULT_CONFIG[GEMINI_MODEL_CONFIG_KEY]
        self.gemini_client = None
        self.gemini_prompt = DEFAULT_CONFIG["gemini_prompt"]
        self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]
        self.gemini_general_prompt = DEFAULT_CONFIG["gemini_general_prompt"]
        self.sound_lock = RLock()  # Lock for sound playback

        # Reload key configuration
        self.reload_key = DEFAULT_CONFIG["reload_key"]

        # Keyboard library configuration - Apenas Win32 é suportado agora
        self.keyboard_library = KEYBOARD_LIB_WIN32

        # --- Keyboard Hotkey Manager ---
        self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
        self.ahk_running = False

        # Comentado devido a problemas com o hook de teclado no Windows
        # self.direct_manager = DirectHotkeyManager(config_file="hotkey_config.json")
        # self.direct_running = False

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
        # Keyboard library handlers
        self.hotkey_press_handler = None
        self.hotkey_release_handler = None
        # Pynput handlers
        self.pynput_keyboard_listener = None
        self.pynput_last_key = None
        self.pynput_is_pressed = False

        # --- Settings Window State ---
        self.settings_window_open = False # Keep track of settings window state

        # --- Periodic Re-registration ---
        self.reregister_timer_thread = None
        self.stop_reregister_event = threading.Event()

        # --- Hotkey Health Monitoring ---
        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()

        # --- Application State ---
        self.current_state = STATE_LOADING_MODEL # Initial state
        self.shutting_down = False # <<< FIX: Flag to prevent double shutdown

        # --- Initialization ---
        self._load_config()
        # Initialize API clients after loading config
        self._init_openrouter_client()
        self._init_gemini_client()
        self._start_model_loading()
        # Run startup cleanup here after config is loaded but before model/hotkeys
        self._cleanup_old_audio_files_on_startup()


    # --- OpenRouter API Integration ---
    def _init_openrouter_client(self):
        """Initialize the OpenRouter API client if enabled and available."""
        # Reset client first
        self.openrouter_client = None

        # Check if text correction is enabled and OpenRouter is selected
        if not self.text_correction_enabled or self.text_correction_service != SERVICE_OPENROUTER:
            logging.info("OpenRouter API is not selected for text correction.")
            return

        # Check if API key is provided
        if not self.openrouter_api_key:
            logging.info("OpenRouter API key not provided.")
            return

        if OpenRouterAPI is None:
            logging.warning("OpenRouterAPI module not available. OpenRouter text correction will be disabled.")
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

    # --- Gemini API Integration ---
    def _init_gemini_client(self):
        """Initialize the Gemini API client if enabled and available."""
        # Reset client first
        self.gemini_client = None

        # Check if text correction is enabled and Gemini is selected
        if not self.text_correction_enabled or self.text_correction_service != SERVICE_GEMINI:
            logging.info("Gemini API is not selected for text correction.")
            return

        # Check if API key is provided
        if not self.gemini_api_key:
            logging.info("Gemini API key not provided.")
            return

        if GeminiAPI is None:
            logging.warning("GeminiAPI module not available. Gemini text correction will be disabled.")
            return

        try:
            logging.info(f"Initializing Gemini API client with model: {self.gemini_model}")
            self.gemini_client = GeminiAPI(
                api_key=self.gemini_api_key,
                model_id=self.gemini_model,
                prompt=self.gemini_prompt
            )
            logging.info("Gemini API client initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing Gemini API client: {e}")
            self.gemini_client = None

    def _get_text_correction_service(self):
        """Returns the appropriate text correction service based on settings."""
        if not self.text_correction_enabled:
            return SERVICE_NONE

        if self.text_correction_service == SERVICE_OPENROUTER and self.openrouter_client:
            return SERVICE_OPENROUTER

        if self.text_correction_service == SERVICE_GEMINI and self.gemini_client:
            return SERVICE_GEMINI

        return SERVICE_NONE

    def _suggest_batch_size(self):
        """Suggests a batch size based on available GPU memory."""
        if not torch.cuda.is_available():
            return DEFAULT_CONFIG[BATCH_SIZE_CONFIG_KEY]

        try:
            if self.gpu_index >= torch.cuda.device_count():
                logging.warning(f"GPU index {self.gpu_index} out of range. Using GPU 0 for memory check.")
                self.gpu_index = 0
            torch.cuda.set_device(self.gpu_index)
            props = torch.cuda.get_device_properties(self.gpu_index)
            total_gb = props.total_memory / 1024**3
            if total_gb >= 20:
                bs = 32
            elif total_gb >= 12:
                bs = 16
            elif total_gb >= 8:
                bs = 8
            else:
                bs = 4
            logging.info(f"Suggested batch size {bs} for GPU with {total_gb:.2f} GB")
            return bs
        except Exception as e:
            logging.warning(f"Could not determine GPU memory for batch size suggestion: {e}")
            return DEFAULT_CONFIG[BATCH_SIZE_CONFIG_KEY]

    def _correct_text_with_openrouter(self, text):
        """Correct the transcribed text using OpenRouter API."""
        if not self.openrouter_client or not text:
            return text

        try:
            logging.info("Sending text to OpenRouter API for correction...")
            corrected_text = self.openrouter_client.correct_text(text)
            logging.info("Text correction completed successfully.")
            return corrected_text
        except Exception as e:
            logging.error(f"Error correcting text with OpenRouter API: {e}")
            return text  # Return original text on error

    def _correct_text_with_gemini(self, text):
        """Correct the transcribed text using Gemini API."""
        if not self.gemini_client or not text:
            return text

        try:
            logging.info("Sending text to Gemini API for correction...")
            if self.gemini_mode == "correction":
                prompt_to_use = self.gemini_prompt
            else: # Assuming "general" mode
                prompt_to_use = self.gemini_general_prompt
            corrected_text = self.gemini_client.correct_text(text, override_prompt=prompt_to_use)
            logging.info("Text correction completed successfully.")
            return corrected_text
        except Exception as e:
            logging.error(f"Error correcting text with Gemini API: {e}")
            return text  # Return original text on error

    # --- Configuration ---
    def _load_config(self):
        """Loads configuration from JSON file or uses defaults."""
        # Start with a fresh copy of defaults
        cfg = DEFAULT_CONFIG.copy()
        config_source = "defaults"
        loaded_config = {}

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
        self.gemini_prompt = str(self.config.get("gemini_prompt", DEFAULT_CONFIG["gemini_prompt"]))

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

        # Keyboard library - Apenas Win32 é suportado agora
        self.keyboard_library = KEYBOARD_LIB_WIN32
        self.config[KEYBOARD_LIBRARY_CONFIG_KEY] = self.keyboard_library

        # Text correction settings validation
        # Text correction enabled
        try:
            self.text_correction_enabled = bool(self.config.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY, DEFAULT_CONFIG[TEXT_CORRECTION_ENABLED_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.text_correction_enabled = DEFAULT_CONFIG[TEXT_CORRECTION_ENABLED_CONFIG_KEY]
            self.config[TEXT_CORRECTION_ENABLED_CONFIG_KEY] = self.text_correction_enabled

        # Text correction service
        try:
            service = str(self.config.get(TEXT_CORRECTION_SERVICE_CONFIG_KEY, DEFAULT_CONFIG[TEXT_CORRECTION_SERVICE_CONFIG_KEY]))
            if service in [SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI]:
                self.text_correction_service = service
            else:
                self.text_correction_service = DEFAULT_CONFIG[TEXT_CORRECTION_SERVICE_CONFIG_KEY]
        except (ValueError, TypeError):
            self.text_correction_service = DEFAULT_CONFIG[TEXT_CORRECTION_SERVICE_CONFIG_KEY]
            self.config[TEXT_CORRECTION_SERVICE_CONFIG_KEY] = self.text_correction_service

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

        # Gemini API key
        try:
            self.gemini_api_key = str(self.config.get(GEMINI_API_KEY_CONFIG_KEY, DEFAULT_CONFIG[GEMINI_API_KEY_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.gemini_api_key = DEFAULT_CONFIG[GEMINI_API_KEY_CONFIG_KEY]
            self.config[GEMINI_API_KEY_CONFIG_KEY] = self.gemini_api_key

        # Gemini model
        try:
            self.gemini_model = str(self.config.get(GEMINI_MODEL_CONFIG_KEY, DEFAULT_CONFIG[GEMINI_MODEL_CONFIG_KEY]))
        except (ValueError, TypeError):
            self.gemini_model = DEFAULT_CONFIG[GEMINI_MODEL_CONFIG_KEY]
            self.config[GEMINI_MODEL_CONFIG_KEY] = self.gemini_model

        # Batch size
        self.batch_size_specified = BATCH_SIZE_CONFIG_KEY in loaded_config
        try:
            bs_val = int(self.config.get(BATCH_SIZE_CONFIG_KEY, DEFAULT_CONFIG[BATCH_SIZE_CONFIG_KEY]))
            if bs_val <= 0:
                raise ValueError("Batch size must be positive")
            self.batch_size = bs_val
        except (ValueError, TypeError):
            self.batch_size = DEFAULT_CONFIG[BATCH_SIZE_CONFIG_KEY]
            self.config[BATCH_SIZE_CONFIG_KEY] = self.batch_size

        # GPU index
        self.gpu_index_specified = GPU_INDEX_CONFIG_KEY in loaded_config
        try:
            gpu_idx_val = int(self.config.get(GPU_INDEX_CONFIG_KEY, DEFAULT_CONFIG[GPU_INDEX_CONFIG_KEY]))
            if gpu_idx_val < 0:
                raise ValueError("GPU index must be >=0")
            self.gpu_index = gpu_idx_val
        except (ValueError, TypeError):
            self.gpu_index = DEFAULT_CONFIG[GPU_INDEX_CONFIG_KEY]
            self.config[GPU_INDEX_CONFIG_KEY] = self.gpu_index

        # Load and validate Gemini mode
        try:
            gemini_mode_val = str(self.config.get("gemini_mode", DEFAULT_CONFIG["gemini_mode"])).lower()
            if gemini_mode_val in ["correction", "general"]:
                self.gemini_mode = gemini_mode_val
            else:
                logging.warning(f"Invalid gemini_mode '{self.config.get('gemini_mode')}' in config. Falling back to '{DEFAULT_CONFIG['gemini_mode']}'.")
                self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid gemini_mode type in config. Falling back to '{DEFAULT_CONFIG['gemini_mode']}'.")
            self.gemini_mode = DEFAULT_CONFIG["gemini_mode"]


        # Load Gemini general prompt
        try:
            self.gemini_general_prompt = str(self.config.get("gemini_general_prompt", DEFAULT_CONFIG["gemini_general_prompt"]))
        except (ValueError, TypeError):
            logging.warning(f"Invalid gemini_general_prompt type in config. Falling back to default.")
            self.gemini_general_prompt = DEFAULT_CONFIG["gemini_general_prompt"]

        logging.info(f"Config source: {config_source}. Applied: Key='{self.record_key}', Mode='{self.record_mode}', AutoPaste={self.auto_paste}, MinDuration={self.min_record_duration}s")
        logging.info(f"Keyboard library: {self.keyboard_library}")
        logging.info(f"Text correction settings: Enabled={self.text_correction_enabled}, Service={self.text_correction_service}")
        logging.info(f"OpenRouter settings: Model={self.openrouter_model}")
        logging.info(f"Gemini settings: Model={self.gemini_model}")
        logging.info(f"Gemini mode: {self.gemini_mode}") # Added logging for new settings
        logging.info(f"Gemini general prompt: {self.gemini_general_prompt}") # Added logging for new settings
        logging.info(f"Batch size: {self.batch_size} (specified: {self.batch_size_specified})")
        logging.info(f"GPU index: {self.gpu_index} (specified: {self.gpu_index_specified})")

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
            # Keyboard library
            KEYBOARD_LIBRARY_CONFIG_KEY: self.keyboard_library,
            # Text correction settings
            TEXT_CORRECTION_ENABLED_CONFIG_KEY: self.text_correction_enabled,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY: self.text_correction_service,
            # Gemini Prompt setting
            "gemini_prompt": self.gemini_prompt,
            # OpenRouter API settings
            OPENROUTER_API_KEY_CONFIG_KEY: self.openrouter_api_key,
            OPENROUTER_MODEL_CONFIG_KEY: self.openrouter_model,
            # Gemini API settings
            GEMINI_API_KEY_CONFIG_KEY: self.gemini_api_key,
            GEMINI_MODEL_CONFIG_KEY: self.gemini_model,
            "gemini_mode": self.gemini_mode,
            "gemini_general_prompt": self.gemini_general_prompt,
            BATCH_SIZE_CONFIG_KEY: self.batch_size,
            GPU_INDEX_CONFIG_KEY: self.gpu_index
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

    # --- Sound Generation and Playback (New OutputStream based) ---
    def _generate_tone_data(self, frequency, duration, volume_factor):
        """Generates a sine wave tone with fade in/out.

        Args:
            frequency (int): Tone frequency in Hz.
            duration (float): Tone duration in seconds.
            volume_factor (float): Volume level (0.0 to 1.0).

        Returns:
            np.ndarray: NumPy array containing the tone samples (float32).
        """
        # Generate a sine wave
        num_samples = int(duration * AUDIO_SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, False)
        tone = np.sin(2 * np.pi * frequency * t) * volume_factor

        # Apply fade in/out to avoid clicks
        fade_samples = int(0.01 * AUDIO_SAMPLE_RATE)  # 10ms fade
        if fade_samples * 2 < len(tone):
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out

        return tone.astype(np.float32)

    class _TonePlaybackCallback:
        """Helper class to manage tone playback within an OutputStream callback."""
        def __init__(self, tone_data, finished_event):
            self.tone_data = tone_data
            self.read_offset = 0
            self.finished_event = finished_event

        def __call__(self, outdata, frames, time, status):
            if status:
                logging.warning(f"Tone playback callback status: {status}")

            remaining_samples = len(self.tone_data) - self.read_offset
            if remaining_samples == 0:
                # No more data, fill with zeros and stop
                outdata.fill(0)
                self.finished_event.set()
                raise sd.CallbackStop()

            # Determine how many samples to copy in this block
            chunk_size = min(frames, remaining_samples)

            # Copy data to outdata
            outdata[:chunk_size] = self.tone_data[self.read_offset : self.read_offset + chunk_size].reshape(-1, 1)
            
            # Fill the rest with zeros if chunk_size is less than frames
            if chunk_size < frames:
                outdata[chunk_size:].fill(0)
                self.finished_event.set()
                raise sd.CallbackStop()

            self.read_offset += chunk_size

    def _play_generated_tone_stream(self, frequency=None, duration=None, volume=None, is_start=True):
        """Plays a generated tone using sd.OutputStream.

        Args:
            frequency: Tone frequency in Hz. If None, uses the configured value.
            duration: Tone duration in seconds. If None, uses the configured value.
            volume: Volume level (0.0 to 1.0). If None, uses the configured value.
            is_start: If True, plays the start tone, otherwise plays the stop tone.
        """
        if not self.sound_enabled:
            logging.debug("Sound playback skipped (disabled in settings)")
            return

        freq = frequency if frequency is not None else self.sound_frequency
        dur = duration if duration is not None else self.sound_duration
        vol = volume if volume is not None else self.sound_volume

        if not is_start:
            freq = int(freq * 0.8)  # Lower pitch for stop tone

        # For reload sound, adjust frequency as per original _play_sound logic
        # This method will be called with specific freq/dur for reload, so no need for is_start check here.
        # The `is_start` parameter is primarily for the default start/stop tones.

        logging.debug(f"Attempting to play tone via OutputStream: {freq}Hz, {dur}s, vol={vol}")

        # Use a separate event to signal completion from the callback
        finished_event = threading.Event()
        stream = None
        try:
            with self.sound_lock:  # Protect sound device access
                # Generate the full tone data once
                tone_data = self._generate_tone_data(freq, dur, vol)

                # Create the callback instance
                callback_instance = self._TonePlaybackCallback(tone_data, finished_event)

                # Open the OutputStream
                stream = sd.OutputStream(
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    callback=callback_instance,
                    dtype='float32'
                )
                stream.start()
                logging.debug("OutputStream started for tone playback.")

            # Wait for the tone to finish playing (signaled by the callback)
            finished_event.wait()
            logging.debug("Tone playback finished (OutputStream).")

        except Exception as e:
            logging.error(f"Error playing tone via OutputStream: {e}", exc_info=True)
        finally:
            if stream is not None:
                try:
                    if stream.active:
                        stream.stop()
                    stream.close()
                    logging.debug("OutputStream stopped and closed.")
                except Exception as e:
                    logging.error(f"Error stopping/closing OutputStream: {e}")


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

            device_str_local = "cuda" if torch.cuda.is_available() else "cpu"
            device_param = device_str_local
            torch_dtype_local = torch.float16 if device_str_local == "cuda" else torch.float32

            logging.info(f"CUDA available: {torch.cuda.is_available()}")
            if device_str_local == "cuda":
                try:
                    if self.gpu_index >= torch.cuda.device_count():
                        logging.warning(f"GPU index {self.gpu_index} out of range. Using GPU 0.")
                        self.gpu_index = 0
                    torch.cuda.set_device(self.gpu_index)
                    props = torch.cuda.get_device_properties(self.gpu_index)
                    total_gb = props.total_memory / 1024**3
                    logging.info(f"Using GPU {self.gpu_index}: {props.name} ({total_gb:.2f} GB)")
                    if total_gb < 4:
                        logging.warning("GPU memory appears low (<4GB). Falling back to CPU.")
                        device_str_local = "cpu"
                        device_param = "cpu"
                        torch_dtype_local = torch.float32
                except Exception as e:
                    logging.error(f"Failed to select GPU {self.gpu_index}: {e}")
                    device_str_local = "cpu"
                    device_param = "cpu"
                    torch_dtype_local = torch.float32

            if not self.batch_size_specified:
                self.batch_size = self._suggest_batch_size()

            if not self.batch_size_specified:
                self.batch_size = self._suggest_batch_size()

            if device_str_local == "cuda":
                device_param = self.gpu_index
            loaded_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch_dtype_local,
                device=device_param
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
            error_message = f"Dependências do modelo ausentes. Por favor, verifique sua instalação. Erro: {e}"
            logging.error(f"Falha no carregamento do pipeline: {error_message}", exc_info=True)
        except torch.cuda.OutOfMemoryError as e:
            error_message = f"Memória CUDA insuficiente. Tente usar um modelo menor ou liberar recursos da GPU. Erro: {e}"
            logging.error(f"Falha no carregamento do pipeline: {error_message}", exc_info=True)
        except Exception as e:
            error_message = f"Falha ao carregar o modelo. Verifique sua conexão com a internet ou o nome do modelo. Erro: {e}"
            logging.error(f"Falha no carregamento do pipeline: {error_message}", exc_info=True)

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

        # Iniciar o KeyboardHotkeyManager e registrar callbacks
        self._start_autohotkey()

        logging.info("Hotkeys registered using keyboard library.")

    def _on_model_load_failed(self, error_msg):
         """Handles model loading failure."""
         logging.error(f"Model load failed: {error_msg}")
         self.pipe = None # Ensure pipe is None on failure
         self._set_state(STATE_ERROR_MODEL) # Set state to indicate error
         self._log_status(f"Erro: Falha ao carregar o modelo. {error_msg}", error=True)
         if hasattr(self, 'root') and self.root.winfo_exists():
             self.root.after(0, lambda: messagebox.showerror("Erro de Carregamento do Modelo", f"Falha ao carregar o modelo Whisper:\n{error_msg}\n\nPor favor, verifique sua conexão com a internet, o nome do modelo nas configurações ou a memória da sua GPU."))

    def _start_autohotkey(self):
        """Inicia o gerenciador de hotkeys e configura os callbacks."""
        with self.hotkey_lock: # Proteger operações de inicialização/configuração
            try:
                # Verificar se o KeyboardHotkeyManager já está em execução
                if self.ahk_running:
                    logging.info("KeyboardHotkeyManager já está em execução.")
                    return True

                # Atualizar a configuração do KeyboardHotkeyManager
                self.ahk_manager.update_config(
                    record_key=self.record_key,
                    reload_key=self.reload_key,
                    record_mode=self.record_mode
                )

                # Configurar callbacks
                self.ahk_manager.set_callbacks(
                    toggle=self.toggle_recording,
                    start=self.start_recording,
                    stop=self.stop_recording_if_needed,
                    reload=self.force_reregister_hotkeys
                )

                # Iniciar o KeyboardHotkeyManager
                success = self.ahk_manager.start()
                if success:
                    self.ahk_running = True
                    logging.info("KeyboardHotkeyManager iniciado com sucesso.")
                    self._log_status(f"Hotkey registrada: {self.record_key.upper()} (modo: {self.record_mode})")
                    return True
                else:
                    logging.error("Falha ao iniciar KeyboardHotkeyManager.")
                    self._set_state(STATE_ERROR_SETTINGS)
                    self._log_status("Erro: Falha ao iniciar KeyboardHotkeyManager.", error=True)
                    return False

            except Exception as e:
                logging.error(f"Erro ao iniciar KeyboardHotkeyManager: {e}", exc_info=True)
                messagebox.showerror("Erro de Hotkey", f"Falha ao iniciar o gerenciador de hotkeys: {e}", parent=settings_window_instance)
                self._set_state(STATE_ERROR_SETTINGS)
                self._log_status(f"Erro: {e}", error=True)
                return False


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

    # --- Pynput Handlers ---
    def _pynput_on_press(self, key):
        """Handler for pynput key press events."""
        try:
            # Convert the key to a string representation
            key_str = self._pynput_key_to_str(key)
            logging.debug(f"Pynput key press: {key_str}")

            # Check if this is our target key
            if key_str.lower() == self.record_key.lower():
                logging.info(f"Pynput detected record key press: {key_str}")

                # Verificar se a tecla já está sendo considerada como pressionada
                if self.pynput_is_pressed and self.pynput_last_key == key_str:
                    logging.warning(f"Key {key_str} already marked as pressed, ignoring duplicate press event")
                    return True  # Suppress the key to evitar comportamento estranho

                # Atualizar o estado da tecla
                self.pynput_last_key = key_str
                self.pynput_is_pressed = True

                # Handle based on record mode
                if self.record_mode == "toggle":
                    self._debounce_logic(self.toggle_recording)
                    return True  # Suppress the key
                elif self.record_mode == "press":
                    with self.recording_lock:
                        if self.is_recording: return True  # Already recording, suppress key
                    self._debounce_logic(self.start_recording)
                    return True  # Suppress the key

            # Check if this is our reload key
            if key_str.lower() == self.reload_key.lower():
                logging.info(f"Pynput detected reload key press: {key_str}")
                # Debounce to prevent multiple rapid calls
                current_time = time.time()
                if hasattr(self, 'last_reload_time') and current_time - self.last_reload_time < HOTKEY_DEBOUNCE_INTERVAL:
                    logging.debug(f"Ignoring reload hotkey press (debounce): {current_time - self.last_reload_time:.2f}s")
                    return False  # Don't suppress other keys

                self.last_reload_time = current_time
                logging.info(f"Reload hotkey pressed: {self.reload_key}")

                # Play a sound to indicate reload is happening
                if self.sound_enabled:
                    threading.Thread(target=self._play_generated_tone_stream, kwargs={"frequency": self.sound_frequency * 1.2, "duration": 0.15, "is_start": True}, daemon=True).start()
                    time.sleep(0.2)  # Small delay between sounds
                    threading.Thread(target=self._play_generated_tone_stream, kwargs={"frequency": self.sound_frequency * 1.2, "duration": 0.15, "is_start": True}, daemon=True).start()

                # Force re-register in a separate thread to avoid blocking
                threading.Thread(target=self.force_reregister_hotkeys, daemon=True, name="ReloadHotkeyThread").start()
                return True  # Suppress the key
        except Exception as e:
            logging.error(f"Error in pynput press handler: {e}", exc_info=True)

        return False  # Don't suppress other keys

    def _pynput_on_release(self, key):
        """Handler for pynput key release events."""
        try:
            # Convert the key to a string representation
            key_str = self._pynput_key_to_str(key)
            logging.debug(f"Pynput key release: {key_str}")

            # Check if this is our target key
            if key_str.lower() == self.record_key.lower():
                logging.info(f"Pynput detected record key release: {key_str}")

                # Verificar se a tecla estava realmente marcada como pressionada
                if not self.pynput_is_pressed or self.pynput_last_key != key_str:
                    logging.warning(f"Key {key_str} was not marked as pressed, but received release event")
                    # Ainda assim, vamos garantir que o estado seja limpo
                    self.pynput_is_pressed = False
                    self.pynput_last_key = None
                    return False  # Não suprimir a tecla neste caso

                # Limpar o estado da tecla
                self.pynput_is_pressed = False
                self.pynput_last_key = None

                # Handle based on record mode
                if self.record_mode == "press":
                    # No debounce needed on release, just stop if recording
                    threading.Thread(target=self.stop_recording_if_needed, daemon=True, name="StopRecordingThread").start()
                    return True  # Suppress the key
        except Exception as e:
            logging.error(f"Error in pynput release handler: {e}", exc_info=True)

        return False  # Don't suppress other keys

    def _pynput_key_to_str(self, key):
        """Convert a pynput key to a string representation."""
        try:
            # Handle special keys
            if hasattr(key, 'name'):
                return key.name.upper()
            # Handle character keys
            elif hasattr(key, 'char'):
                return key.char
            # Handle other keys
            else:
                return str(key)
        except Exception as e:
            logging.error(f"Error converting pynput key to string: {e}")
            return str(key)

    def register_hotkeys(self):
        """Clears old hooks and registers new ones based on current config."""
        # Primeiro, garantir que todos os hooks anteriores sejam removidos
        self._cleanup_hotkeys() # Unhook previous first
        time.sleep(0.2) # Aumentado o delay para garantir limpeza completa

        if not self.record_key:
            logging.error("Cannot register hotkey: record_key is empty.")
            self._set_state(STATE_ERROR_SETTINGS)
            self._log_status("Error: No record key set!", error=True)
            return

        logging.info(f"Attempting to register hotkeys: Key='{self.record_key}', Mode='{self.record_mode}', Library='{self.keyboard_library}'")

        # Apenas KeyboardHotkeyManager é suportado agora
        success = False

        # Usar apenas o KeyboardHotkeyManager
        try:
            # Iniciar o KeyboardHotkeyManager
            success = self._start_autohotkey()
            logging.info(f"KeyboardHotkeyManager registration result: {success}")
        except Exception as e:
            logging.error(f"KeyboardHotkeyManager registration failed: {e}", exc_info=True)
            success = False

        # Check if registration was successful
        if success:
            # If no exception, assume success
            self._log_status(f"Global hotkey registered: {self.record_key.upper()} (mode: {self.record_mode})")
            logging.info(f"Hotkey registration successful with KeyboardHotkeyManager")
            # Ensure state reflects idle state correctly after registration
            if self.current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
                 self._set_state(STATE_IDLE)
        else:
            # All registration attempts failed
            status_msg = f"Error: Hotkey registration failed with all libraries."
            self._set_state(STATE_ERROR_SETTINGS)
            self._log_status(status_msg, error=True)

    def _cleanup_hotkeys(self):
        """Unhooks all keyboard listeners."""
        logging.debug("Attempting to unhook keyboard listeners...")
        with self.keyboard_lock: # Protect keyboard operations
            # Clean up KeyboardHotkeyManager
            try:
                if self.ahk_running:
                    try:
                        # Primeiro, limpar o dicionário de handlers para evitar erros
                        if hasattr(self.ahk_manager, 'hotkey_handlers'):
                            self.ahk_manager.hotkey_handlers.clear()

                        # Agora parar o gerenciador
                        self.ahk_manager.stop()
                    except Exception as inner_e:
                        logging.error(f"Error stopping KeyboardHotkeyManager: {inner_e}")
                    finally:
                        # Garantir que o estado seja atualizado mesmo em caso de erro
                        self.ahk_running = False
                        logging.info("KeyboardHotkeyManager stopped.")
                        time.sleep(0.2)  # Pequeno delay para garantir que o processo foi encerrado
            except Exception as e:
                logging.error(f"Error during KeyboardHotkeyManager cleanup: {e}")

    def _register_reload_hotkey(self):
        """Registers the reload hotkey with KeyboardHotkeyManager."""
        # O hotkey de reload é gerenciado pelo KeyboardHotkeyManager
        # Não é necessário fazer nada adicional, pois o KeyboardHotkeyManager já registra
        # o hotkey de reload quando é iniciado no método _start_autohotkey
        logging.info(f"Reload hotkey is handled by KeyboardHotkeyManager: {self.reload_key}")
        return True

    def _reload_hotkey_handler(self):
        """Handler for the reload hotkey press."""
        # Debounce to prevent multiple rapid calls
        current_time = time.time()
        if hasattr(self, 'last_reload_time') and current_time - self.last_reload_time < HOTKEY_DEBOUNCE_INTERVAL:
            logging.debug(f"Ignoring reload hotkey press (debounce): {current_time - self.last_reload_time:.2f}s")
            return

        # Armazenar o tempo atual para debounce
        self.last_reload_time = current_time
        logging.info(f"Reload hotkey pressed: {self.reload_key}")

        # Play a sound to indicate reload is happening
        if self.sound_enabled:
            threading.Thread(target=self._play_generated_tone_stream, kwargs={"frequency": self.sound_frequency * 1.2, "duration": 0.15, "is_start": True}, daemon=True).start()
            time.sleep(0.2)  # Small delay between sounds
            threading.Thread(target=self._play_generated_tone_stream, kwargs={"frequency": self.sound_frequency * 1.2, "duration": 0.15, "is_start": True}, daemon=True).start()

        # Force re-register in a separate thread to avoid blocking
        threading.Thread(target=self.force_reregister_hotkeys, daemon=True, name="ReloadHotkeyThread").start()

    def _reload_keyboard_and_suppress(self):
        """Recarrega completamente o KeyboardHotkeyManager."""
        # Protege contra recarregamentos durante gravações
        with self.keyboard_lock: # Use the lock for the whole operation
            max_attempts = 3
            attempt = 0
            last_error = None

            # Cleanup all existing hotkeys first
            self._cleanup_hotkeys()
            time.sleep(0.3)  # Aumentado o tempo de espera para garantir limpeza completa

            while attempt < max_attempts:
                attempt += 1
                try:
                    logging.info(f"Tentativa {attempt}/{max_attempts} de recarregamento do KeyboardHotkeyManager...")

                    # Reiniciar o KeyboardHotkeyManager
                    if self.ahk_running:
                        self.ahk_manager.stop()
                        self.ahk_running = False
                        logging.info("KeyboardHotkeyManager parado para reinicialização.")
                        time.sleep(0.2)  # Pequeno delay para garantir que o processo foi encerrado

                    # Criar nova instância
                    self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
                    logging.info("Nova instância de KeyboardHotkeyManager criada.")

                    # Se chegou até aqui, o recarregamento foi bem-sucedido
                    logging.info("Recarregamento do KeyboardHotkeyManager concluído com sucesso.")
                    break
                except Exception as e:
                    last_error = e
                    logging.error(f"Erro na tentativa {attempt} de recarregamento: {e}")
                    time.sleep(1)  # Espera um pouco antes de tentar novamente

            if attempt >= max_attempts and last_error is not None:
                logging.error(f"Falha após {max_attempts} tentativas de recarregamento. Último erro: {last_error}")
                return False

            # Agora, registra os hotkeys novamente
            return self.register_hotkeys()

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
        """Manually attempts to re-register hotkeys if state allows. Returns True on success, False on failure.
        
        Uses self.hotkey_lock to ensure atomic operations on KeyboardHotkeyManager,
        preventing race conditions during hotkey re-registration.
        """
        with self.state_lock:
            current_state = self.current_state

        # Allow retry from any state except active recording/saving/loading
        if current_state not in [STATE_RECORDING, STATE_SAVING, STATE_LOADING_MODEL]:
            logging.info(f"Manual trigger: State is {current_state}. Attempting hotkey re-registration.")
            with self.hotkey_lock: # Proteger operações de hotkey
                try:
                    # Parar o KeyboardHotkeyManager atual
                    if self.ahk_running:
                        self.ahk_manager.stop()
                        self.ahk_running = False # Definir como False após o stop ser bem sucedido
                        time.sleep(0.5)  # Pequeno delay para garantir que o processo foi encerrado

                    # Atualizar a configuração e reiniciar o KeyboardHotkeyManager
                    self.ahk_manager.update_config(
                        record_key=self.record_key,
                        reload_key=self.reload_key,
                        record_mode=self.record_mode
                    )

                    # Configurar callbacks
                    self.ahk_manager.set_callbacks(
                        toggle=self.toggle_recording,
                        start=self.start_recording,
                        stop=self.stop_recording_if_needed,
                        reload=self.force_reregister_hotkeys # Mantém a capacidade de auto-chamada se necessário no futuro
                    )

                    success = self.ahk_manager.start()
                    if success:
                        self.ahk_running = True # Definir como True somente se start() for bem-sucedido
                        logging.info("Manual KeyboardHotkeyManager re-registration successful.")
                        # If successful, try to move back to IDLE state if we were in error
                        if current_state.startswith("ERROR"):
                            self._set_state(STATE_IDLE)
                        self._log_status("Recarregamento do KeyboardHotkeyManager concluído.", error=False)
                        return True # Indicate success
                    else:
                        # self.ahk_running já deve ser False ou não definido como True
                        logging.error("Manual KeyboardHotkeyManager re-registration failed.")
                        self._log_status("Falha ao recarregar KeyboardHotkeyManager.", error=True)
                        # Ensure state reflects error if reload failed
                        self._set_state(STATE_ERROR_SETTINGS)
                        return False # Indicate failure
                except Exception as e:
                    self.ahk_running = False # Garantir que ahk_running seja False em caso de exceção
                    logging.error(f"Exception during manual KeyboardHotkeyManager re-registration: {e}", exc_info=True)
                    self._log_status(f"Erro ao recarregar KeyboardHotkeyManager: {e}", error=True)
                    self._set_state(STATE_ERROR_SETTINGS) # Indicate an issue
                    return False # Indicate failure
        else:
            logging.warning(f"Manual trigger: Cannot re-register hotkeys. Current state is {current_state}.")
            self._log_status(f"Não é possível recarregar agora (Estado: {current_state}).", error=True)
            return False # Indicate failure (state not suitable)

    # --- Audio Recording ---
    def _audio_callback(self, indata, *_, **__):
        """Callback function for the sounddevice InputStream."""
        status = __.get('status')
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
        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": True}, daemon=True, name="StartSoundThread").start()

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
            threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": False}, daemon=True, name="StopSoundThread").start()

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
            # ATENÇÃO: Revertido para 'inputs' (parâmetro posicional) e removida a lógica de soundfile/attention_mask
            # devido a TypeError na versão atual do transformers.
            # A correção para 'input_features' e 'attention_mask' será reintroduzida
            # após a atualização da biblioteca transformers para uma versão compatível.
            result = self.pipe(audio_filename, chunk_length_s=30, batch_size=self.batch_size, return_timestamps=False)

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
                    # Apply text correction if enabled
                    correction_service = self._get_text_correction_service()

                    if correction_service == SERVICE_GEMINI:
                        try:
                            logging.info("Applying Gemini text correction...")
                            original_text = text_result
                            corrected_text = self._correct_text_with_gemini(text_result)
                            if corrected_text != original_text:
                                logging.info("Text was corrected by Gemini")
                                print("[INFO] Text was successfully corrected by Gemini API")
                                text_result = corrected_text
                            else:
                                logging.info("Gemini API returned text unchanged")
                                print("[INFO] Gemini API processed text but made no changes")
                        except Exception as e:
                            logging.error(f"Error during Gemini text correction: {e}")
                            print(f"[ERROR] Gemini text correction failed: {e}")
                            # Continue with original text on error

                    elif correction_service == SERVICE_OPENROUTER:
                        try:
                            logging.info("Applying OpenRouter text correction...")
                            original_text = text_result
                            corrected_text = self._correct_text_with_openrouter(text_result)
                            if corrected_text != original_text:
                                logging.info("Text was corrected by OpenRouter")
                                print("[INFO] Text was successfully corrected by OpenRouter API")
                                text_result = corrected_text
                            else:
                                logging.info("OpenRouter API returned text unchanged")
                                print("[INFO] OpenRouter API processed text but made no changes")
                        except Exception as e:
                            logging.error(f"Error during OpenRouter text correction: {e}")
                            print(f"[ERROR] OpenRouter text correction failed: {e}")
                            # Continue with original text on error

                    self._handle_transcription_result(text_result) # Handle copy/paste
                    self._set_state(STATE_IDLE) # Back to idle after success
                else: # No text or "[No speech detected]"
                     logging.warning(f"Processed {audio_filename} with no significant text.")
                     self._log_status("Transcription finished: No speech detected.", error=False) # Log as info
                     self._set_state(STATE_IDLE) # Back to idle

            # --- Cleanup ---
            self._delete_audio_file(audio_filename) # Delete file after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cleared GPU cache after transcription task.")


    # --- Settings Application Logic (Called from Settings Thread) ---
    def apply_settings_from_external(self, new_key, new_mode, new_auto_paste,
                                   new_sound_enabled=None, new_sound_frequency=None,
                                   new_sound_duration=None, new_sound_volume=None,
                                   new_reload_key=None,
                                   new_text_correction_enabled=None, new_text_correction_service=None,
                                   new_openrouter_api_key=None, new_openrouter_model=None,
                                   new_gemini_api_key=None, new_gemini_model=None,
                                   new_gemini_mode=None, new_gemini_prompt=None, new_gemini_general_prompt=None,
                                   new_batch_size=None, new_gpu_index=None):
        """Applies settings passed from the external settings window/thread."""
        logging.info("Applying new configuration from external source.")
        key_changed = False
        mode_changed = False
        config_needs_saving = False
        gemini_changed = False # Initialize gemini_changed here

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
            config_needs_saving = True
            logging.info(f"Auto paste changed to: {self.auto_paste}")

        # Apply sound settings
        if new_sound_enabled is not None:
            sound_enabled_bool = bool(new_sound_enabled)
            if sound_enabled_bool != self.sound_enabled:
                self.sound_enabled = sound_enabled_bool
                config_needs_saving = True
                logging.info(f"Sound enabled changed to: {self.sound_enabled}")

        if new_sound_frequency is not None:
            try:
                freq_val = int(new_sound_frequency)
                if 20 <= freq_val <= 20000 and freq_val != self.sound_frequency:
                    self.sound_frequency = freq_val
                    config_needs_saving = True
                    logging.info(f"Sound frequency changed to: {self.sound_frequency} Hz")
            except (ValueError, TypeError):
                logging.warning(f"Invalid sound frequency value: {new_sound_frequency}")

        if new_sound_duration is not None:
            try:
                dur_val = float(new_sound_duration)
                if 0.05 <= dur_val <= 2.0 and dur_val != self.sound_duration:
                    self.sound_duration = dur_val
                    config_needs_saving = True
                    logging.info(f"Sound duration changed to: {self.sound_duration} seconds")
            except (ValueError, TypeError):
                logging.warning(f"Invalid sound duration value: {new_sound_duration}")

        if new_sound_volume is not None:
            try:
                vol_val = float(new_sound_volume)
                if 0.0 <= vol_val <= 1.0 and vol_val != self.sound_volume:
                    self.sound_volume = vol_val
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

        # Batch size
        if new_batch_size is not None:
            try:
                bs_val = int(new_batch_size)
                if bs_val > 0 and bs_val != self.batch_size:
                    self.batch_size = bs_val
                    config_needs_saving = True
                    self.batch_size_specified = True
                    logging.info(f"Batch size changed to: {self.batch_size}")
            except (ValueError, TypeError):
                logging.warning(f"Invalid batch size value: {new_batch_size}")

        # GPU index
        if new_gpu_index is not None:
            try:
                gpu_idx_val = int(new_gpu_index)
                if gpu_idx_val >= 0 and gpu_idx_val != self.gpu_index:
                    self.gpu_index = gpu_idx_val
                    config_needs_saving = True
                    self.gpu_index_specified = True
                    logging.info(f"GPU index changed to: {self.gpu_index}")
            except (ValueError, TypeError):
                logging.warning(f"Invalid GPU index value: {new_gpu_index}")

        # Keyboard library is always Win32
        self.keyboard_library = KEYBOARD_LIB_WIN32

        # Apply text correction settings
        text_correction_changed = False

        # Text correction enabled
        if new_text_correction_enabled is not None:
            enabled_bool = bool(new_text_correction_enabled)
            if enabled_bool != self.text_correction_enabled:
                self.text_correction_enabled = enabled_bool
                text_correction_changed = True
                config_needs_saving = True
                logging.info(f"Text correction enabled changed to: {self.text_correction_enabled}")

        # Text correction service
        if new_text_correction_service is not None:
            service_str = str(new_text_correction_service)
            if service_str in [SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI] and service_str != self.text_correction_service:
                self.text_correction_service = service_str
                text_correction_changed = True
                config_needs_saving = True
                logging.info(f"Text correction service changed to: {self.text_correction_service}")

        # Apply OpenRouter settings if provided
        openrouter_changed = False
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

        # Apply Gemini settings if provided
        gemini_changed = False
        if new_gemini_api_key is not None:
            api_key_str = str(new_gemini_api_key)
            if api_key_str != self.gemini_api_key:
                self.gemini_api_key = api_key_str
                gemini_changed = True
                config_needs_saving = True
                logging.info("Gemini API key updated")

        if new_gemini_model is not None:
            model_str = str(new_gemini_model)
            if model_str != self.gemini_model:
                self.gemini_model = model_str
                gemini_changed = True
                config_needs_saving = True
                logging.info(f"Gemini model changed to: {self.gemini_model}")

        # Apply Gemini mode
        if new_gemini_mode is not None:
            mode_str = str(new_gemini_mode).lower()
            if mode_str in ["correction", "general"] and mode_str != self.gemini_mode:
                self.gemini_mode = mode_str
                gemini_changed = True
                config_needs_saving = True
                logging.info(f"Gemini mode changed to: {self.gemini_mode}")

        # Apply Gemini correction prompt
        if new_gemini_prompt is not None:
            prompt_str = str(new_gemini_prompt)
            if prompt_str != self.gemini_prompt:
                self.gemini_prompt = prompt_str
                gemini_changed = True
                config_needs_saving = True
                logging.info("Gemini correction prompt updated.")

        # Apply Gemini general prompt
        if new_gemini_general_prompt is not None:
            general_prompt_str = str(new_gemini_general_prompt)
            if general_prompt_str != self.gemini_general_prompt:
                self.gemini_general_prompt = general_prompt_str
                gemini_changed = True
                config_needs_saving = True
                logging.info("Gemini general prompt updated.")

        # Reinitialize API clients if settings changed
        if text_correction_changed or openrouter_changed:
            self._init_openrouter_client()

        if text_correction_changed or gemini_changed:
            # Reinitialize Gemini client if any relevant setting changed
            if new_gemini_api_key is not None and new_gemini_api_key != self.gemini_api_key:
                gemini_changed = True
            if new_gemini_model is not None and new_gemini_model != self.gemini_model:
                gemini_changed = True
            if new_gemini_mode is not None and new_gemini_mode != self.gemini_mode:
                gemini_changed = True
            if new_gemini_prompt is not None and new_gemini_prompt != self.gemini_prompt:
                gemini_changed = True
            if new_gemini_general_prompt is not None and new_gemini_general_prompt != self.gemini_general_prompt:
                gemini_changed = True

            if gemini_changed:
                self._init_gemini_client()

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

    def _hotkey_health_check_task(self):
        """Thread task that periodically verifica a saúde das bibliotecas de hotkeys e alterna entre elas se necessário."""
        logging.info("Hotkey health monitoring thread started.")

        while not self.stop_health_check_event.wait(HOTKEY_HEALTH_CHECK_INTERVAL):
            # wait() returns True if the event was set (stop requested), False on timeout
            # So the loop continues as long as wait() returns False (timeout occurred)

            # Verificar se o aplicativo está em um estado que permite verificação
            with self.state_lock:
                current_state = self.current_state

            if current_state in [STATE_IDLE, STATE_TRANSCRIBING]:
                logging.debug(f"Verificando saúde das bibliotecas de hotkeys. Estado atual: {current_state}")

                # Verificar se o KeyboardHotkeyManager está funcionando corretamente
                if not self.ahk_running:
                    logging.warning("KeyboardHotkeyManager não está em execução. Tentando reiniciar.")
                    self.force_reregister_hotkeys()
                    self._log_status("Tentativa de reiniciar KeyboardHotkeyManager.", error=False)
                else:
                    logging.debug("KeyboardHotkeyManager está funcionando corretamente.")
            else:
                logging.debug(f"Pulando verificação de saúde das hotkeys. Estado atual: {current_state}")

        logging.info("Hotkey health monitoring thread stopped.")

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
        try:
            self.stop_reregister_event.set()
        except Exception as e:
            logging.error(f"Error signaling re-register thread to stop: {e}")

        # 2. Stop KeyboardHotkeyManager - usando _cleanup_hotkeys que já tem tratamento de erros melhorado
        try:
            logging.info("Stopping KeyboardHotkeyManager...")
            self._cleanup_hotkeys()
        except Exception as e:
            logging.error(f"Error during hotkey cleanup in shutdown: {e}")

        # 3. Stop recording if active (don't save)
        try:
            with self.recording_lock: # Ensure lock is used
                if self.is_recording:
                    logging.warning("Recording active during shutdown. Forcing stop...")
                    self.is_recording = False # Signal recording thread to stop
                    # Try to close stream gracefully if it exists
                    stream = self.audio_stream
                    if stream:
                        try:
                            if hasattr(stream, 'active') and stream.active:
                                stream.stop()
                                stream.close()
                                logging.info("Audio stream stopped and closed during shutdown.")
                        except Exception as e:
                            logging.error(f"Error stopping stream on close: {e}")

                    try:
                        self.recording_data.clear()
                    except Exception as e:
                        logging.error(f"Error clearing recording data: {e}")

                    self.audio_stream = None # Clear reference
        except Exception as e:
            logging.error(f"Error stopping recording during shutdown: {e}")

        # 4. Stop any running transcription? Difficult to interrupt cleanly. Log instead.
        try:
            with self.transcription_lock: # Ensure lock is used
                if self.transcription_in_progress:
                    logging.warning("Shutting down while transcription is in progress. Transcription may not complete.")
                    # We can't easily kill the transformers pipeline thread.
        except Exception as e:
            logging.error(f"Error checking transcription status during shutdown: {e}")

        # 5. Wait briefly for the timer thread to exit (optional but good practice)
        try:
            if self.reregister_timer_thread and self.reregister_timer_thread.is_alive():
                logging.debug("Waiting for periodic re-register thread to stop...")
                self.reregister_timer_thread.join(timeout=1.5) # Wait slightly longer
                if self.reregister_timer_thread.is_alive():
                    logging.warning("Periodic re-register thread did not stop gracefully.")
        except Exception as e:
            logging.error(f"Error waiting for timer thread during shutdown: {e}")

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
    """Runs the CustomTkinter settings window GUI in its own mainloop (futuristic, compact, interactive)."""
    global core_instance, settings_window_instance, settings_thread_running
    import customtkinter as ctk
    import tkinter.messagebox as messagebox
    import logging
    import threading
    import time

    # Definir a flag settings_thread_running como True imediatamente no início da thread
    settings_thread_running = True
    logging.info("SettingsGUIThread started, settings_thread_running set to True.")

    # Garantir que a flag settings_thread_running seja redefinida quando a função terminar
    try:
        # Verificação de segurança para evitar múltiplas instâncias
        settings_window_exists = False
        try:
            settings_window_exists = settings_window_instance is not None and settings_window_instance.winfo_exists()
        except Exception:
            # Se ocorrer qualquer erro na verificação, assumimos que a janela não existe
            settings_window_instance = None

        if settings_window_exists:
            logging.warning("Settings window already exists. Not creating a new one.")
            try:
                settings_window_instance.lift()
                settings_window_instance.focus_force()
                # A flag settings_thread_running já foi definida como True no início.
                # Se a janela já existe e foi focada, não precisamos redefini-la para False aqui,
                # pois a thread continua ativa e a janela está sendo usada.
                # A limpeza para False ocorrerá no finally ou em close_settings quando a janela for fechada.
                return
            except Exception as e:
                logging.warning(f"Could not focus existing settings window: {e}")
                # Se não conseguir focar a janela existente, vamos criar uma nova
                settings_window_instance = None
    except Exception as e:
        logging.error(f"Error checking for existing settings window: {e}")
        # Em caso de erro muito precoce, garantir que a flag seja limpa
        settings_thread_running = False
        return

    # Apply dark theme and blue accent
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Create a temporary hidden root for this instance of the settings window
    temp_tk_root = ctk.CTk()
    temp_tk_root.withdraw()

    def on_temp_root_close():
        global settings_window_instance
        # Sempre definir como None, independentemente de erros
        settings_window_instance = None

        # Verificação mais segura para evitar erros de threading
        root_exists = False
        try:
            root_exists = temp_tk_root is not None and temp_tk_root.winfo_exists()
        except Exception as e:
            logging.debug(f"Error checking if temp_tk_root exists: {e}")

        if root_exists:
            try:
                temp_tk_root.destroy()
            except Exception as e:
                logging.warning(f"Error destroying temp_tk_root: {e}")
    temp_tk_root.protocol("WM_DELETE_WINDOW", on_temp_root_close)

    # Create Toplevel as child of the temporary root
    try:
        settings_win = ctk.CTkToplevel(temp_tk_root)
        settings_window_instance = settings_win
    except Exception as e:
        logging.error(f"Failed to create Toplevel for settings: {e}", exc_info=True)
        on_temp_root_close()
        return

    # --- Configure the Toplevel window (INDENTED under the try) ---
    settings_win.title("Whisper Recorder Settings") # Already English
    settings_win.resizable(False, True)
    settings_win.attributes("-topmost", True)

    # --- Calculate Center Position ---
    settings_win.update_idletasks()
    window_width = int(SETTINGS_WINDOW_GEOMETRY.split('x')[0])
    window_height = int(SETTINGS_WINDOW_GEOMETRY.split('x')[1])
    screen_width = settings_win.winfo_screenwidth()
    screen_height = settings_win.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    logging.info(f"Centering settings window at {x_cordinate}, {y_cordinate}")
    settings_win.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

    # --- Variables ---
    auto_paste_var = ctk.BooleanVar(value=core_instance.auto_paste)
    mode_var = ctk.StringVar(value=core_instance.record_mode)
    detected_key_var = ctk.StringVar(value=core_instance.record_key.upper())
    new_record_key_temp = None
    reload_key_var = ctk.StringVar(value=core_instance.reload_key.upper())
    new_reload_key_temp = None
    sound_enabled_var = ctk.BooleanVar(value=core_instance.sound_enabled)
    sound_frequency_var = ctk.IntVar(value=core_instance.sound_frequency)
    sound_duration_var = ctk.DoubleVar(value=core_instance.sound_duration)
    sound_volume_var = ctk.DoubleVar(value=core_instance.sound_volume)
    text_correction_enabled_var = ctk.BooleanVar(value=core_instance.text_correction_enabled)
    text_correction_service_var = ctk.StringVar(value=core_instance.text_correction_service)
    openrouter_api_key_var = ctk.StringVar(value=core_instance.openrouter_api_key)
    openrouter_model_var = ctk.StringVar(value=core_instance.openrouter_model)
    gemini_api_key_var = ctk.StringVar(value=core_instance.gemini_api_key)
    gemini_model_var = ctk.StringVar(value=core_instance.gemini_model)
    gemini_mode_var = ctk.StringVar(value=core_instance.gemini_mode) # Variável para o modo Gemini
    batch_size_var = ctk.IntVar(value=core_instance.batch_size)
    gpu_index_var = ctk.IntVar(value=core_instance.gpu_index)
    # keyboard_library_var removida pois não é mais usada

    # Function to toggle visibility of Gemini prompt widgets
    def toggle_gemini_prompt_visibility(*args):
        selected_mode = gemini_mode_var.get()
        try:
            # Esconder todos os widgets de prompt primeiro para evitar sobreposição ou ordem errada
            # Correction prompt widgets
            if 'gemini_prompt_correction_frame' in locals() and gemini_prompt_correction_frame.winfo_exists():
                gemini_prompt_correction_frame.pack_forget()
            if 'restore_correction_prompt_button' in locals() and restore_correction_prompt_button.winfo_exists():
                restore_correction_prompt_button.pack_forget()

            # General prompt widgets
            # General prompt widgets
            if 'gemini_prompt_general_frame' in locals() and gemini_prompt_general_frame.winfo_exists():
                gemini_prompt_general_frame.pack_forget()
            # O restore_button é empacotado separadamente, então sua visibilidade é controlada abaixo

            # Exibir apenas os widgets relevantes para o modo selecionado
            if selected_mode == "correction":
                if 'gemini_prompt_correction_frame' in locals() and gemini_prompt_correction_frame.winfo_exists():
                    gemini_prompt_correction_frame.pack(fill="x", padx=0, pady=(5, 0))
                if 'restore_correction_prompt_button' in locals() and restore_correction_prompt_button.winfo_exists():
                    restore_correction_prompt_button.pack(anchor="w", padx=5, pady=(0, 10))
                # Esconder o botão de restauração do prompt geral no modo correção
                if 'restore_button' in locals() and restore_button.winfo_exists():
                    restore_button.pack_forget()

            elif selected_mode == "general":
                if 'gemini_prompt_general_frame' in locals() and gemini_prompt_general_frame.winfo_exists():
                    gemini_prompt_general_frame.pack(fill="x", padx=0, pady=(5, 0))
                # Exibir o botão de restauração do prompt geral no modo geral
                if 'restore_button' in locals() and restore_button.winfo_exists():
                    restore_button.pack(anchor="w", padx=5, pady=(0, 10))

        except NameError as e: # Captura caso os widgets não estejam definidos ainda
            logging.debug(f"toggle_gemini_prompt_visibility: Widget not defined yet during initial call - {e}")
        except Exception as e:
            logging.error(f"Error in toggle_gemini_prompt_visibility: {e}")

    # Trace changes to gemini_mode_var
    gemini_mode_var.trace_add("write", toggle_gemini_prompt_visibility)
 
    # --- Functions ---
    detect_key_thread = None  # Keep track of the detection thread
 
    def detect_key_task_internal():
        """Internal task to detect key press, runs in a thread."""
        nonlocal new_record_key_temp
        detected_key_str = "ERROR"
        new_record_key_temp = None
        logging.info("Detect key task started (in detect thread).")

        # Schedule button update on the Tkinter thread using temp_tk_root.after
        if settings_window_instance and settings_window_instance.winfo_exists():
            # Use configure for CTkButton
            temp_tk_root.after(0, lambda: detect_key_button.configure(text="PRESS KEY...", state="disabled"))
        else:
            logging.warning("Settings window closed before starting key detection UI update.")
            return

        try:
            if core_instance:
                logging.info("Unhooking global hotkeys for detection (from detect thread)...")
                with core_instance.keyboard_lock:
                    # Garantir que todos os hooks sejam removidos antes da detecção
                    core_instance._cleanup_hotkeys()
                    # Resetar estado das teclas
                    core_instance.pynput_last_key = None
                    core_instance.pynput_is_pressed = False
                    core_instance.hotkey_press_handler = None
                    core_instance.hotkey_release_handler = None

            # Aguardar um pouco mais para garantir que todos os hooks foram removidos
            time.sleep(0.3)

            logging.info("Waiting for key event...")
            # Usar o KeyboardHotkeyManager para detectar a tecla
            detected_key = None
            try:
                # Pausar o KeyboardHotkeyManager atual
                if core_instance and core_instance.ahk_running and core_instance.ahk_manager:
                    core_instance.ahk_manager.stop()
                    core_instance.ahk_running = False
                    time.sleep(0.2)  # Aguardar para garantir que o processo foi encerrado

                # Detectar a tecla com KeyboardHotkeyManager
                logging.info("Tentando detectar tecla com KeyboardHotkeyManager...")
                from keyboard_hotkey_manager import KeyboardHotkeyManager
                temp_manager = KeyboardHotkeyManager()
                detected_key = temp_manager.detect_key()
                temp_manager.stop()

                # Reiniciar os gerenciadores de hotkeys originais
                if core_instance:
                    logging.info("Restaurando hotkeys após detecção...")
                    core_instance.register_hotkeys()
            except Exception as e:
                logging.error(f"Erro ao detectar tecla com KeyboardHotkeyManager: {e}")
                detected_key = None

            if detected_key:
                # Verificar se a tecla detectada é válida
                if len(detected_key) > 0:
                    new_record_key_temp = detected_key.lower()
                    detected_key_str = new_record_key_temp.upper()
                    logging.info(f"Tecla válida detectada: {detected_key_str}")
                else:
                    logging.warning("Combinação de tecla vazia detectada")
                    detected_key_str = "INVALID KEY"
            else:
                detected_key_str = "DETECTION FAILED"
        except Exception as e:
            logging.error(f"Error detecting key: {e}", exc_info=True)
            detected_key_str = "ERROR"
            new_record_key_temp = None
            if core_instance:
                logging.error("Re-registering original hotkeys after detection error...")
                try:
                    core_instance.register_hotkeys()
                except Exception as reg_error:
                    logging.error(f"Error re-registering hotkeys: {reg_error}")
        finally:
            if settings_window_instance and settings_window_instance.winfo_exists():
                temp_tk_root.after(0, lambda: update_detection_ui(detected_key_str))
            logging.info("Detect key task finished (in detect thread).")

    def update_detection_ui(key_text, is_reload_key=False):
        """Updates the key label and button state (runs in Tkinter thread via after())."""
        if settings_window_instance and settings_window_instance.winfo_exists():
            if is_reload_key:
                reload_key_var.set(key_text)
                try:
                    # Use configure for CTkButton
                    detect_reload_key_button.configure(text="Detect Key", state="normal")
                except Exception:
                    logging.warning("detect_reload_key_button not found or not a valid widget during UI update.")
            else:
                detected_key_var.set(key_text)
                try:
                    # Use configure for CTkButton
                    detect_key_button.configure(text="Detect Key", state="normal")
                except Exception:
                    logging.warning("detect_key_button not found or not a valid widget during UI update.")
        else:
            logging.warning("Settings window closed before UI update for key detection.")

    def start_detect_key():
        """Starts the key detection thread."""
        nonlocal detect_key_thread
        if detect_key_thread and detect_key_thread.is_alive():
            logging.warning("Key detection thread already running.")
            return
        if settings_window_instance and settings_window_instance.winfo_exists():
            detected_key_var.set("PRESS KEY...")
        detect_key_thread = threading.Thread(target=detect_key_task_internal, daemon=True, name="DetectKeyThread")
        detect_key_thread.start()

    def detect_reload_key_task_internal():
        """Internal task to detect reload key press, runs in a thread."""
        nonlocal new_reload_key_temp
        detected_key_str = "ERROR"
        new_reload_key_temp = None
        logging.info("Detect reload key task started (in detect thread).")

        if settings_window_instance and settings_window_instance.winfo_exists():
            # Use configure for CTkButton
            temp_tk_root.after(0, lambda: detect_reload_key_button.configure(text="PRESS KEY...", state="disabled"))
        else:
            logging.warning("Settings window closed before starting reload key detection UI update.")
            return

        try:
            if core_instance:
                logging.info("Unhooking global hotkeys for reload key detection...")
                with core_instance.keyboard_lock:
                    # Garantir que todos os hooks sejam removidos antes da detecção
                    core_instance._cleanup_hotkeys()
                    # Resetar estado das teclas
                    core_instance.pynput_last_key = None
                    core_instance.pynput_is_pressed = False
                    core_instance.hotkey_press_handler = None
                    core_instance.hotkey_release_handler = None

            # Aguardar um pouco mais para garantir que todos os hooks foram removidos
            time.sleep(0.3)

            logging.info("Waiting for reload key event...")
            # Usar o KeyboardHotkeyManager para detectar a tecla
            detected_key = None
            try:
                # Pausar o KeyboardHotkeyManager atual
                with core_instance.hotkey_lock:
                    if core_instance and core_instance.ahk_running and core_instance.ahk_manager:
                        core_instance.ahk_manager.stop()
                        core_instance.ahk_running = False
                        time.sleep(0.2)  # Aguardar para garantir que o processo foi encerrado

                # Detectar a tecla com KeyboardHotkeyManager
                logging.info("Tentando detectar tecla com KeyboardHotkeyManager...")
                temp_manager = KeyboardHotkeyManager()
                detected_key = temp_manager.detect_key()
                temp_manager.stop()

                # Reiniciar os gerenciadores de hotkeys originais
                with core_instance.hotkey_lock:
                    if core_instance:
                        logging.info("Restaurando hotkeys após detecção...")
                        core_instance.register_hotkeys()
            except Exception as e:
                logging.error(f"Erro ao detectar tecla com KeyboardHotkeyManager: {e}", exc_info=True)
                messagebox.showerror("Erro de Detecção de Tecla", f"Falha ao detectar a tecla de atalho: {e}", parent=settings_window_instance)
                detected_key = None

            # Verificar se a tecla detectada é válida
            if detected_key:
                if len(detected_key) > 0:
                    # Verificar se a tecla de reload é diferente da tecla de gravação
                    if detected_key.lower() == core_instance.record_key.lower():
                        logging.warning(f"Reload key cannot be the same as record key: {detected_key}")
                        detected_key_str = "SAME AS RECORD KEY"
                    else:
                        new_reload_key_temp = detected_key.lower()
                        detected_key_str = new_reload_key_temp.upper()
                        logging.info(f"Tecla válida detectada: {detected_key_str}")
                else:
                    logging.warning("Combinação de tecla vazia detectada")
                    detected_key_str = "INVALID KEY"
            else:
                detected_key_str = "DETECTION FAILED"
        except Exception as e:
            logging.error(f"Error detecting reload key: {e}", exc_info=True)
            messagebox.showerror("Erro de Detecção de Tecla", f"Ocorreu um erro inesperado durante a detecção da tecla de recarga: {e}", parent=settings_window_instance)
            detected_key_str = "ERROR"
            new_reload_key_temp = None
            with core_instance.hotkey_lock:
                if core_instance:
                    logging.error("Re-registering original hotkeys after reload key detection error...")
                    try:
                        core_instance.register_hotkeys()
                    except Exception as reg_error:
                        logging.error(f"Error re-registering hotkeys: {reg_error}")
        finally:
            if settings_window_instance and settings_window_instance.winfo_exists():
                temp_tk_root.after(0, lambda: update_detection_ui(detected_key_str, is_reload_key=True))
            logging.info("Detect reload key task finished (in detect thread).")

    def start_detect_reload_key():
        """Starts the reload key detection thread."""
        nonlocal detect_key_thread
        if detect_key_thread and detect_key_thread.is_alive():
            logging.warning("Key detection thread already running.")
            return
        if settings_window_instance and settings_window_instance.winfo_exists():
            reload_key_var.set("PRESS KEY...")
        detect_key_thread = threading.Thread(target=detect_reload_key_task_internal, daemon=True, name="DetectReloadKeyThread")
        detect_key_thread.start()

    def apply_settings():
        """Applies the selected settings by calling the core instance method."""
        nonlocal new_record_key_temp, new_reload_key_temp
        logging.info("Apply settings clicked (in Tkinter thread).")

        if core_instance.pipe is None:
            messagebox.showwarning("Apply Settings", "Model not loaded yet. Cannot apply.", parent=settings_win) # Already English
            return
        with core_instance.recording_lock:
            if core_instance.is_recording:
                messagebox.showwarning("Apply Settings", "Cannot apply while recording.", parent=settings_win) # Already English
                return
        with core_instance.transcription_lock:
            if core_instance.transcription_in_progress:
                messagebox.showwarning("Apply Settings", "Cannot apply while transcribing.", parent=settings_win) # Already English
                return

        key_to_apply = new_record_key_temp
        mode_to_apply = mode_var.get()
        auto_paste_to_apply = auto_paste_var.get()
        reload_key_to_apply = new_reload_key_temp

        sound_enabled_to_apply = sound_enabled_var.get()

        try:
            sound_freq_to_apply = int(sound_frequency_var.get())
            if not (20 <= sound_freq_to_apply <= 20000):
                messagebox.showwarning("Invalid Value", "Frequency must be between 20 and 20000 Hz", parent=settings_win) # Already English
                return
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Value", "Frequency must be a number", parent=settings_win) # Already English
            return

        try:
            sound_duration_to_apply = float(sound_duration_var.get())
            if not (0.05 <= sound_duration_to_apply <= 2.0):
                messagebox.showwarning("Invalid Value", "Duration must be between 0.05 and 2.0 seconds", parent=settings_win) # Already English
                return
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Value", "Duration must be a number", parent=settings_win) # Already English
            return

        try:
            sound_volume_to_apply = float(sound_volume_var.get())
            if not (0.0 <= sound_volume_to_apply <= 1.0):
                messagebox.showwarning("Invalid Value", "Volume must be between 0.0 and 1.0", parent=settings_win) # Already English
                return
        except (ValueError, TypeError):
            messagebox.showwarning("Invalid Value", "Volume must be a number", parent=settings_win) # Already English
            return


        try:
            if hasattr(core_instance, 'apply_settings_from_external'):
                core_instance.apply_settings_from_external(
                    # Pass all relevant settings to the core instance method
                    new_key=key_to_apply,
                    new_mode=mode_to_apply,
                    new_auto_paste=auto_paste_to_apply,
                    new_sound_enabled=sound_enabled_to_apply,
                    new_sound_frequency=sound_freq_to_apply,
                    new_sound_duration=sound_duration_to_apply,
                    new_sound_volume=sound_volume_to_apply,
                    new_reload_key=reload_key_to_apply,
                    new_text_correction_enabled=text_correction_enabled_var.get(),
                    new_text_correction_service=text_correction_service_var.get(),
                    new_openrouter_api_key=openrouter_api_key_var.get(),
                    new_openrouter_model=openrouter_model_var.get(),
                    new_gemini_api_key=gemini_api_key_var.get(),
                    new_gemini_model=gemini_model_var.get(),
                    new_batch_size=batch_size_var.get(),
                    new_gpu_index=gpu_index_var.get()
                ) # Fechar parênteses da chamada da função
            else:
                logging.critical("CRITICAL: apply_settings_from_external method not found on core_instance!")
                messagebox.showerror("Internal Error", "Cannot apply settings: Core method missing.", parent=settings_win) # Already English
                return
        except Exception as e:
            logging.error(f"Error calling apply_settings_from_external from settings thread: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to apply settings:\n{e}", parent=settings_win) # Already English
            return

        new_record_key_temp = None
        new_reload_key_temp = None # Resetar também a variável temporária da reload key
        close_settings()

    def close_settings():
        """Closes the settings window and its temporary root (runs in Tkinter thread)."""
        global settings_window_instance
        logging.info("Settings window closing sequence started (in Tkinter thread).")

        # Verificação mais segura para evitar erros de threading
        settings_window_exists = False
        try:
            settings_window_exists = settings_window_instance is not None and settings_window_instance.winfo_exists()
        except Exception as e:
            logging.debug(f"Error checking if settings window exists during close: {e}")

        if settings_window_exists:
            logging.debug("Destroying settings Toplevel window...")
            try:
                settings_win.destroy()
            except Exception as e:
                logging.warning(f"Error destroying settings window: {e}")
        else:
            logging.warning("Tried to close settings window, but instance or window no longer exists.")


        # Definir settings_window_instance como None após a destruição
        settings_window_instance = None

        # Destruir temp_tk_root apenas se ele existir e for válido
        if temp_tk_root and temp_tk_root.winfo_exists():
            logging.debug("Destroying temporary Tk root for settings window...")
            try:
                temp_tk_root.destroy()
            except Exception as e:
                logging.warning(f"Error destroying temp root: {e}")


        logging.info("Settings window closing sequence finished (in Tkinter thread).")

        # Garantir que a flag settings_thread_running seja redefinida
        global settings_thread_running
        settings_thread_running = False

    # --- UI Construction ---
    settings_win.protocol("WM_DELETE_WINDOW", lambda: close_settings())

    # Main frame with scrollable content
    main_frame = ctk.CTkFrame(settings_win, fg_color="#222831", corner_radius=16)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Scrollable frame for settings
    scrollable = ctk.CTkScrollableFrame(main_frame, fg_color="#222831", corner_radius=16)
    scrollable.pack(fill="both", expand=True, padx=0, pady=0)

    # --- Auto Paste Section ---
    auto_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    auto_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(auto_frame, text="Auto Paste Text", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(side="left", padx=(0, 15)) # Already English
    ctk.CTkSwitch(auto_frame, text="Enabled", variable=auto_paste_var, onvalue=True, offvalue=False).pack(side="left") # Already English

    # --- Record Hotkey Section ---
    key_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    key_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(key_frame, text="Record Hotkey", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(side="left", padx=(0, 15)) # Already English
    key_display_label = ctk.CTkLabel(key_frame, textvariable=detected_key_var, font=("Segoe UI", 12, "bold"), width=120, fg_color="#393E46", text_color="#00a0ff", corner_radius=8)
    key_display_label.pack(side="left", padx=5)
    detect_key_button = ctk.CTkButton(key_frame, text="Detect Key", command=lambda: start_detect_key(), width=100, fg_color="#00a0ff", hover_color="#0078d7") # Already English
    detect_key_button.pack(side="left", padx=5)

    # --- Reload Hotkey Section ---
    reload_key_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    reload_key_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(reload_key_frame, text="Reload Hotkey", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(side="left", padx=(0, 15)) # Already English
    reload_key_display_label = ctk.CTkLabel(reload_key_frame, textvariable=reload_key_var, font=("Segoe UI", 12, "bold"), width=120, fg_color="#393E46", text_color="#00a0ff", corner_radius=8)
    reload_key_display_label.pack(side="left", padx=5)
    detect_reload_key_button = ctk.CTkButton(reload_key_frame, text="Detect Key", command=lambda: start_detect_reload_key(), width=100, fg_color="#00a0ff", hover_color="#0078d7") # Already English
    detect_reload_key_button.pack(side="left", padx=5)

    # --- Record Mode Section ---
    mode_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    mode_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(mode_frame, text="Record Mode", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(side="left", padx=(0, 15)) # Already English
    ctk.CTkRadioButton(mode_frame, text="Toggle", variable=mode_var, value="toggle").pack(side="left", padx=5) # Already English
    ctk.CTkRadioButton(mode_frame, text="Press/Hold", variable=mode_var, value="press").pack(side="left", padx=5) # Already English

    # Seção de biblioteca de teclado removida, pois não é configurável pelo usuário

    # --- Sound Settings Section ---
    sound_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    sound_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(sound_section_frame, text="Sound Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5) # Already English
    ctk.CTkSwitch(sound_section_frame, text="Enable Sound Feedback", variable=sound_enabled_var, onvalue=True, offvalue=False).pack(anchor="w", padx=5, pady=(5, 0)) # Already English
    # Volume
    ctk.CTkLabel(sound_section_frame, text="Volume", font=("Segoe UI", 11), text_color="#fff").pack(anchor="w", padx=5) # Already English
    ctk.CTkSlider(sound_section_frame, variable=sound_volume_var, from_=0.0, to=1.0, number_of_steps=100).pack(fill="x", padx=10)
    # Frequency
    freq_row = ctk.CTkFrame(sound_section_frame, fg_color="#222831")
    freq_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(freq_row, text="Frequency (Hz):", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkEntry(freq_row, textvariable=sound_frequency_var, width=80).pack(side="left", padx=5)
    # Duration
    dur_row = ctk.CTkFrame(sound_section_frame, fg_color="#222831")
    dur_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(dur_row, text="Duration (s):", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkEntry(dur_row, textvariable=sound_duration_var, width=80).pack(side="left", padx=5)
    # Test Sound
    def play_test_sound():
        if not core_instance:
            messagebox.showerror("Error", "Core instance not available", parent=settings_win) # Already English
            return
        enabled = sound_enabled_var.get()
        if not enabled:
            messagebox.showinfo("Sound Test", "Sound is disabled. Enable it first to test.", parent=settings_win) # Already English
            return
        try:
            freq = int(sound_frequency_var.get())
            duration = float(sound_duration_var.get())
            volume = float(sound_volume_var.get())
            if not (20 <= freq <= 20000):
                messagebox.showwarning("Invalid Value", "Frequency must be between 20 and 20000 Hz", parent=settings_win) # Already English
                return
            if not (0.05 <= duration <= 2.0):
                messagebox.showwarning("Invalid Value", "Duration must be between 0.05 and 2.0 seconds", parent=settings_win) # Already English
                return
            if not (0.0 <= volume <= 1.0):
                messagebox.showwarning("Invalid Value", "Volume must be between 0.0 and 1.0", parent=settings_win) # Already English
                return
            threading.Thread(
                target=core_instance._play_generated_tone_stream,
                kwargs={"frequency": freq, "duration": duration, "volume": volume, "is_start": True},
                daemon=True,
                name="TestSoundThread"
            ).start()
        except (ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Invalid sound settings: {e}", parent=settings_win) # Already English
    ctk.CTkButton(sound_section_frame, text="Test Sound", command=play_test_sound, fg_color="#00a0ff", hover_color="#0078d7", width=120).pack(anchor="w", padx=5, pady=(5, 0)) # Already English

    # --- Text Correction Section ---
    text_correction_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    text_correction_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(text_correction_section_frame, text="Text Correction Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5)
    ctk.CTkSwitch(text_correction_section_frame, text="Enable Text Correction", variable=text_correction_enabled_var, onvalue=True, offvalue=False).pack(anchor="w", padx=5, pady=(5, 0))
    # Service selection
    service_row = ctk.CTkFrame(text_correction_section_frame, fg_color="#222831")
    service_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(service_row, text="Service:", width=120).pack(side="left", padx=5)
    ctk.CTkRadioButton(service_row, text="None", variable=text_correction_service_var, value=SERVICE_NONE).pack(side="left", padx=5)
    ctk.CTkRadioButton(service_row, text="OpenRouter", variable=text_correction_service_var, value=SERVICE_OPENROUTER).pack(side="left", padx=5)
    ctk.CTkRadioButton(service_row, text="Gemini", variable=text_correction_service_var, value=SERVICE_GEMINI).pack(side="left", padx=5)



    # --- Gemini API Section ---
    gemini_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    gemini_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(gemini_section_frame, text="Gemini API Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5) # Already English
    gemini_api_row = ctk.CTkFrame(gemini_section_frame, fg_color="#222831")
    gemini_api_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(gemini_api_row, text="API Key:", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkEntry(gemini_api_row, textvariable=gemini_api_key_var).pack(side="left", fill="x", expand=True, padx=5)
    gemini_model_row = ctk.CTkFrame(gemini_section_frame, fg_color="#222831")
    gemini_model_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(gemini_model_row, text="Model:", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkEntry(gemini_model_row, textvariable=gemini_model_var).pack(side="left", fill="x", expand=True, padx=5)
 
    gemini_prompt_correction_var = ctk.StringVar(value=core_instance.gemini_prompt) # Variable for correction prompt

    # Correction Prompt Textbox and Label
    # Correction Prompt Label and Textbox
    # Frame para o rótulo e textbox de correção
    gemini_prompt_correction_frame = ctk.CTkFrame(gemini_section_frame, fg_color="#222831")
    # O frame será empacotado em toggle_gemini_prompt_visibility

    # Rótulo para o prompt de correção (empacotado acima do textbox)
    ctk.CTkLabel(gemini_prompt_correction_frame, text="Correction Prompt:", font=("Segoe UI", 12), text_color="#fff").pack(anchor="w", padx=5, pady=(5, 0)) # Empacotar acima

    # Textbox para o prompt de correção
    gemini_prompt_correction_textbox = ctk.CTkTextbox(gemini_prompt_correction_frame, width=500, height=150, wrap="word")
    gemini_prompt_correction_textbox.pack(fill="x", expand=True, padx=5, pady=(0, 5)) # Empacotar abaixo do rótulo, preenchendo horizontalmente

    # Load initial correction prompt value
    gemini_prompt_correction_textbox.insert("0.0", core_instance.gemini_prompt)

    # Add Restore Default Correction Prompt button
    def restore_default_correction_prompt():
        logging.info("Restoring Gemini correction prompt to default.")
        gemini_prompt_correction_textbox.delete("1.0", "end") # CORRIGIDO
        gemini_prompt_correction_textbox.insert("0.0", DEFAULT_CONFIG["gemini_prompt"]) # CORRIGIDO

    restore_correction_prompt_button = ctk.CTkButton(gemini_section_frame, text="Restore Default Correction Prompt", command=restore_default_correction_prompt, width=250, fg_color="#00a0ff", hover_color="#0078d7")
    # .pack() will be called in toggle_gemini_prompt_visibility

    # --- Gemini Mode Section ---
    gemini_mode_row = ctk.CTkFrame(gemini_section_frame, fg_color="#222831")
    gemini_mode_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(gemini_mode_row, text="Gemini Mode:", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkRadioButton(gemini_mode_row, text="Correction", variable=gemini_mode_var, value="correction").pack(side="left", padx=5) # Already English
    ctk.CTkRadioButton(gemini_mode_row, text="General", variable=gemini_mode_var, value="general").pack(side="left", padx=5) # Already English
 
 
 
 
 
    # General Prompt Label and Textbox
    # Frame para o rótulo e textbox geral
    gemini_prompt_general_frame = ctk.CTkFrame(gemini_section_frame, fg_color="#222831")
    # O frame será empacotado em toggle_gemini_prompt_visibility

    # Rótulo para o prompt geral (empacotado acima do textbox)
    ctk.CTkLabel(gemini_prompt_general_frame, text="General Prompt:", font=("Segoe UI", 12), text_color="#fff").pack(anchor="w", padx=5, pady=(5, 0)) # Empacotar acima

    # Textbox para o prompt geral
    gemini_prompt_textbox = ctk.CTkTextbox(gemini_prompt_general_frame, width=500, height=150, wrap="word")
    gemini_prompt_textbox.pack(fill="x", expand=True, padx=5, pady=(0, 5)) # Empacotar abaixo do rótulo, preenchendo horizontalmente

    # Load current General Prompt text
    gemini_prompt_textbox.insert("0.0", core_instance.gemini_general_prompt)

    # Function to restore default Gemini prompt (for "Load Correction Prompt as Base" button)
    def restore_gemini_prompt():
        logging.info("Restoring Gemini prompt to default.")
        gemini_prompt_textbox.delete("1.0", "end")
        gemini_prompt_textbox.insert("0.0", DEFAULT_CONFIG["gemini_prompt"])

    # Add Restore Default button
    restore_button = ctk.CTkButton(gemini_section_frame, text="Load Correction Prompt as Base", command=restore_gemini_prompt, width=120, fg_color="#00a0ff", hover_color="#0078d7") # Renamed for clarity

    # --- GPU Settings Section ---
    gpu_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    gpu_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(gpu_section_frame, text="GPU Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5)
    gpu_index_row = ctk.CTkFrame(gpu_section_frame, fg_color="#222831")
    gpu_index_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(gpu_index_row, text="GPU Index:", width=120).pack(side="left", padx=5)
    ctk.CTkEntry(gpu_index_row, textvariable=gpu_index_var, width=80).pack(side="left", padx=5)
    batch_row = ctk.CTkFrame(gpu_section_frame, fg_color="#222831")
    batch_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(batch_row, text="Batch Size:", width=120).pack(side="left", padx=5)
    ctk.CTkEntry(batch_row, textvariable=batch_size_var, width=80).pack(side="left", padx=5)

    # --- OpenRouter API Section ---
    openrouter_section_frame = ctk.CTkFrame(scrollable, fg_color="#222831", corner_radius=12)
    openrouter_section_frame.pack(fill="x", pady=(0, 10), padx=0)
    ctk.CTkLabel(openrouter_section_frame, text="OpenRouter API Settings", font=("Segoe UI", 13, "bold"), text_color="#00a0ff").pack(anchor="w", padx=5) # Already English
    openrouter_api_row = ctk.CTkFrame(openrouter_section_frame, fg_color="#222831")
    openrouter_api_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(openrouter_api_row, text="API Key:", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkEntry(openrouter_api_row, textvariable=openrouter_api_key_var).pack(side="left", fill="x", expand=True, padx=5)
    openrouter_model_row = ctk.CTkFrame(openrouter_section_frame, fg_color="#222831")
    openrouter_model_row.pack(fill="x", padx=0, pady=(5, 0))
    ctk.CTkLabel(openrouter_model_row, text="Model:", width=120).pack(side="left", padx=5) # Already English
    ctk.CTkEntry(openrouter_model_row, textvariable=openrouter_model_var).pack(side="left", fill="x", expand=True, padx=5)
 
    # --- Action Buttons ---
    # Note: The button_frame was defined earlier in the original code but is placed outside the scrollable frame in CTk
    button_frame = ctk.CTkFrame(settings_win, fg_color="#222831", corner_radius=12) # Recreate outside scrollable
    button_frame.pack(side="bottom", fill="x", padx=10, pady=10)
    ctk.CTkButton(button_frame, text="Apply", command=lambda: apply_settings(), width=120, fg_color="#00a0ff", hover_color="#0078d7").pack(side="right", padx=5) # Already English
    ctk.CTkButton(button_frame, text="Cancel", command=lambda: close_settings(), width=120, fg_color="#393E46", hover_color="#444444").pack(side="right", padx=5) # Already English
 
    # --- Force UI Update before Making Visible ---
    settings_win.update_idletasks()
    settings_win.update()
 
    # --- Make Visible and Start Mainloop ---
    settings_win.transient(temp_tk_root)
    settings_win.deiconify()
    settings_win.lift()
    settings_win.focus_force()
 
    # Set initial visibility of Gemini prompt based on loaded config
    toggle_gemini_prompt_visibility()

    logging.info("Starting mainloop for settings window thread...")
    try:
        temp_tk_root.mainloop()
    except Exception as e:
        logging.error(f"Error during settings window mainloop: {e}", exc_info=True)
    finally:
        logging.info("Settings window thread mainloop finished.")
        # Ensure cleanup happens even if mainloop crashes (close_settings handles this now)

# --- Callbacks for pystray Menu Items ---
def on_start_recording_menu_click(*_):
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

def on_stop_recording_menu_click(*_):
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
def on_toggle_recording_menu_click(*_):
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


# Variável de bloqueio para evitar múltiplas janelas de configurações
settings_window_lock = threading.Lock()
settings_thread_running = False

def on_settings_menu_click(*_):
    """Starts the settings GUI in a separate thread."""
    global settings_window_instance, settings_thread_running

    # Usar um lock para evitar condições de corrida ao verificar/criar a janela
    with settings_window_lock:
        # 1. Verificar settings_window_instance e sua atividade (winfo_exists)
        if settings_window_instance and settings_window_instance.winfo_exists():
            try:
                settings_window_instance.lift()
                settings_window_instance.focus_force()
                logging.info("Focused existing settings window.")
                return # Retorna imediatamente se a janela já existe e foi focada
            except Exception as e:
                logging.warning(f"Could not focus existing settings window: {e}. Attempting to create new one.")
                settings_window_instance = None # Resetar para tentar criar uma nova

        # 2. Verificar settings_thread_running
        if settings_thread_running:
            logging.warning("Settings window creation might be in progress by another thread or a previous attempt failed to clean up. Ignoring request.")
            return # Retorna para evitar iniciar outra thread concorrente

        # 3. Se chegou aqui, pode iniciar uma nova thread
        # A responsabilidade de definir settings_thread_running = True será da própria run_settings_gui
        logging.info("Starting settings window thread...")
        settings_thread = threading.Thread(target=run_settings_gui, daemon=True, name="SettingsGUIThread")
        settings_thread.start()

# --- NEW: Callback for Force Re-register Menu Item ---
def on_force_reregister_menu_click(*_):
    """Forces a reload of the keyboard library and hotkey re-registration."""
    global core_instance
    logging.info("Force keyboard/hotkey reload requested from tray menu.")
    if core_instance:
        if hasattr(core_instance, 'force_reregister_hotkeys'):
            # Run the core logic. It logs success/failure internally.
            # Consider running in a thread if it blocks pystray for too long,
            # but let's try direct call first for simplicity.
            core_instance.force_reregister_hotkeys()
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
def create_dynamic_menu(_):
    """Creates the tray icon menu dynamically based on recording state."""
    global core_instance # Ensure core_instance is accessible

    # Default menu if core_instance is not ready
    if not core_instance:
        logging.warning("create_dynamic_menu called but core_instance is None.")
        return (pystray.MenuItem('Loading...', None, enabled=False),)
    
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

    default_action_text = 'Loading...'
    default_action_callback = None
    default_enabled = False # Disable by default

    if is_recording:
        default_action_text = '⏹️ Stop Recording'
        default_action_callback = on_toggle_recording_menu_click
        default_enabled = True
    elif is_idle:
        default_action_text = '▶️ Start Recording'
        default_action_callback = on_toggle_recording_menu_click
        default_enabled = True
    elif is_loading:
        default_action_text = 'Loading Model...'
    elif is_transcribing:
        default_action_text = 'Transcribing...'
    elif is_saving:
        default_action_text = 'Saving...'
    elif is_error:
        default_action_text = 'Error (Check Logs/⚙️)'
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
            '⚙️ Settings',
            on_settings_menu_click,
            enabled=(not is_loading and not is_recording) # Allow settings unless loading/recording
        ),
        # --- NEW: Force Reload Item ---
        pystray.MenuItem(
            '🔄 Reload Keyboard/Hotkey',
            on_force_reregister_menu_click,
            enabled=can_force_reload # Enable based on state
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem('❌ Exit', on_exit_app)
    ]

    return tuple(menu_items) # Return as a tuple

# --- Tray Exit Function ---
def on_exit_app(*_):
    """Callback to cleanly exit the application."""
    global core_instance, tray_icon
    logging.info("Exit requested from tray icon.")

    # Limpar variáveis Tkinter primeiro

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
    # Register atexit handlers
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
