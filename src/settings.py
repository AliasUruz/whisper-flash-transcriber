import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

# --- Constants & Enums ---

class AppState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ERROR = "error"
    LOADING = "loading"
    SHUTDOWN = "shutdown"

# Supported Gemini models (in order of preference)
# Last updated: 2025-12-22
VALID_MODELS = [
    "gemini-3-flash-preview",   # Latest, best quality (released 2025-12-17)
    "gemini-2.5-flash-lite"     # Fallback, faster and cheaper
]

DEFAULT_PROMPT = (
    "Correct the text's punctuation and grammar without altering its meaning. "
    "Make it more expressive where appropriate, remove unnecessary repetitions, "
    "and improve flow. Combine sentences that make sense together. "
    "Maintain the original language and tone."
)

@dataclass
class AppSettings:
    hotkey: str = "f3"
    mouse_hotkey: bool = False
    auto_paste: bool = True
    input_device_index: Optional[int] = None
    model_path: str = ""
    
    # AI Settings
    gemini_enabled: bool = False
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_prompt: str = DEFAULT_PROMPT
    gemini_api_key: str = "" # In memory only, separated on disk
    
    # Sound Settings
    sound_enabled: bool = True
    sound_volume: int = 50
    sound_freq_start: int = 800
    sound_freq_stop: int = 500
    
    # Processing
    append_space: bool = False
    
    # Internal
    last_device_strategy: Optional[str] = None
    first_run: bool = True

class SettingsManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SettingsManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        logging.info("Initializing SettingsManager (Dataclass Edition)...")
        self.config_path = Path.home() / ".whisper_flash_transcriber" / "config.json"
        self.secrets_path = Path.home() / ".whisper_flash_transcriber" / "secrets.json"
        
        # Ensure directory exists
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create config directory: {e}")
        
        self.settings: AppSettings = AppSettings()
        self._settings_lock = threading.Lock()
        
        self.load()
        self._initialized = True

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load settings from {path}: {e}")
            return {}

    def load(self):
        """Loads settings from disk, merging public config and secrets."""
        with self._settings_lock:
            # Load raw dicts
            public_config = self._load_json(self.config_path)
            secrets = self._load_json(self.secrets_path)
            
            # Merge secrets into public config for internal object creation
            combined_data = public_config.copy()
            if "gemini_api_key" in secrets:
                combined_data["gemini_api_key"] = secrets["gemini_api_key"]

            # Validation / Migration logic could go here
            if combined_data.get("gemini_model") not in VALID_MODELS:
                # If invalid, keep default from dataclass or set specifically
                if "gemini_model" in combined_data:
                    logging.warning(f"Invalid model '{combined_data['gemini_model']}'. Reverting to default.")
                    del combined_data["gemini_model"]

            # Update dataclass fields
            # We iterate over the dataclass fields to only load known keys
            current_dict = asdict(self.settings)
            
            for key, value in combined_data.items():
                if key in current_dict:
                    setattr(self.settings, key, value)
            
    def get_settings(self) -> AppSettings:
        """Returns the typed settings object."""
        return self.settings

    def update(self, **kwargs):
        """Update specific settings safely."""
        with self._settings_lock:
            for key, value in kwargs.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                else:
                    logging.warning(f"Attempted to set unknown setting: {key}")
            self._save_to_disk()

    def save(self):
        """Force save to disk."""
        with self._settings_lock:
            self._save_to_disk()

    def _save_to_disk(self):
        try:
            data = asdict(self.settings)
            
            # Separate secrets
            secrets_data = {
                "gemini_api_key": data.pop("gemini_api_key", "")
            }
            
            # Save public
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            # Save secrets
            with open(self.secrets_path, "w", encoding="utf-8") as f:
                json.dump(secrets_data, f, indent=2)
                
            logging.info("Settings saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
