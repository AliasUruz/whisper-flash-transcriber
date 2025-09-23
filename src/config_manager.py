import os
import json
import logging
import copy
import hashlib
import time
from pathlib import Path
from typing import Any, List

import requests
import tkinter.messagebox as messagebox

from .model_manager import list_catalog, list_installed
try:
    from distutils.util import strtobool
except Exception:  # Python >= 3.12
    from setuptools._distutils.util import strtobool


def _parse_bool(value):
    """Converte diferentes representações de booleanos em objetos ``bool``."""
    if isinstance(value, str):
        try:
            return bool(strtobool(value))
        except ValueError:
            return bool(value)
    return bool(value)


# --- Constantes de Configuração (movidas de whisper_tkinter.py) ---
CONFIG_FILE = "config.json"
SECRETS_FILE = "secrets.json" # Nova constante para o arquivo de segredos

DEFAULT_CONFIG = {
    "record_key": "F3",
    "record_mode": "toggle",
    "auto_paste": True,
    "min_record_duration": 0.5,
    "sound_enabled": True,
    "sound_frequency": 400,
    "sound_duration": 0.3,
    "sound_volume": 0.5,
    "agent_key": "F4",
    "keyboard_library": "win32",
    "text_correction_enabled": False,
    "text_correction_service": "none",
    "openrouter_api_key": "",
    "openrouter_model": "deepseek/deepseek-chat-v3-0324:free",
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash-lite",
    "gemini_agent_model": "gemini-2.5-flash-lite",
    "openrouter_timeout": 30,
    "gemini_timeout": 120,
    "ai_provider": "gemini",
    "openrouter_prompt": "",
    "prompt_agentico": (
        "You are an AI assistant that executes text commands. "
        "The user will provide an instruction followed by the text to be processed. "
        "Your task is to execute the instruction on the text and return ONLY the final result. "
        "Do not add explanations, greetings, or any extra text. "
        "The output language should match the main language of the provided text."
    ),
    "gemini_prompt": (
        "You are a meticulous speech-to-text correction AI. "
        "Your primary task is to correct punctuation, capitalization, and minor transcription errors in the text below "
        "while preserving the original content and structure as closely as possible. "
        "Key instructions: - Correct punctuation, such as adding commas, periods, and question marks. "
        "- Fix capitalization at the beginning of sentences. "
        "- Remove only obvious speech disfluencies (e.g., \"I-I mean\"). "
        "- DO NOT summarize, paraphrase, or change the original meaning. "
        "- Return ONLY the corrected text, with no additional comments or explanations. "
        "Transcribed speech: {text}"
    ),
    "batch_size": 16, # Valor padrão para o modo automático
    "batch_size_mode": "auto", # Novo: 'auto' ou 'manual'
    "manual_batch_size": 8, # Novo: Valor para o modo manual
    "gpu_index": 0,
    "hotkey_stability_service_enabled": True, # Nova configuração unificada
    "use_vad": False,
    "vad_threshold": 0.5,
    # Duração máxima da pausa preservada antes que o silêncio seja descartado
    "vad_silence_duration": 1.0,
    "display_transcripts_in_terminal": False,
    "gemini_model_options": [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ],
    "save_temp_recordings": False,
    "record_storage_mode": "auto",
    "record_storage_limit": 0,
    "max_memory_seconds_mode": "manual",
    "max_memory_seconds": 30,
    "min_free_ram_mb": 1000,
    "auto_ram_threshold_percent": 10,
    "min_transcription_duration": 1.0, # Nova configuração
    "chunk_length_sec": 30,
    "chunk_length_mode": "manual",
    "enable_torch_compile": False,
    "launch_at_startup": False,
    "clear_gpu_cache": True,
    "asr_model_id": "openai/whisper-large-v3-turbo",
    "asr_backend": "transformers",
    "asr_compute_device": "auto",
    "asr_dtype": "float16",
    "asr_ct2_compute_type": "default",
    "asr_cache_dir": str((Path.home() / ".cache" / "whisper_flash_transcriber" / "asr").expanduser()),
    "asr_installed_models": [],
    "asr_curated_catalog": [],
    "asr_curated_catalog_url": "",
    "asr_last_download_status": {
        "status": "unknown",
        "timestamp": "",
        "model_id": "",
        "backend": "",
        "message": "",
        "details": "",
    },
    "asr_last_prompt_decision": {
        "model_id": "",
        "backend": "",
        "decision": "",
        "timestamp": 0,
    },
}

# Outras constantes de configuração (movidas de whisper_tkinter.py)
LAST_MODEL_PROMPT_DECISION_CONFIG_KEY = "asr_last_prompt_decision"
MIN_RECORDING_DURATION_CONFIG_KEY = "min_record_duration"
MIN_TRANSCRIPTION_DURATION_CONFIG_KEY = "min_transcription_duration"
AGENT_KEY_CONFIG_KEY = "agent_key"
SOUND_ENABLED_CONFIG_KEY = "sound_enabled"
SOUND_FREQUENCY_CONFIG_KEY = "sound_frequency"
SOUND_DURATION_CONFIG_KEY = "sound_duration"
SOUND_VOLUME_CONFIG_KEY = "sound_volume"
HOTKEY_STABILITY_SERVICE_ENABLED_CONFIG_KEY = "hotkey_stability_service_enabled" # Nova constante unificada
BATCH_SIZE_CONFIG_KEY = "batch_size" # Agora é o batch size padrão para o modo auto
BATCH_SIZE_MODE_CONFIG_KEY = "batch_size_mode" # Novo
MANUAL_BATCH_SIZE_CONFIG_KEY = "manual_batch_size" # Novo
GPU_INDEX_CONFIG_KEY = "gpu_index"
SAVE_TEMP_RECORDINGS_CONFIG_KEY = "save_temp_recordings"
RECORD_STORAGE_MODE_CONFIG_KEY = "record_storage_mode"
RECORD_STORAGE_LIMIT_CONFIG_KEY = "record_storage_limit"
MAX_MEMORY_SECONDS_MODE_CONFIG_KEY = "max_memory_seconds_mode"
DISPLAY_TRANSCRIPTS_KEY = "display_transcripts_in_terminal"
USE_VAD_CONFIG_KEY = "use_vad"
VAD_THRESHOLD_CONFIG_KEY = "vad_threshold"
VAD_SILENCE_DURATION_CONFIG_KEY = "vad_silence_duration"
CHUNK_LENGTH_SEC_CONFIG_KEY = "chunk_length_sec"
LAUNCH_AT_STARTUP_CONFIG_KEY = "launch_at_startup"
DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY = DISPLAY_TRANSCRIPTS_KEY
KEYBOARD_LIBRARY_CONFIG_KEY = "keyboard_library"
KEYBOARD_LIB_WIN32 = "win32"
TEXT_CORRECTION_ENABLED_CONFIG_KEY = "text_correction_enabled"
TEXT_CORRECTION_SERVICE_CONFIG_KEY = "text_correction_service"
ENABLE_AI_CORRECTION_CONFIG_KEY = TEXT_CORRECTION_ENABLED_CONFIG_KEY
SERVICE_NONE = "none"
SERVICE_OPENROUTER = "openrouter"
SERVICE_GEMINI = "gemini"
OPENROUTER_API_KEY_CONFIG_KEY = "openrouter_api_key"
OPENROUTER_MODEL_CONFIG_KEY = "openrouter_model"
OPENROUTER_TIMEOUT_CONFIG_KEY = "openrouter_timeout"
GEMINI_API_KEY_CONFIG_KEY = "gemini_api_key"
GEMINI_MODEL_CONFIG_KEY = "gemini_model"
GEMINI_AGENT_MODEL_CONFIG_KEY = "gemini_agent_model"
GEMINI_MODEL_OPTIONS_CONFIG_KEY = "gemini_model_options"
# Novas constantes de timeout de APIs externas
GEMINI_TIMEOUT_CONFIG_KEY = "gemini_timeout"
# Novas constantes para otimizações de desempenho
CHUNK_LENGTH_MODE_CONFIG_KEY = "chunk_length_mode"
ENABLE_TORCH_COMPILE_CONFIG_KEY = "enable_torch_compile"
AI_PROVIDER_CONFIG_KEY = TEXT_CORRECTION_SERVICE_CONFIG_KEY
GEMINI_AGENT_PROMPT_CONFIG_KEY = "prompt_agentico"
OPENROUTER_PROMPT_CONFIG_KEY = "openrouter_prompt"
OPENROUTER_AGENT_PROMPT_CONFIG_KEY = "openrouter_agent_prompt"
GEMINI_PROMPT_CONFIG_KEY = "gemini_prompt"
SETTINGS_WINDOW_GEOMETRY = "550x700"
REREGISTER_INTERVAL_SECONDS = 60
MAX_HOTKEY_FAILURES = 3
HOTKEY_HEALTH_CHECK_INTERVAL = 10
CLEAR_GPU_CACHE_CONFIG_KEY = "clear_gpu_cache"
ASR_BACKEND_CONFIG_KEY = "asr_backend"
ASR_MODEL_ID_CONFIG_KEY = "asr_model_id"
ASR_COMPUTE_DEVICE_CONFIG_KEY = "asr_compute_device"
ASR_DTYPE_CONFIG_KEY = "asr_dtype"
ASR_CT2_COMPUTE_TYPE_CONFIG_KEY = "asr_ct2_compute_type"
ASR_CT2_CPU_THREADS_CONFIG_KEY = "asr_ct2_cpu_threads"
ASR_CACHE_DIR_CONFIG_KEY = "asr_cache_dir"
ASR_INSTALLED_MODELS_CONFIG_KEY = "asr_installed_models"
ASR_CURATED_CATALOG_CONFIG_KEY = "asr_curated_catalog"
ASR_CURATED_CATALOG_URL_CONFIG_KEY = "asr_curated_catalog_url"
ASR_LAST_DOWNLOAD_STATUS_KEY = "asr_last_download_status"


def _normalize_asr_backend(name: str | None) -> str | None:
    """Return canonical backend name.

    Accepts legacy aliases like "faster-whisper" or "ctranslate2" and maps
    them to the internal identifier "ct2". The function is case-insensitive
    and returns ``None`` unchanged.
    """
    if not isinstance(name, str):
        return name
    normalized = name.strip().lower()
    if normalized in {"faster-whisper", "faster_whisper", "ctranslate2"}:
        return "ct2"
    return normalized



class ConfigManager:
    def __init__(self, config_file=CONFIG_FILE, default_config=DEFAULT_CONFIG):
        self.config_file = config_file
        self.default_config = default_config
        self.config = {}
        self._config_hash = None
        self._secrets_hash = None
        self._invalid_timeout_cache: dict[str, Any] = {}
        self.load_config()
        url = self.config.get(ASR_CURATED_CATALOG_URL_CONFIG_KEY, "")
        if url:
            self.update_asr_curated_catalog_from_url(url)

    def _compute_hash(self, data) -> str:
        """Gera um hash SHA256 determinístico para o dicionário informado."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    def load_config(self):
        cfg = copy.deepcopy(self.default_config)
        loaded_config_from_file = {}
        config_file_exists = os.path.exists(self.config_file)

        try:
            if config_file_exists:
                with open(self.config_file, "r", encoding='utf-8') as f:
                    loaded_config_from_file = json.load(f)
                self._config_hash = self._compute_hash(loaded_config_from_file)
                cfg.update(loaded_config_from_file)
                logging.info(f"Configuration loaded from {self.config_file}.")
            else:
                logging.warning(f"{self.config_file} not found. A new one will be created with default settings.")
                self.config = cfg
                self.save_config() # Salva o arquivo de configuração padrão
                self._config_hash = self._compute_hash(cfg)

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding {self.config_file}: {e}. The file is corrupted. A new one will be created with default settings.")
            cfg = copy.deepcopy(self.default_config)
            self.config = cfg
            self.save_config() # Sobrescreve o arquivo corrompido
            loaded_config_from_file = {}
            self._config_hash = self._compute_hash(cfg)

        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {self.config_file}: {e}. Using default configuration.", exc_info=True)
            loaded_config_from_file = {}
            cfg = copy.deepcopy(self.default_config)

        # Migrations for older configs
        if "vad_enabled" in loaded_config_from_file:
            logging.info("Migrating legacy 'vad_enabled' key to 'use_vad'.")
            cfg["use_vad"] = loaded_config_from_file.pop("vad_enabled")
        if ("record_storage_mode" not in loaded_config_from_file and "record_to_memory" in loaded_config_from_file):
            logging.info("Migrating legacy 'record_to_memory' key to 'record_storage_mode'.")
            rec_mem = _parse_bool(loaded_config_from_file["record_to_memory"])
            cfg["record_storage_mode"] = "memory" if rec_mem else "disk"
        
        old_agent_prompt = (
            "Você é um assistente de IA que integra um sistema operacional. "
            "Se o usuário pedir uma ação que possa ser resolvida por um comando de terminal "
            "(como listar arquivos, verificar o IP, etc.), responda exclusivamente com o comando "
            "dentro das tags <cmd>comando</cmd>. Para todas as outras solicitações, responda normalmente."
        )
        current_agent_prompt = cfg.get("prompt_agentico", "")
        if current_agent_prompt == old_agent_prompt:
            cfg["prompt_agentico"] = self.default_config["prompt_agentico"]
            logging.info("Old agent prompt detected and migrated to the new standard.")

        # Load secrets
        secrets_loaded = {}
        try:
            if os.path.exists(SECRETS_FILE):
                with open(SECRETS_FILE, "r", encoding='utf-8') as f:
                    secrets_loaded = json.load(f)
                cfg.update(secrets_loaded)
                self._secrets_hash = self._compute_hash(secrets_loaded)
                logging.info(f"Secrets loaded from {SECRETS_FILE}.")
            else:
                logging.info(f"{SECRETS_FILE} not found. API keys might be missing.")
                self._secrets_hash = None
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error reading or decoding {SECRETS_FILE}: {e}. API keys might be missing or invalid.")
            self._secrets_hash = None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {SECRETS_FILE}: {e}. API keys might be missing or invalid.", exc_info=True)
            self._secrets_hash = None

        # Validate ASR cache directory
        asr_cache_dir = cfg.get(ASR_CACHE_DIR_CONFIG_KEY, "")
        asr_cache_dir = os.path.expanduser(str(asr_cache_dir)) if isinstance(asr_cache_dir, str) else ""
        if not asr_cache_dir or not os.path.isdir(asr_cache_dir):
            asr_cache_dir = os.path.expanduser(self.default_config[ASR_CACHE_DIR_CONFIG_KEY])
            cfg[ASR_CACHE_DIR_CONFIG_KEY] = asr_cache_dir
        try:
            os.makedirs(asr_cache_dir, exist_ok=True)
        except Exception as e:
            logging.warning(f"Failed to create ASR cache directory '{asr_cache_dir}': {e}")

        cfg["asr_curated_catalog"] = list_catalog()
        try:
            cfg["asr_installed_models"] = list_installed(asr_cache_dir)
        except OSError:
            messagebox.showerror("Configuração", "Diretório de cache inválido. Verifique as configurações.")
            cfg["asr_installed_models"] = []
        except Exception as e:
            logging.warning(f"Failed to list installed models: {e}")
            cfg["asr_installed_models"] = []

        self.config = cfg
        self._validate_and_apply_config(loaded_config_from_file)
        self.save_config()

    def _validate_and_apply_config(self, loaded_config):
        self.config["record_key"] = str(self.config.get("record_key", self.default_config["record_key"])).lower()
        self.config["record_mode"] = str(self.config.get("record_mode", self.default_config["record_mode"])).lower()
        if self.config["record_mode"] not in ["toggle", "press"]:
            logging.warning(f"Invalid record_mode '{self.config['record_mode']}'. Falling back to '{self.default_config['record_mode']}'.")
            self.config["record_mode"] = self.default_config['record_mode']
        
        # Unificar auto_paste e agent_auto_paste
        self.config["auto_paste"] = _parse_bool(
            self.config.get("auto_paste", self.default_config["auto_paste"])
        )
        self.config["agent_auto_paste"] = self.config["auto_paste"]  # Garante que agent_auto_paste seja sempre igual a auto_paste

        # Flag para exibir transcrições brutas no log
        self.config[DISPLAY_TRANSCRIPTS_KEY] = _parse_bool(
            self.config.get(
                DISPLAY_TRANSCRIPTS_KEY,
                self.default_config[DISPLAY_TRANSCRIPTS_KEY],
            )
        )

        # Persistência opcional de gravações temporárias
        self.config[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = _parse_bool(
            self.config.get(
                SAVE_TEMP_RECORDINGS_CONFIG_KEY,
                self.default_config[SAVE_TEMP_RECORDINGS_CONFIG_KEY],
            )
        )

        self.config[LAUNCH_AT_STARTUP_CONFIG_KEY] = _parse_bool(
            self.config.get(
                LAUNCH_AT_STARTUP_CONFIG_KEY,
                self.default_config[LAUNCH_AT_STARTUP_CONFIG_KEY],
            )
        )


        self.config[RECORD_STORAGE_MODE_CONFIG_KEY] = str(
            self.config.get(
                RECORD_STORAGE_MODE_CONFIG_KEY,
                self.default_config[RECORD_STORAGE_MODE_CONFIG_KEY],
            )
        ).lower()
        if self.config[RECORD_STORAGE_MODE_CONFIG_KEY] == "hybrid":
            logging.info("record_storage_mode 'hybrid' mapeado para 'auto'.")
            self.config[RECORD_STORAGE_MODE_CONFIG_KEY] = "auto"
        allowed_storage_modes = ["disk", "memory", "auto", "hybrid"]
        if self.config[RECORD_STORAGE_MODE_CONFIG_KEY] not in allowed_storage_modes:
            logging.warning(
                f"Invalid record_storage_mode '{self.config[RECORD_STORAGE_MODE_CONFIG_KEY]}'. "
                f"Falling back to '{self.default_config[RECORD_STORAGE_MODE_CONFIG_KEY]}'."
            )
            self.config[RECORD_STORAGE_MODE_CONFIG_KEY] = self.default_config[
                RECORD_STORAGE_MODE_CONFIG_KEY
            ]
        try:
            self.config[RECORD_STORAGE_LIMIT_CONFIG_KEY] = int(
                self.config.get(
                    RECORD_STORAGE_LIMIT_CONFIG_KEY,
                    self.default_config[RECORD_STORAGE_LIMIT_CONFIG_KEY],
                )
            )
        except (ValueError, TypeError):
            self.config[RECORD_STORAGE_LIMIT_CONFIG_KEY] = self.default_config[
                RECORD_STORAGE_LIMIT_CONFIG_KEY
            ]

        self.config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY] = str(
            self.config.get(
                MAX_MEMORY_SECONDS_MODE_CONFIG_KEY,
                self.default_config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY],
            )
        ).lower()
        if self.config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY] not in ["manual", "auto"]:
            logging.warning(
                f"Invalid max_memory_seconds_mode '{self.config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY]}'. "
                f"Falling back to '{self.default_config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY]}'."
            )
            self.config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY] = self.default_config[
                MAX_MEMORY_SECONDS_MODE_CONFIG_KEY
            ]

        try:
            self.config["max_memory_seconds"] = float(
                self.config.get(
                    "max_memory_seconds",
                    self.default_config["max_memory_seconds"],
                )
            )
        except (ValueError, TypeError):
            self.config["max_memory_seconds"] = self.default_config["max_memory_seconds"]

        try:
            self.config["min_free_ram_mb"] = int(
                self.config.get(
                    "min_free_ram_mb",
                    self.default_config["min_free_ram_mb"],
                )
            )
        except (ValueError, TypeError):
            self.config["min_free_ram_mb"] = self.default_config["min_free_ram_mb"]

        # auto_ram_threshold_percent: inteiro 1..50 (limite de segurança)
        try:
            raw_thr = self.config.get("auto_ram_threshold_percent", self.default_config.get("auto_ram_threshold_percent", 10))
            thr = int(raw_thr)
            if not (1 <= thr <= 50):
                logging.warning(f"Invalid auto_ram_threshold_percent '{thr}'. Must be between 1 and 50. Using default (10).")
                thr = self.default_config.get("auto_ram_threshold_percent", 10)
            self.config["auto_ram_threshold_percent"] = thr
        except (ValueError, TypeError):
            self.config["auto_ram_threshold_percent"] = self.default_config.get("auto_ram_threshold_percent", 10)

        try:
            self.config[CHUNK_LENGTH_SEC_CONFIG_KEY] = float(
                self.config.get(
                    CHUNK_LENGTH_SEC_CONFIG_KEY,
                    self.default_config[CHUNK_LENGTH_SEC_CONFIG_KEY],
                )
            )
        except (ValueError, TypeError):
            self.config[CHUNK_LENGTH_SEC_CONFIG_KEY] = self.default_config[
                CHUNK_LENGTH_SEC_CONFIG_KEY
            ]

        # chunk_length_mode: 'auto' | 'manual'
        raw_chunk_mode = str(self.config.get(CHUNK_LENGTH_MODE_CONFIG_KEY, self.default_config.get(CHUNK_LENGTH_MODE_CONFIG_KEY, "manual"))).lower()
        if raw_chunk_mode not in ["auto", "manual"]:
            logging.warning(f"Invalid chunk_length_mode '{raw_chunk_mode}'. Falling back to 'manual'.")
            raw_chunk_mode = "manual"
        self.config[CHUNK_LENGTH_MODE_CONFIG_KEY] = raw_chunk_mode

        # enable_torch_compile: bool
        self.config[ENABLE_TORCH_COMPILE_CONFIG_KEY] = _parse_bool(
            self.config.get(ENABLE_TORCH_COMPILE_CONFIG_KEY, self.default_config.get(ENABLE_TORCH_COMPILE_CONFIG_KEY, False))
        )

        self.config[ASR_BACKEND_CONFIG_KEY] = str(
            self.config.get(ASR_BACKEND_CONFIG_KEY, self.default_config[ASR_BACKEND_CONFIG_KEY])
        )

        self.config[ASR_MODEL_ID_CONFIG_KEY] = str(
            self.config.get(ASR_MODEL_ID_CONFIG_KEY, self.default_config[ASR_MODEL_ID_CONFIG_KEY])
        )
    
        # Para gpu_index_specified e batch_size_specified
        self.config["batch_size_specified"] = BATCH_SIZE_CONFIG_KEY in loaded_config
        self.config["gpu_index_specified"] = GPU_INDEX_CONFIG_KEY in loaded_config
        
        # Lógica de validação para gpu_index
        try:
            raw_gpu_idx_val = loaded_config.get(GPU_INDEX_CONFIG_KEY, -1)
            gpu_idx_val = int(raw_gpu_idx_val)
            if gpu_idx_val < -1:
                logging.warning(f"Invalid GPU index '{gpu_idx_val}'. Must be -1 (auto) or >= 0. Using auto (-1).")
                self.config[GPU_INDEX_CONFIG_KEY] = -1
            else:
                self.config[GPU_INDEX_CONFIG_KEY] = gpu_idx_val
        except (ValueError, TypeError):
            logging.warning(f"Invalid GPU index value '{self.config.get(GPU_INDEX_CONFIG_KEY)}' in config. Falling back to automatic selection (-1).")
            self.config[GPU_INDEX_CONFIG_KEY] = -1
            self.config["gpu_index_specified"] = False # Se falhou a leitura, não foi especificado corretamente

        # Lógica de validação para min_transcription_duration
        try:
            raw_min_duration_val = loaded_config.get(
                MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
                self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY],
            )
            min_duration_val = float(raw_min_duration_val)
            if not (0.1 <= min_duration_val <= 10.0):  # Exemplo de range razoável
                logging.warning(
                    f"Invalid min_transcription_duration '{min_duration_val}'. "
                    "Must be between 0.1 and 10.0. Using default "
                    f"({self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]})."
                )
                self.config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY] = (
                    self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]
                )
            else:
                self.config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY] = min_duration_val
        except (ValueError, TypeError):
            logging.warning(
                "Invalid min_transcription_duration value '%s' in config. "
                "Falling back to default (%s).",
                self.config.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY),
                self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY],
            )
            self.config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY] = (
                self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]
            )

        # Validação para min_record_duration
        try:
            raw_min_rec_val = loaded_config.get(
                MIN_RECORDING_DURATION_CONFIG_KEY,
                self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY],
            )
            min_rec_val = float(raw_min_rec_val)
            if not (0.1 <= min_rec_val <= 10.0):
                logging.warning(
                    f"Invalid min_record_duration '{min_rec_val}'. "
                    "Must be between 0.1 and 10.0. Using default "
                    f"({self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY]})."
                )
                self.config[MIN_RECORDING_DURATION_CONFIG_KEY] = (
                    self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY]
                )
            else:
                self.config[MIN_RECORDING_DURATION_CONFIG_KEY] = min_rec_val
        except (ValueError, TypeError):
            logging.warning(
                "Invalid min_record_duration value '%s' in config. "
                "Falling back to default (%s).",
                self.config.get(MIN_RECORDING_DURATION_CONFIG_KEY),
                self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY],
            )
            self.config[MIN_RECORDING_DURATION_CONFIG_KEY] = (
                self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY]
            )

        # Lógica para uso do VAD
        self.config[USE_VAD_CONFIG_KEY] = _parse_bool(
            self.config.get(USE_VAD_CONFIG_KEY, self.default_config[USE_VAD_CONFIG_KEY])
        )
        self.config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY] = _parse_bool(
            self.config.get(
                DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY,
                self.default_config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY]
            )
        )
        try:
            raw_threshold = self.config.get(
                VAD_THRESHOLD_CONFIG_KEY, self.default_config[VAD_THRESHOLD_CONFIG_KEY]
            )
            self.config[VAD_THRESHOLD_CONFIG_KEY] = float(raw_threshold)
        except (ValueError, TypeError):
            logging.warning(
                "Invalid vad_threshold value '%s' in config. Using default (%s).",
                self.config.get(VAD_THRESHOLD_CONFIG_KEY),
                self.default_config[VAD_THRESHOLD_CONFIG_KEY],
            )
            self.config[VAD_THRESHOLD_CONFIG_KEY] = (
                self.default_config[VAD_THRESHOLD_CONFIG_KEY]
            )

        try:
            raw_silence = self.config.get(
                VAD_SILENCE_DURATION_CONFIG_KEY,
                self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY],
            )
            silence_val = float(raw_silence)
            if silence_val < 0.1:
                logging.warning(
                    "Invalid vad_silence_duration '%s'. Must be >= 0.1. Using default (%s).",
                    silence_val,
                    self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY],
                )
                silence_val = self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = silence_val
        except (ValueError, TypeError):
            logging.warning(
                "Invalid vad_silence_duration value '%s' in config. Using default (%s).",
                self.config.get(VAD_SILENCE_DURATION_CONFIG_KEY),
                self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY],
            )
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = (
                self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]
            )

        self.config[ASR_BACKEND_CONFIG_KEY] = str(
            self.config.get(ASR_BACKEND_CONFIG_KEY, self.default_config[ASR_BACKEND_CONFIG_KEY])
        )
        self.config[ASR_MODEL_ID_CONFIG_KEY] = str(
            self.config.get(ASR_MODEL_ID_CONFIG_KEY, self.default_config[ASR_MODEL_ID_CONFIG_KEY])
        )
        self.config[ASR_COMPUTE_DEVICE_CONFIG_KEY] = str(
            self.config.get(ASR_COMPUTE_DEVICE_CONFIG_KEY, self.default_config[ASR_COMPUTE_DEVICE_CONFIG_KEY])
        )
        self.config[ASR_DTYPE_CONFIG_KEY] = str(
            self.config.get(ASR_DTYPE_CONFIG_KEY, self.default_config[ASR_DTYPE_CONFIG_KEY])
        )
        self.config[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY] = str(
            self.config.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY, self.default_config[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY])
        )
        self.config[ASR_CACHE_DIR_CONFIG_KEY] = os.path.expanduser(
            self.config.get(ASR_CACHE_DIR_CONFIG_KEY, self.default_config[ASR_CACHE_DIR_CONFIG_KEY])
        )
        installed = self.config.get(
            ASR_INSTALLED_MODELS_CONFIG_KEY,
            self.default_config[ASR_INSTALLED_MODELS_CONFIG_KEY],
        )
        if not isinstance(installed, list):
            installed = self.default_config[ASR_INSTALLED_MODELS_CONFIG_KEY]
        self.config[ASR_INSTALLED_MODELS_CONFIG_KEY] = installed

        curated = self.config.get(
            ASR_CURATED_CATALOG_CONFIG_KEY,
            self.default_config[ASR_CURATED_CATALOG_CONFIG_KEY],
        )
        if not isinstance(curated, list):
            curated = self.default_config[ASR_CURATED_CATALOG_CONFIG_KEY]
        self.config[ASR_CURATED_CATALOG_CONFIG_KEY] = curated

        default_download_status = copy.deepcopy(
            self.default_config.get(ASR_LAST_DOWNLOAD_STATUS_KEY, {})
        )
        stored_status = self.config.get(
            ASR_LAST_DOWNLOAD_STATUS_KEY,
            default_download_status,
        )
        if not isinstance(stored_status, dict):
            stored_status = default_download_status
        sanitized_status = copy.deepcopy(default_download_status)
        if isinstance(stored_status, dict):
            for key, value in stored_status.items():
                if key not in sanitized_status:
                    sanitized_status[key] = ""
                sanitized_status[key] = "" if value is None else str(value)
        if not sanitized_status.get("status"):
            sanitized_status["status"] = default_download_status.get("status", "unknown")
        self.config[ASR_LAST_DOWNLOAD_STATUS_KEY] = sanitized_status

        safe_config = self.config.copy()
        safe_config.pop(GEMINI_API_KEY_CONFIG_KEY, None)
        safe_config.pop(OPENROUTER_API_KEY_CONFIG_KEY, None)
        logging.info(f"Settings applied: {safe_config}")


    def save_config(self):
        """Salva as configurações não sensíveis no config.json e as sensíveis no secrets.json."""
        config_to_save = copy.deepcopy(self.config)
        secrets_to_save = {}

        secret_keys = [GEMINI_API_KEY_CONFIG_KEY, OPENROUTER_API_KEY_CONFIG_KEY]

        # Separar segredos da configuração principal
        for key in secret_keys:
            if key in config_to_save:
                secrets_to_save[key] = config_to_save.pop(key)

        # Remover chaves não persistentes
        keys_to_ignore = [
            "tray_menu_items",
            "hotkey_manager",
            "asr_curated_catalog",
        ]
        for key in keys_to_ignore:
            if key in config_to_save:
                del config_to_save[key]

        # Salvar config.json apenas se mudar
        new_config_hash = self._compute_hash(config_to_save)
        if new_config_hash != self._config_hash:
            temp_file_config = self.config_file + ".tmp"
            try:
                with open(temp_file_config, "w", encoding='utf-8') as f:
                    json.dump(config_to_save, f, indent=4)
                os.replace(temp_file_config, self.config_file)
                self._config_hash = new_config_hash
                logging.info(f"Configuration saved to {self.config_file}")
            except Exception as e:
                logging.error(f"Error saving configuration to {self.config_file}: {e}")
                if os.path.exists(temp_file_config):
                    os.remove(temp_file_config)
        else:
            logging.info(f"No changes detected in {self.config_file}.")

        # Salvar secrets.json somente se houver mudanças
        temp_file_secrets = SECRETS_FILE + ".tmp"
        existing_secrets = {}
        if os.path.exists(SECRETS_FILE):
            try:
                with open(SECRETS_FILE, "r", encoding='utf-8') as f:
                    existing_secrets = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode {SECRETS_FILE}, will overwrite.")
            except FileNotFoundError:
                pass

        existing_secrets.update(secrets_to_save)
        new_secrets_hash = self._compute_hash(existing_secrets)

        if new_secrets_hash != self._secrets_hash:
            if existing_secrets or os.path.exists(SECRETS_FILE):
                try:
                    with open(temp_file_secrets, "w", encoding='utf-8') as f:
                        json.dump(existing_secrets, f, indent=4)
                    os.replace(temp_file_secrets, SECRETS_FILE)
                    logging.info(f"Secrets saved to {SECRETS_FILE}")
                except Exception as e:
                    logging.error(f"Error saving secrets to {SECRETS_FILE}: {e}")
                    if os.path.exists(temp_file_secrets):
                        os.remove(temp_file_secrets)
                else:
                    self._secrets_hash = new_secrets_hash
            else:
                # Não há arquivo nem segredos a salvar
                self._secrets_hash = new_secrets_hash
        else:
            logging.info(f"No changes detected in {SECRETS_FILE}.")

    def get(self, key, default=None):
        """Recupera valores da configuração permitindo acesso aninhado.

        Chaves separadas por pontos indicam níveis dentro de dicionários,
        possibilitando obter dinamicamente valores aninhados como
        ``parent.child``.
        """
        if isinstance(key, str) and "." in key:
            current = self.config
            for part in key.split("."):
                if not isinstance(current, dict):
                    return default
                current = current.get(part)
                if current is None:
                    return default
            return current
        value = self.config.get(key, default)
        if key == ASR_BACKEND_CONFIG_KEY:
            value = _normalize_asr_backend(value)
        return value

    def get_timeout(self, key: str, default: float | int) -> float:
        """Retorna um timeout positivo em segundos para a chave informada."""
        value = self.get(key, default)
        try:
            timeout_value = float(value)
            if timeout_value <= 0:
                raise ValueError
        except (TypeError, ValueError):
            cached_value = self._invalid_timeout_cache.get(key)
            if value != cached_value:
                logging.warning(
                    "Invalid timeout '%s' for key '%s'; using default %.2f seconds.",
                    value,
                    key,
                    float(default),
                )
                self._invalid_timeout_cache[key] = value
            return float(default)
        else:
            if key in self._invalid_timeout_cache:
                self._invalid_timeout_cache.pop(key, None)
            return timeout_value

    def set(self, key, value):
        if key == ASR_BACKEND_CONFIG_KEY:
            value = _normalize_asr_backend(value)
        self.config[key] = value

    def get_asr_model(self) -> str:
        """Compatibilidade: retorna o ``asr_model_id`` atual."""
        return self.config.get(
            ASR_MODEL_ID_CONFIG_KEY, self.default_config[ASR_MODEL_ID_CONFIG_KEY]
        )

    def set_asr_model(self, model_id: str):
        """Compatibilidade: define o ``asr_model_id``."""
        self.config[ASR_MODEL_ID_CONFIG_KEY] = model_id

    def get_api_key(self, provider: str) -> str:
        if provider == SERVICE_GEMINI:
            return self.get(GEMINI_API_KEY_CONFIG_KEY)
        if provider == SERVICE_OPENROUTER:
            return self.get(OPENROUTER_API_KEY_CONFIG_KEY)
        return ""

    def get_asr_backend(self):
        value = self.config.get(
            ASR_BACKEND_CONFIG_KEY,
            self.default_config[ASR_BACKEND_CONFIG_KEY],
        )
        return _normalize_asr_backend(value)

    def set_asr_backend(self, value: str):
        self.config[ASR_BACKEND_CONFIG_KEY] = _normalize_asr_backend(value)

    def get_asr_model_id(self):
        return self.config.get(
            ASR_MODEL_ID_CONFIG_KEY,
            self.default_config[ASR_MODEL_ID_CONFIG_KEY],
        )

    def set_asr_model_id(self, value: str):
        self.config[ASR_MODEL_ID_CONFIG_KEY] = str(value)

    def get_asr_compute_device(self):
        return self.config.get(
            ASR_COMPUTE_DEVICE_CONFIG_KEY,
            self.default_config[ASR_COMPUTE_DEVICE_CONFIG_KEY],
        )

    def set_asr_compute_device(self, value: str):
        self.config[ASR_COMPUTE_DEVICE_CONFIG_KEY] = str(value)

    def get_asr_dtype(self):
        return self.config.get(
            ASR_DTYPE_CONFIG_KEY,
            self.default_config[ASR_DTYPE_CONFIG_KEY],
        )

    def set_asr_dtype(self, value: str):
        self.config[ASR_DTYPE_CONFIG_KEY] = str(value)

    def get_asr_ct2_compute_type(self):
        return self.config.get(
            ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            self.default_config[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY],
        )

    def set_asr_ct2_compute_type(self, value: str):
        self.config[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY] = str(value)

    def get_asr_cache_dir(self):
        return self.config.get(
            ASR_CACHE_DIR_CONFIG_KEY,
            self.default_config[ASR_CACHE_DIR_CONFIG_KEY],
        )

    def set_asr_cache_dir(self, value: str):
        self.config[ASR_CACHE_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_asr_installed_models(self) -> List[str]:
        return self.config.get(
            ASR_INSTALLED_MODELS_CONFIG_KEY,
            self.default_config[ASR_INSTALLED_MODELS_CONFIG_KEY],
        )

    def set_asr_installed_models(self, value: List[str]):
        if isinstance(value, list):
            self.config[ASR_INSTALLED_MODELS_CONFIG_KEY] = list(value)
        else:
            self.config[ASR_INSTALLED_MODELS_CONFIG_KEY] = list(
                self.default_config[ASR_INSTALLED_MODELS_CONFIG_KEY]
            )

    def get_last_asr_download_status(self) -> dict:
        status = self.config.get(
            ASR_LAST_DOWNLOAD_STATUS_KEY,
            copy.deepcopy(self.default_config.get(ASR_LAST_DOWNLOAD_STATUS_KEY, {})),
        )
        if not isinstance(status, dict):
            return copy.deepcopy(
                self.default_config.get(ASR_LAST_DOWNLOAD_STATUS_KEY, {})
            )
        return copy.deepcopy(status)

    def set_last_asr_download_status(self, value: dict):
        default_status = copy.deepcopy(
            self.default_config.get(ASR_LAST_DOWNLOAD_STATUS_KEY, {})
        )
        sanitized = copy.deepcopy(default_status)
        if isinstance(value, dict):
            for key, raw in value.items():
                normalized = "" if raw is None else str(raw)
                sanitized[key] = normalized
        if not sanitized.get("status"):
            sanitized["status"] = default_status.get("status", "unknown")
        self.config[ASR_LAST_DOWNLOAD_STATUS_KEY] = sanitized

    def get_asr_curated_catalog(self):
        return self.config.get(
            ASR_CURATED_CATALOG_CONFIG_KEY,
            self.default_config[ASR_CURATED_CATALOG_CONFIG_KEY],
        )

    def set_asr_curated_catalog(self, value: list):
        if isinstance(value, list):
            self.config[ASR_CURATED_CATALOG_CONFIG_KEY] = value
        else:
            self.config[ASR_CURATED_CATALOG_CONFIG_KEY] = self.default_config[
                ASR_CURATED_CATALOG_CONFIG_KEY
            ]

    def update_asr_curated_catalog_from_url(self, url: str, timeout: int = 10) -> bool:
        """Carrega um catálogo curado de modelos de ASR a partir de ``url``.

        O JSON recebido deve ser uma lista de dicionários contendo pelo menos o
        campo ``model_id``. Caso os dados sejam válidos, o catálogo interno é
        atualizado e salvo no arquivo de configuração.

        :param url: Endereço da fonte externa.
        :param timeout: Tempo máximo de espera para a requisição em segundos.
        :return: ``True`` se o catálogo foi atualizado com sucesso.
        """

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            is_list = isinstance(data, list)
            all_have_ids = all(
                isinstance(item, dict) and "model_id" in item
                for item in data
            )
            if is_list and all_have_ids:
                self.set_asr_curated_catalog(data)
                try:
                    self.save_config()
                except Exception:
                    logging.warning("Falha ao salvar config após atualizar catálogo curado.")
                return True
            logging.warning("Formato inválido de catálogo obtido de %s", url)
        except Exception as e:
            logging.error("Erro ao atualizar catálogo curado de %s: %s", url, e)
        return False

    def get_use_vad(self):
        return self.config.get(USE_VAD_CONFIG_KEY, self.default_config[USE_VAD_CONFIG_KEY])

    def set_use_vad(self, value: bool):
        self.config[USE_VAD_CONFIG_KEY] = bool(value)

    def get_vad_threshold(self):
        return self.config.get(VAD_THRESHOLD_CONFIG_KEY, self.default_config[VAD_THRESHOLD_CONFIG_KEY])

    def set_vad_threshold(self, value: float):
        try:
            self.config[VAD_THRESHOLD_CONFIG_KEY] = float(value)
        except (ValueError, TypeError):
            self.config[VAD_THRESHOLD_CONFIG_KEY] = self.default_config[VAD_THRESHOLD_CONFIG_KEY]

    def get_vad_silence_duration(self):
        return self.config.get(VAD_SILENCE_DURATION_CONFIG_KEY, self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY])

    def set_vad_silence_duration(self, value: float):
        try:
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = float(value)
        except (ValueError, TypeError):
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]

    def get_display_transcripts_in_terminal(self):
        return self.config.get(
            DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY,
            self.default_config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY]
        )

    def set_display_transcripts_in_terminal(self, value: bool):
        self.config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY] = bool(value)

    def get_save_temp_recordings(self):
        return self.config.get(
            SAVE_TEMP_RECORDINGS_CONFIG_KEY,
            self.default_config[SAVE_TEMP_RECORDINGS_CONFIG_KEY],
        )

    def set_save_temp_recordings(self, value: bool):
        self.config[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = bool(value)


    def get_record_storage_mode(self):
        return self.config.get(
            RECORD_STORAGE_MODE_CONFIG_KEY,
            self.default_config[RECORD_STORAGE_MODE_CONFIG_KEY],
        )

    def set_record_storage_mode(self, value: str):
        self.config[RECORD_STORAGE_MODE_CONFIG_KEY] = str(value)

    def get_record_storage_limit(self):
        return self.config.get(
            RECORD_STORAGE_LIMIT_CONFIG_KEY,
            self.default_config[RECORD_STORAGE_LIMIT_CONFIG_KEY],
        )

    def set_record_storage_limit(self, value: int):
        try:
            self.config[RECORD_STORAGE_LIMIT_CONFIG_KEY] = int(value)
        except (ValueError, TypeError):
            self.config[RECORD_STORAGE_LIMIT_CONFIG_KEY] = self.default_config[
                RECORD_STORAGE_LIMIT_CONFIG_KEY
            ]

    def get_max_memory_seconds_mode(self):
        return self.config.get(
            MAX_MEMORY_SECONDS_MODE_CONFIG_KEY,
            self.default_config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY],
        )

    def set_max_memory_seconds_mode(self, value: str):
        if value in ["manual", "auto"]:
            self.config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY] = value
        else:
            self.config[MAX_MEMORY_SECONDS_MODE_CONFIG_KEY] = self.default_config[
                MAX_MEMORY_SECONDS_MODE_CONFIG_KEY
            ]

    def get_max_memory_seconds(self):
        return self.config.get(
            "max_memory_seconds",
            self.default_config["max_memory_seconds"],
        )

    def set_max_memory_seconds(self, value: float | int):
        try:
            self.config["max_memory_seconds"] = float(value)
        except (ValueError, TypeError):
            self.config["max_memory_seconds"] = self.default_config["max_memory_seconds"]

    def get_min_free_ram_mb(self):
        return self.config.get(
            "min_free_ram_mb",
            self.default_config["min_free_ram_mb"],
        )

    def set_min_free_ram_mb(self, value: int):
        try:
            self.config["min_free_ram_mb"] = int(value)
        except (ValueError, TypeError):
            self.config["min_free_ram_mb"] = self.default_config["min_free_ram_mb"]

    def get_launch_at_startup(self):
        return self.config.get(
            LAUNCH_AT_STARTUP_CONFIG_KEY,
            self.default_config[LAUNCH_AT_STARTUP_CONFIG_KEY],
        )

    def set_launch_at_startup(self, value: bool):
        self.config[LAUNCH_AT_STARTUP_CONFIG_KEY] = bool(value)

    def get_chunk_length_sec(self):
        return self.config.get(
            CHUNK_LENGTH_SEC_CONFIG_KEY,
            self.default_config[CHUNK_LENGTH_SEC_CONFIG_KEY],
        )

    def get_chunk_length_mode(self):
        return self.config.get(
            CHUNK_LENGTH_MODE_CONFIG_KEY,
            self.default_config.get(CHUNK_LENGTH_MODE_CONFIG_KEY, "manual"),
        )

    def set_chunk_length_mode(self, value: str):
        val = str(value).lower()
        if val not in ["auto", "manual"]:
            val = "manual"
        self.config[CHUNK_LENGTH_MODE_CONFIG_KEY] = val

    def get_enable_torch_compile(self):
        return self.config.get(
            ENABLE_TORCH_COMPILE_CONFIG_KEY,
            self.default_config.get(ENABLE_TORCH_COMPILE_CONFIG_KEY, False),
        )

    def set_enable_torch_compile(self, value: bool):
        self.config[ENABLE_TORCH_COMPILE_CONFIG_KEY] = bool(value)

    def set_chunk_length_sec(self, value: float | int):
        try:
            self.config[CHUNK_LENGTH_SEC_CONFIG_KEY] = float(value)
        except (ValueError, TypeError):
            self.config[CHUNK_LENGTH_SEC_CONFIG_KEY] = self.default_config[
                CHUNK_LENGTH_SEC_CONFIG_KEY
            ]

    def get_min_record_duration(self):
        return self.config.get(
            MIN_RECORDING_DURATION_CONFIG_KEY,
            self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY],
        )

    def set_min_record_duration(self, value: float | int):
        try:
            self.config[MIN_RECORDING_DURATION_CONFIG_KEY] = float(value)
        except (ValueError, TypeError):
            self.config[MIN_RECORDING_DURATION_CONFIG_KEY] = self.default_config[
                MIN_RECORDING_DURATION_CONFIG_KEY
            ]
        self.save_config()

    def get_last_model_prompt_decision(self) -> dict:
        decision = self.config.get(
            LAST_MODEL_PROMPT_DECISION_CONFIG_KEY,
            self.default_config[LAST_MODEL_PROMPT_DECISION_CONFIG_KEY],
        )
        if not isinstance(decision, dict):
            decision = copy.deepcopy(
                self.default_config[LAST_MODEL_PROMPT_DECISION_CONFIG_KEY]
            )
        return {
            "model_id": str(decision.get("model_id", "") or ""),
            "backend": str(decision.get("backend", "") or ""),
            "decision": str(decision.get("decision", "") or "").strip().lower(),
            "timestamp": float(decision.get("timestamp", 0.0) or 0.0),
        }

    def record_model_prompt_decision(
        self,
        decision: str,
        model_id: str,
        backend: str,
        *,
        save: bool = True,
    ) -> None:
        normalized_decision = str(decision or "").strip().lower()
        if normalized_decision not in {"accept", "defer"}:
            normalized_decision = ""
        timestamp = time.time() if normalized_decision else 0.0
        self.config[LAST_MODEL_PROMPT_DECISION_CONFIG_KEY] = {
            "model_id": str(model_id or ""),
            "backend": str(backend or ""),
            "decision": normalized_decision,
            "timestamp": timestamp,
        }
        if save:
            self.save_config()

    def reset_last_model_prompt_decision(self, *, save: bool = False) -> None:
        self.config[LAST_MODEL_PROMPT_DECISION_CONFIG_KEY] = copy.deepcopy(
            self.default_config[LAST_MODEL_PROMPT_DECISION_CONFIG_KEY]
        )
        if save:
            self.save_config()
