import copy
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox
from typing import Any, List

import requests

from .config_schema import coerce_with_defaults
from .model_manager import get_curated_entry, list_catalog, list_installed, normalize_backend_label
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
SECRETS_FILE = "secrets.json"  # Nova constante para o arquivo de segredos

DEFAULT_CONFIG = {
    "record_key": "F3",
    "record_mode": "toggle",
    "auto_paste": True,
    "agent_auto_paste": True,
    "auto_paste_modifier": "auto",
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
    "batch_size": 16,  # Valor padrão para o modo automático
    "batch_size_mode": "auto",  # Novo: 'auto' ou 'manual'
    "manual_batch_size": 8,  # Novo: Valor para o modo manual
    "gpu_index": 0,
    "hotkey_stability_service_enabled": True,  # Nova configuração unificada
    "use_vad": False,
    "vad_threshold": 0.5,
    # Duração máxima da pausa preservada antes que o silêncio seja descartado
    "vad_silence_duration": 1.0,
    # Valores alinhados com AppConfig em config_schema.py para coerência de VAD.
    "vad_pre_speech_padding_ms": 150,
    "vad_post_speech_padding_ms": 300,
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
    "min_transcription_duration": 1.0,  # Nova configuração
    "chunk_length_sec": 30,
    "chunk_length_mode": "manual",
    "enable_torch_compile": False,
    "launch_at_startup": False,
    "clear_gpu_cache": True,
    "asr_model_id": "openai/whisper-large-v3-turbo",
    "asr_backend": "ctranslate2",
    "asr_compute_device": "auto",
    "asr_dtype": "float16",
    "asr_ct2_compute_type": "int8_float16",
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
VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY = "vad_pre_speech_padding_ms"
VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY = "vad_post_speech_padding_ms"
CHUNK_LENGTH_SEC_CONFIG_KEY = "chunk_length_sec"
LAUNCH_AT_STARTUP_CONFIG_KEY = "launch_at_startup"
DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY = DISPLAY_TRANSCRIPTS_KEY
KEYBOARD_LIBRARY_CONFIG_KEY = "keyboard_library"
KEYBOARD_LIB_WIN32 = "win32"
TEXT_CORRECTION_ENABLED_CONFIG_KEY = "text_correction_enabled"
TEXT_CORRECTION_SERVICE_CONFIG_KEY = "text_correction_service"
ENABLE_AI_CORRECTION_CONFIG_KEY = TEXT_CORRECTION_ENABLED_CONFIG_KEY
AUTO_PASTE_MODIFIER_CONFIG_KEY = "auto_paste_modifier"
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
    """Return canonical backend name for persistence and UI consistency."""
    if not isinstance(name, str):
        return name
    normalized = name.strip().lower()
    if normalized in {"faster whisper", "faster_whisper"}:
        normalized = "faster-whisper"
    if normalized in {"ct2", "ctranslate2"}:
        return "ctranslate2"
    if normalized == "faster-whisper":
        return "faster-whisper"
    return normalized


class ConfigManager:
    """Gerencia persistência de configuração e segredos do aplicativo."""

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
        raw_cfg = copy.deepcopy(self.default_config)
        loaded_config_from_file: dict[str, Any] = {}

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as file_descriptor:
                    loaded_config_from_file = json.load(file_descriptor)
                self._config_hash = self._compute_hash(loaded_config_from_file)
                raw_cfg.update(loaded_config_from_file)
                logging.info("Configuration loaded from %s.", self.config_file)
            except json.JSONDecodeError as exc:
                logging.error(
                    "Error decoding %s: %s. Recreating with defaults.",
                    self.config_file,
                    exc,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                logging.error(
                    "Unexpected error while loading %s: %s.",
                    self.config_file,
                    exc,
                    exc_info=True,
                )
        else:
            logging.info(
                "%s not found. Creating it with default settings for the first launch.",
                self.config_file,
            )

        # Migrate legacy keys before validation
        if "vad_enabled" in loaded_config_from_file:
            logging.info("Migrating legacy 'vad_enabled' key to 'use_vad'.")
            raw_cfg["use_vad"] = _parse_bool(loaded_config_from_file.get("vad_enabled"))
        if (
            "record_storage_mode" not in loaded_config_from_file
            and "record_to_memory" in loaded_config_from_file
        ):
            logging.info("Migrating legacy 'record_to_memory' key to 'record_storage_mode'.")
            record_to_memory = _parse_bool(loaded_config_from_file.get("record_to_memory"))
            raw_cfg["record_storage_mode"] = "memory" if record_to_memory else "disk"

        old_agent_prompt = (
            "Você é um assistente de IA que integra um sistema operacional. "
            "Se o usuário pedir uma ação que possa ser resolvida por um comando de terminal "
            "(como listar arquivos, verificar o IP, etc.), responda exclusivamente com o comando "
            "dentro das tags <cmd>comando</cmd>. Para todas as outras solicitações, responda normalmente."
        )
        if raw_cfg.get("prompt_agentico") == old_agent_prompt:
            raw_cfg["prompt_agentico"] = self.default_config["prompt_agentico"]
            logging.info("Old agent prompt detected and migrated to the new standard.")

        secrets_loaded: dict[str, Any] = {}
        if os.path.exists(SECRETS_FILE):
            try:
                with open(SECRETS_FILE, "r", encoding="utf-8") as file_descriptor:
                    secrets_loaded = json.load(file_descriptor)
                raw_cfg.update(secrets_loaded)
                self._secrets_hash = self._compute_hash(secrets_loaded)
                logging.info("Secrets loaded from %s.", SECRETS_FILE)
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                logging.warning(
                    "Error reading %s: %s. Secrets will be ignored until corrected.",
                    SECRETS_FILE,
                    exc,
                )
                self._secrets_hash = None
            except Exception as exc:  # pragma: no cover - defensive path
                logging.error(
                    "Unexpected error loading %s: %s.",
                    SECRETS_FILE,
                    exc,
                    exc_info=True,
                )
                self._secrets_hash = None
        else:
            logging.info("%s not found. API keys might be missing.", SECRETS_FILE)
            self._secrets_hash = None

        sanitized_cfg, validation_warnings = coerce_with_defaults(raw_cfg, self.default_config)
        for warning in validation_warnings:
            logging.warning(warning)

        self.config = sanitized_cfg
        self._apply_runtime_overrides(loaded_config=loaded_config_from_file)
        self.save_config()

    def _apply_runtime_overrides(
        self,
        *,
        loaded_config: dict[str, Any] | None = None,
        applied_updates: dict[str, Any] | None = None,
    ) -> None:
        """Apply derived configuration values after schema validation."""

        cfg = self.config

        def _source_value(key: str, *, default: Any) -> Any:
            if applied_updates and key in applied_updates:
                return applied_updates[key]
            if loaded_config and key in loaded_config:
                return loaded_config[key]
            return default

        # Normalize hotkey fields for internal consumption
        cfg["record_key"] = str(cfg.get("record_key", self.default_config["record_key"])).lower()
        cfg["record_mode"] = str(cfg.get("record_mode", self.default_config["record_mode"])).lower()

        # Agent auto paste mirrors auto paste unless explicitly overridden
        auto_paste_value = bool(cfg.get("auto_paste", self.default_config["auto_paste"]))
        agent_auto_paste = cfg.get("agent_auto_paste")
        cfg["auto_paste"] = auto_paste_value
        cfg["agent_auto_paste"] = auto_paste_value if agent_auto_paste is None else bool(agent_auto_paste)

        # Ensure boolean switches remain booleans
        cfg[DISPLAY_TRANSCRIPTS_KEY] = bool(
            cfg.get(DISPLAY_TRANSCRIPTS_KEY, self.default_config[DISPLAY_TRANSCRIPTS_KEY])
        )
        cfg[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = bool(
            cfg.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY, self.default_config[SAVE_TEMP_RECORDINGS_CONFIG_KEY])
        )
        cfg[LAUNCH_AT_STARTUP_CONFIG_KEY] = bool(
            cfg.get(LAUNCH_AT_STARTUP_CONFIG_KEY, self.default_config[LAUNCH_AT_STARTUP_CONFIG_KEY])
        )

        # Track whether the user explicitly defined batch size / GPU index
        batch_size_specified = bool(cfg.get("batch_size_specified"))
        gpu_index_specified = bool(cfg.get("gpu_index_specified"))

        if loaded_config is not None:
            batch_size_specified = BATCH_SIZE_CONFIG_KEY in loaded_config
            gpu_index_specified = GPU_INDEX_CONFIG_KEY in loaded_config

        if applied_updates is not None:
            if BATCH_SIZE_CONFIG_KEY in applied_updates:
                batch_size_specified = True
            if GPU_INDEX_CONFIG_KEY in applied_updates:
                gpu_index_specified = True

        cfg["batch_size_specified"] = batch_size_specified
        cfg["gpu_index_specified"] = gpu_index_specified

        cache_dir_value = cfg.get(ASR_CACHE_DIR_CONFIG_KEY, self.default_config[ASR_CACHE_DIR_CONFIG_KEY])
        cache_path = Path(str(cache_dir_value)).expanduser()
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive path
            logging.warning("Failed to create ASR cache directory '%s': %s", cache_path, exc)
        cfg[ASR_CACHE_DIR_CONFIG_KEY] = str(cache_path)

        cfg[ASR_CURATED_CATALOG_CONFIG_KEY] = list_catalog()
        default_model_id = str(self.default_config[ASR_MODEL_ID_CONFIG_KEY])
        configured_model_id = str(
            cfg.get(ASR_MODEL_ID_CONFIG_KEY, default_model_id)
        ).strip() or default_model_id
        curated_entry = get_curated_entry(configured_model_id)
        if curated_entry is None:
            logging.warning(
                "Configured ASR model '%s' is not part of the curated catalog; falling back to '%s'.",
                configured_model_id,
                default_model_id,
            )
            configured_model_id = default_model_id
            curated_entry = get_curated_entry(configured_model_id)
        cfg[ASR_MODEL_ID_CONFIG_KEY] = configured_model_id

        expected_backend = normalize_backend_label(
            curated_entry.get("backend") if curated_entry else None
        )
        configured_backend = _normalize_asr_backend(
            cfg.get(ASR_BACKEND_CONFIG_KEY, self.default_config[ASR_BACKEND_CONFIG_KEY])
        )
        if expected_backend and configured_backend != expected_backend:
            logging.warning(
                "Configured ASR backend '%s' is incompatible with curated model '%s'; forcing '%s'.",
                configured_backend,
                configured_model_id,
                expected_backend,
            )
            configured_backend = expected_backend
        if not configured_backend:
            configured_backend = _normalize_asr_backend(
                self.default_config[ASR_BACKEND_CONFIG_KEY]
            )
        cfg[ASR_BACKEND_CONFIG_KEY] = configured_backend

        try:
            cfg[ASR_INSTALLED_MODELS_CONFIG_KEY] = list_installed(str(cache_path))
        except OSError:
            messagebox.showerror("Configuração", "Diretório de cache inválido. Verifique as configurações.")
            cfg[ASR_INSTALLED_MODELS_CONFIG_KEY] = []
        except Exception as exc:  # pragma: no cover - defensive path
            logging.warning("Failed to list installed models: %s", exc)
            cfg[ASR_INSTALLED_MODELS_CONFIG_KEY] = []

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

        backend_value = _normalize_asr_backend(
            self.config.get(ASR_BACKEND_CONFIG_KEY, self.default_config[ASR_BACKEND_CONFIG_KEY])
        )
        if not backend_value:
            backend_value = _normalize_asr_backend(self.default_config[ASR_BACKEND_CONFIG_KEY])
        self.config[ASR_BACKEND_CONFIG_KEY] = backend_value

        model_id_value = str(
            self.config.get(ASR_MODEL_ID_CONFIG_KEY, self.default_config[ASR_MODEL_ID_CONFIG_KEY])
            or ""
        ).strip()
        if not model_id_value:
            model_id_value = str(self.default_config[ASR_MODEL_ID_CONFIG_KEY])
        self.config[ASR_MODEL_ID_CONFIG_KEY] = model_id_value
    
        # Lógica de validação para gpu_index
        try:
            raw_gpu_idx_val = _source_value(
                GPU_INDEX_CONFIG_KEY,
                default=cfg.get(GPU_INDEX_CONFIG_KEY, -1),
            )
            gpu_idx_val = int(raw_gpu_idx_val)
            if gpu_idx_val < -1:
                logging.warning(f"Invalid GPU index '{gpu_idx_val}'. Must be -1 (auto) or >= 0. Using auto (-1).")
                self.config[GPU_INDEX_CONFIG_KEY] = -1
            else:
                self.config[GPU_INDEX_CONFIG_KEY] = gpu_idx_val
        except (ValueError, TypeError):
            logging.warning(f"Invalid GPU index value '{self.config.get(GPU_INDEX_CONFIG_KEY)}' in config. Falling back to automatic selection (-1).")
            self.config[GPU_INDEX_CONFIG_KEY] = -1
            self.config["gpu_index_specified"] = False  # Se falhou a leitura, não foi especificado corretamente

        # Lógica de validação para min_transcription_duration
        try:
            raw_min_duration_val = _source_value(
                MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
                default=self.config.get(
                    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
                    self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY],
                ),
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
            raw_min_rec_val = _source_value(
                MIN_RECORDING_DURATION_CONFIG_KEY,
                default=self.config.get(
                    MIN_RECORDING_DURATION_CONFIG_KEY,
                    self.default_config[MIN_RECORDING_DURATION_CONFIG_KEY],
                ),
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

        try:
            raw_pre_padding = self.config.get(
                VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
                self.default_config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY],
            )
            pre_padding_val = float(raw_pre_padding)
            if pre_padding_val < 0.0:
                logging.warning(
                    "Invalid vad_pre_speech_padding_ms '%s'. Must be >= 0. Using default (%s).",
                    pre_padding_val,
                    self.default_config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY],
                )
                pre_padding_val = float(
                    self.default_config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY]
                )
            self.config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY] = pre_padding_val
        except (ValueError, TypeError, KeyError):
            logging.warning(
                "Invalid vad_pre_speech_padding_ms value '%s' in config. Using default (%s).",
                self.config.get(VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY),
                self.default_config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY],
            )
            self.config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY] = float(
                self.default_config[VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY]
            )

        try:
            raw_post_padding = self.config.get(
                VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
                self.default_config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY],
            )
            post_padding_val = float(raw_post_padding)
            if post_padding_val < 0.0:
                logging.warning(
                    "Invalid vad_post_speech_padding_ms '%s'. Must be >= 0. Using default (%s).",
                    post_padding_val,
                    self.default_config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY],
                )
                post_padding_val = float(
                    self.default_config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY]
                )
            self.config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY] = post_padding_val
        except (ValueError, TypeError, KeyError):
            logging.warning(
                "Invalid vad_post_speech_padding_ms value '%s' in config. Using default (%s).",
                self.config.get(VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY),
                self.default_config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY],
            )
            self.config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY] = float(
                self.default_config[VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY]
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

        safe_config = dict(self.config)
        safe_config.pop(GEMINI_API_KEY_CONFIG_KEY, None)
        safe_config.pop(OPENROUTER_API_KEY_CONFIG_KEY, None)
        logging.info("Settings applied: %s", str(safe_config))

    def apply_updates(self, updates: dict[str, Any]) -> tuple[set[str], list[str]]:
        """Apply partial configuration updates validated by the schema."""

        if not updates:
            return set(), []

        filtered_updates = dict(updates)
        if "record_to_memory" in filtered_updates:
            logging.info(
                "Ignoring legacy 'record_to_memory' update in favor of 'record_storage_mode'."
            )
            filtered_updates.pop("record_to_memory", None)

        if not filtered_updates:
            return set(), []

        previous_config = copy.deepcopy(self.config)
        candidate = copy.deepcopy(self.config)
        candidate.update(filtered_updates)

        sanitized_cfg, warnings = coerce_with_defaults(candidate, self.default_config)
        for warning in warnings:
            logging.warning(warning)

        self.config = sanitized_cfg
        self._apply_runtime_overrides(applied_updates=filtered_updates)
        self.save_config()

        changed_keys = {
            key
            for key, value in self.config.items()
            if previous_config.get(key) != value
        }
        return changed_keys, warnings

    def reset_to_defaults(self) -> tuple[dict[str, Any], set[str]]:
        """Reset the configuration to the baked-in defaults and persist them."""

        previous_config = copy.deepcopy(self.config)

        sanitized_cfg, warnings = coerce_with_defaults(
            copy.deepcopy(self.default_config),
            self.default_config,
        )
        for warning in warnings:
            logging.warning(warning)

        self.config = sanitized_cfg
        self._apply_runtime_overrides(applied_updates=self.default_config)
        self.save_config()

        changed_keys = {
            key
            for key, value in self.config.items()
            if previous_config.get(key) != value
        }

        return copy.deepcopy(self.config), changed_keys

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

        secrets_file_exists = os.path.exists(SECRETS_FILE)
        should_write_secrets = new_secrets_hash != self._secrets_hash or not secrets_file_exists

        if should_write_secrets:
            try:
                with open(temp_file_secrets, "w", encoding='utf-8') as f:
                    json.dump(existing_secrets, f, indent=4)
                os.replace(temp_file_secrets, SECRETS_FILE)
                self._secrets_hash = new_secrets_hash
                logging.info(f"Secrets saved to {SECRETS_FILE}")
            except Exception as e:
                logging.error(f"Error saving secrets to {SECRETS_FILE}: {e}")
                if os.path.exists(temp_file_secrets):
                    os.remove(temp_file_secrets)
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

    def record_model_download_status(
        self,
        *,
        status: str,
        model_id: str,
        backend: str,
        message: str = "",
        details: str = "",
        timestamp: datetime | None = None,
        save: bool = True,
    ) -> None:
        """Capture and persist metadata about the last model download attempt."""

        normalized_status = str(status or "").strip().lower()
        if not normalized_status:
            normalized_status = "unknown"
        timestamp_value = timestamp or datetime.now(timezone.utc)
        payload = {
            "status": normalized_status,
            "timestamp": timestamp_value.isoformat(),
            "model_id": str(model_id or ""),
            "backend": str(backend or ""),
            "message": str(message or ""),
            "details": str(details or ""),
        }
        self.set_last_asr_download_status(payload)
        if save:
            self.save_config()

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
