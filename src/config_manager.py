import copy
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tkinter import messagebox
from typing import Any, List, Mapping, TYPE_CHECKING

import requests

from .config_schema import (
    AppConfig,
    coerce_with_defaults,
    flatten_config_tree,
    normalize_payload_tree,
)
from .hotkey_normalization import _normalize_key_name
from .model_manager import (
    HardwareProfile,
    build_runtime_catalog,
    get_curated_entry,
    list_catalog,
    list_installed,
    normalize_backend_label,
    select_recommended_model,
)
from .logging_utils import StructuredMessage, get_logger, log_context
from .utils import get_total_memory_mb
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

_BASE_STORAGE_ROOT = (Path.home() / ".cache" / "whisper_flash_transcriber").expanduser()
_DEFAULT_STORAGE_ROOT_DIR = str(_BASE_STORAGE_ROOT)
_DEFAULT_MODELS_STORAGE_DIR = str((_BASE_STORAGE_ROOT / "models").expanduser())
_DEFAULT_ASR_CACHE_DIR = str((Path(_DEFAULT_MODELS_STORAGE_DIR) / "asr").expanduser())
_DEFAULT_DEPS_INSTALL_DIR = str((_BASE_STORAGE_ROOT / "deps").expanduser())
_DEFAULT_HF_HOME_DIR = str((Path(_DEFAULT_DEPS_INSTALL_DIR) / "huggingface").expanduser())
_DEFAULT_RECORDINGS_DIR = str((_BASE_STORAGE_ROOT / "recordings").expanduser())
_DEFAULT_PYTHON_PACKAGES_DIR = str((_BASE_STORAGE_ROOT / "python_packages").expanduser())
_DEFAULT_VAD_MODELS_DIR = str((_BASE_STORAGE_ROOT / "vad").expanduser())
_DEFAULT_HF_CACHE_DIR = str((_BASE_STORAGE_ROOT / "hf_cache").expanduser())


_PROFILE_ENV_VAR = "WHISPER_FLASH_PROFILE_DIR"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_profile_dir() -> Path:
    """Return the directory that should hold persistent profile artifacts."""

    override = os.environ.get(_PROFILE_ENV_VAR)
    if override:
        try:
            return Path(override).expanduser()
        except Exception:
            logging.warning("Invalid profile directory override '%s'; falling back to defaults.", override)
    return _BASE_STORAGE_ROOT


PROFILE_DIR = _resolve_profile_dir().expanduser()
CONFIG_FILE_NAME = "config.json"
SECRETS_FILE_NAME = "secrets.json"
HOTKEY_CONFIG_FILE_NAME = "hotkey_config.json"
CONFIG_FILE = str((PROFILE_DIR / CONFIG_FILE_NAME).expanduser())
SECRETS_FILE = str((PROFILE_DIR / SECRETS_FILE_NAME).expanduser())
HOTKEY_CONFIG_FILE = str((PROFILE_DIR / HOTKEY_CONFIG_FILE_NAME).expanduser())
LEGACY_CONFIG_LOCATIONS: tuple[Path, ...] = (
    (_PROJECT_ROOT / CONFIG_FILE_NAME).expanduser(),
    (Path.cwd() / CONFIG_FILE_NAME).expanduser(),
)
LEGACY_SECRETS_LOCATIONS: tuple[Path, ...] = (
    (_PROJECT_ROOT / SECRETS_FILE_NAME).expanduser(),
    (Path.cwd() / SECRETS_FILE_NAME).expanduser(),
)
LEGACY_HOTKEY_LOCATIONS: tuple[Path, ...] = (
    (_PROJECT_ROOT / HOTKEY_CONFIG_FILE_NAME).expanduser(),
    (Path.cwd() / HOTKEY_CONFIG_FILE_NAME).expanduser(),
)


DEFAULT_CONFIG = {
    "show_advanced": False,
    "record_key": "F3",
    "record_mode": "toggle",
    "hotkey_debounce_ms": 200,
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
    "openrouter_max_attempts": 3,
    "gemini_timeout": 120,
    "text_correction_timeout": 15,
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
    "ui_language": "en-US",
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
    "max_memory_seconds_mode": "auto",
    "max_memory_seconds": 30,
    "min_free_ram_mb": 1000,
    "auto_ram_threshold_percent": 10,
    "max_parallel_downloads": 1,
    "min_transcription_duration": 1.0,  # Nova configuração
    "chunk_length_sec": 30,
    "chunk_length_mode": "auto",
    "launch_at_startup": False,
    "clear_gpu_cache": True,
    "storage_root_dir": _DEFAULT_STORAGE_ROOT_DIR,
    "models_storage_dir": _DEFAULT_MODELS_STORAGE_DIR,
    "deps_install_dir": _DEFAULT_DEPS_INSTALL_DIR,
    "hf_home_dir": _DEFAULT_HF_HOME_DIR,
    "recordings_dir": _DEFAULT_RECORDINGS_DIR,
    "asr_model_id": "distil-whisper/distil-large-v3",
    "asr_backend": "ctranslate2",
    "asr_compute_device": "auto",
    "asr_ct2_compute_type": "int8_float16",
    "asr_cache_dir": _DEFAULT_ASR_CACHE_DIR,
    "asr_installed_models": [],
    "asr_curated_catalog": list_catalog(),
    "asr_curated_catalog_url": "",
    "asr_last_download_status": {
        "status": "unknown",
        "timestamp": "",
        "model_id": "",
        "backend": "",
        "message": "",
        "details": "",
        "target_dir": "",
        "bytes_downloaded": 0,
        "throughput_bps": 0.0,
        "duration_seconds": 0.0,
    },
    "asr_download_history": [],
    "asr_last_prompt_decision": {
        "model_id": "",
        "backend": "",
        "decision": "",
        "timestamp": 0,
    },
    "first_run_completed": False,
}

DEFAULT_CONFIG_TREE = normalize_payload_tree(DEFAULT_CONFIG)


DEFAULT_CONFIG_TREE = AppConfig.model_validate(
    normalize_payload_tree(DEFAULT_CONFIG)
).model_dump()


LOGGER = get_logger("whisper_flash_transcriber.config", component="ConfigManager")
BOOTSTRAP_LOGGER = get_logger(
    "whisper_flash_transcriber.config.bootstrap", component="Bootstrap"
)

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
HOTKEY_DEBOUNCE_MS_CONFIG_KEY = "hotkey_debounce_ms"
BATCH_SIZE_CONFIG_KEY = "batch_size" # Agora é o batch size padrão para o modo auto
BATCH_SIZE_MODE_CONFIG_KEY = "batch_size_mode" # Novo
MANUAL_BATCH_SIZE_CONFIG_KEY = "manual_batch_size" # Novo
GPU_INDEX_CONFIG_KEY = "gpu_index"
SAVE_TEMP_RECORDINGS_CONFIG_KEY = "save_temp_recordings"
RECORD_STORAGE_MODE_CONFIG_KEY = "record_storage_mode"
RECORD_STORAGE_LIMIT_CONFIG_KEY = "record_storage_limit"
RECORDINGS_DIR_CONFIG_KEY = "recordings_dir"
PYTHON_PACKAGES_DIR_CONFIG_KEY = "python_packages_dir"
VAD_MODELS_DIR_CONFIG_KEY = "vad_models_dir"
HF_CACHE_DIR_CONFIG_KEY = "hf_cache_dir"
MAX_MEMORY_SECONDS_MODE_CONFIG_KEY = "max_memory_seconds_mode"
UI_LANGUAGE_CONFIG_KEY = "ui_language"
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
TEXT_CORRECTION_TIMEOUT_CONFIG_KEY = "text_correction_timeout"
ENABLE_AI_CORRECTION_CONFIG_KEY = TEXT_CORRECTION_ENABLED_CONFIG_KEY
AUTO_PASTE_MODIFIER_CONFIG_KEY = "auto_paste_modifier"
SERVICE_NONE = "none"
SERVICE_OPENROUTER = "openrouter"
SERVICE_GEMINI = "gemini"
OPENROUTER_API_KEY_CONFIG_KEY = "openrouter_api_key"
OPENROUTER_MODEL_CONFIG_KEY = "openrouter_model"
OPENROUTER_TIMEOUT_CONFIG_KEY = "openrouter_timeout"
OPENROUTER_MAX_ATTEMPTS_CONFIG_KEY = "openrouter_max_attempts"
GEMINI_API_KEY_CONFIG_KEY = "gemini_api_key"
GEMINI_MODEL_CONFIG_KEY = "gemini_model"
GEMINI_AGENT_MODEL_CONFIG_KEY = "gemini_agent_model"
GEMINI_MODEL_OPTIONS_CONFIG_KEY = "gemini_model_options"
# Novas constantes de timeout de APIs externas
GEMINI_TIMEOUT_CONFIG_KEY = "gemini_timeout"
GEMINI_MAX_ATTEMPTS_CONFIG_KEY = "gemini_max_attempts"
# Novas constantes para otimizações de desempenho
CHUNK_LENGTH_MODE_CONFIG_KEY = "chunk_length_mode"
ENABLE_TORCH_COMPILE_CONFIG_KEY = "enable_torch_compile"
GEMINI_AGENT_PROMPT_CONFIG_KEY = "prompt_agentico"
OPENROUTER_PROMPT_CONFIG_KEY = "openrouter_prompt"
GEMINI_PROMPT_CONFIG_KEY = "gemini_prompt"
FIRST_RUN_COMPLETED_CONFIG_KEY = "first_run_completed"
SETTINGS_WINDOW_GEOMETRY = "550x700"
REREGISTER_INTERVAL_SECONDS = 60
MAX_HOTKEY_FAILURES = 3
HOTKEY_HEALTH_CHECK_INTERVAL = 10
CLEAR_GPU_CACHE_CONFIG_KEY = "clear_gpu_cache"
STORAGE_ROOT_DIR_CONFIG_KEY = "storage_root_dir"
RECORDINGS_DIR_CONFIG_KEY = "recordings_dir"
ASR_BACKEND_CONFIG_KEY = "asr_backend"
ASR_MODEL_ID_CONFIG_KEY = "asr_model_id"
ASR_COMPUTE_DEVICE_CONFIG_KEY = "asr_compute_device"
ASR_CT2_COMPUTE_TYPE_CONFIG_KEY = "asr_ct2_compute_type"
ASR_CT2_CPU_THREADS_CONFIG_KEY = "asr_ct2_cpu_threads"
MODELS_STORAGE_DIR_CONFIG_KEY = "models_storage_dir"
DEPS_INSTALL_DIR_CONFIG_KEY = "deps_install_dir"
HF_HOME_DIR_CONFIG_KEY = "hf_home_dir"
ASR_CACHE_DIR_CONFIG_KEY = "asr_cache_dir"
ASR_INSTALLED_MODELS_CONFIG_KEY = "asr_installed_models"
ASR_CURATED_CATALOG_CONFIG_KEY = "asr_curated_catalog"
ASR_CURATED_CATALOG_URL_CONFIG_KEY = "asr_curated_catalog_url"
ASR_LAST_DOWNLOAD_STATUS_KEY = "asr_last_download_status"

SIMPLE_MODE_RESET_KEYS: tuple[str, ...] = (
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY,
    MANUAL_BATCH_SIZE_CONFIG_KEY,
    USE_VAD_CONFIG_KEY,
    VAD_THRESHOLD_CONFIG_KEY,
    VAD_SILENCE_DURATION_CONFIG_KEY,
    VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
    VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    RECORD_STORAGE_MODE_CONFIG_KEY,
    RECORD_STORAGE_LIMIT_CONFIG_KEY,
    MAX_MEMORY_SECONDS_MODE_CONFIG_KEY,
    "max_memory_seconds",
    CHUNK_LENGTH_MODE_CONFIG_KEY,
    CHUNK_LENGTH_SEC_CONFIG_KEY,
)


@dataclass(frozen=True)
class PersistenceRecord:
    """Snapshot do resultado de persistência de um artefato."""

    path: Path
    existed_before: bool
    wrote: bool
    created: bool
    verified: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "existed_before": self.existed_before,
            "wrote": self.wrote,
            "created": self.created,
            "verified": self.verified,
            "error": self.error,
        }


@dataclass(frozen=True)
class PersistenceOutcome:
    """Resumo das operações de persistência do ciclo atual."""

    config: PersistenceRecord
    secrets: PersistenceRecord


class ConfigPersistenceError(RuntimeError):
    """Erro disparado quando não é possível persistir a configuração em disco."""


CONFIG_LOGGER = get_logger("whisper_flash_transcriber.config", component="ConfigManager")


def _normalize_asr_backend(name: str | None) -> str | None:
    """Return canonical backend name for persistence and UI consistency."""
    if name is None or not isinstance(name, str):
        return name

    normalized = name.strip().lower()
    if not normalized:
        return ""

    alias_map = {
        "ct2": "ctranslate2",
        "ctranslate2": "ctranslate2",
        "faster whisper": "ctranslate2",
        "faster_whisper": "ctranslate2",
        "faster-whisper": "ctranslate2",
        "transformer": "ctranslate2",
        "transformers": "ctranslate2",
        "auto": "ctranslate2",
    }

    mapped = alias_map.get(normalized)
    if mapped:
        return mapped

    CONFIG_LOGGER.debug(
        log_context(
            "Mapping unsupported ASR backend to 'ctranslate2'.",
            event="config.unsupported_backend_normalized",
            backend=name,
        )
    )
    return "ctranslate2"


class ConfigManager:
    """Gerencia persistência de configuração e segredos do aplicativo."""

    def __init__(
        self,
        config_file: str = CONFIG_FILE,
        default_config_tree: dict[str, Any] | None = None,
    ):
        resolved_config_path = Path(config_file).expanduser()
        if not resolved_config_path.is_absolute():
            candidate_name = resolved_config_path.name
            if candidate_name == CONFIG_FILE_NAME:
                resolved_config_path = Path(CONFIG_FILE).expanduser()
            else:
                resolved_config_path = (Path.cwd() / resolved_config_path).expanduser()

        if default_config_tree is None:
            default_config_tree = DEFAULT_CONFIG_TREE
        self.default_config_tree = copy.deepcopy(default_config_tree)
        self.default_config = flatten_config_tree(self.default_config_tree)
        self.config_file = str(resolved_config_path)
        self.config_path = resolved_config_path
        self._config_path = resolved_config_path
        self.secrets_path = Path(SECRETS_FILE).expanduser()
        self._secrets_path = self.secrets_path
        self.profile_dir = self._config_path.parent

        try:
            self.profile_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.warning(
                "Unable to prepare profile directory '%s': %s", self.profile_dir, exc
            )

        config_existed = self._config_path.exists()
        secrets_existed = self._secrets_path.exists()

        migrated_config = None
        migrated_secrets = None
        if not config_existed:
            migrated_config = self._maybe_migrate_profile_file(
                target=self._config_path,
                candidates=LEGACY_CONFIG_LOCATIONS,
                label="config",
            )
            config_existed = self._config_path.exists()
        if not secrets_existed:
            migrated_secrets = self._maybe_migrate_profile_file(
                target=self._secrets_path,
                candidates=LEGACY_SECRETS_LOCATIONS,
                label="secrets",
            )
            secrets_existed = self._secrets_path.exists()

        self.config_tree = copy.deepcopy(self.default_config_tree)
        self.config = {}
        self._config_hash = None
        self._secrets_hash = None
        self._invalid_timeout_cache: dict[str, Any] = {}
        self._invalid_retry_cache: dict[str, Any] = {}
        self._config_existed_on_boot = config_existed
        self._bootstrap_state: dict[str, dict[str, Any]] = {
            "config": {
                "path": self._config_path,
                "existed": config_existed,
                "written": False,
                "verified": False,
                "error": None,
            },
            "secrets": {
                "path": self._secrets_path,
                "existed": secrets_existed,
                "written": False,
                "verified": False,
                "error": None,
            },
        }
        if migrated_config:
            self._bootstrap_state["config"]["migrated_from"] = migrated_config
        if migrated_secrets:
            self._bootstrap_state["secrets"]["migrated_from"] = migrated_secrets

        self._bootstrap_logged = False
        self._runtime_hardware_profile: HardwareProfile | None = None
        self._runtime_model_catalog: list[dict[str, Any]] = []
        self._runtime_recommendation: dict[str, Any] | None = None
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

        if self.config_path.exists():
            try:
                with self.config_path.open("r", encoding="utf-8") as file_descriptor:
                    loaded_config_from_file = json.load(file_descriptor)
                self._config_hash = self._compute_hash(loaded_config_from_file)
                raw_cfg.update(loaded_config_from_file)
                LOGGER.info(
                    log_context(
                        "Configuration loaded from disk.",
                        event="config.load.success",
                        path=str(self.config_path),
                        keys=len(loaded_config_from_file),
                    )
                )
            except json.JSONDecodeError as exc:
                LOGGER.error(
                    log_context(
                        "Error decoding configuration file; recreating from defaults.",
                        event="config.load.invalid_json",
                        path=str(self.config_path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.error(
                    log_context(
                        "Unexpected error while loading configuration file.",
                        event="config.load.failure",
                        path=str(self.config_path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
        else:
            LOGGER.info(
                log_context(
                    "Configuration file not found; generating a fresh profile.",
                    event="config.load.first_run",
                    path=str(self.config_path),
                )
            )
            BOOTSTRAP_LOGGER.info(
                StructuredMessage(
                    "Configuration file missing; defaults will be materialized.",
                    event="config.bootstrap.config_missing",
                    details={
                        "path": str(self._config_path),
                        "first_run": True,
                    },
                )
            )

        # Migrate legacy keys before validation
        if "vad_enabled" in loaded_config_from_file:
            logging.info("Migrating legacy 'vad_enabled' key to 'use_vad'.")
            raw_cfg["use_vad"] = _parse_bool(loaded_config_from_file.get("vad_enabled"))
        legacy_record_to_memory = loaded_config_from_file.get("record_to_memory")
        if (
            "record_storage_mode" not in loaded_config_from_file
            and "record_to_memory" in loaded_config_from_file
        ):
            logging.info("Migrating legacy 'record_to_memory' key to 'record_storage_mode'.")
            record_to_memory = _parse_bool(legacy_record_to_memory)
            raw_cfg["record_storage_mode"] = "memory" if record_to_memory else "disk"

        if "record_to_memory" in raw_cfg:
            raw_cfg.pop("record_to_memory", None)
        if "record_to_memory" in loaded_config_from_file:
            loaded_config_from_file.pop("record_to_memory", None)

        old_agent_prompt = (
            "Você é um assistente de IA que integra um sistema operacional. "
            "Se o usuário pedir uma ação que possa ser resolvida por um comando de terminal "
            "(como listar arquivos, verificar o IP, etc.), responda exclusivamente com o comando "
            "dentro das tags <cmd>comando</cmd>. Para todas as outras solicitações, responda normalmente."
        )
        if raw_cfg.get("prompt_agentico") == old_agent_prompt:
            raw_cfg["prompt_agentico"] = self.default_config["prompt_agentico"]
            logging.info("Old agent prompt detected and migrated to the new standard.")

        if FIRST_RUN_COMPLETED_CONFIG_KEY not in loaded_config_from_file:
            raw_cfg[FIRST_RUN_COMPLETED_CONFIG_KEY] = bool(self._config_existed_on_boot)

        secrets_loaded: dict[str, Any] = {}
        if self.secrets_path.exists():
            try:
                with self.secrets_path.open("r", encoding="utf-8") as file_descriptor:
                    secrets_loaded = json.load(file_descriptor)
                raw_cfg.update(secrets_loaded)
                self._secrets_hash = self._compute_hash(secrets_loaded)
                LOGGER.info(
                    log_context(
                        "Secrets loaded from disk.",
                        event="config.secrets.load.success",
                        path=str(self.secrets_path),
                        keys=len(secrets_loaded),
                    )
                )
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                LOGGER.warning(
                    log_context(
                        "Error reading secrets file; ignoring secrets until corrected.",
                        event="config.secrets.load.invalid",
                        path=str(self.secrets_path),
                        error=str(exc),
                    )
                )
                self._secrets_hash = None
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.error(
                    log_context(
                        "Unexpected error while loading secrets file.",
                        event="config.secrets.load.failure",
                        path=str(self.secrets_path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
                self._secrets_hash = None
        else:
            LOGGER.info(
                log_context(
                    "Secrets file not found; creating a fresh store.",
                    event="config.secrets.load.first_run",
                    path=str(self.secrets_path),
                )
            )
            self._secrets_hash = None
            BOOTSTRAP_LOGGER.info(
                StructuredMessage(
                    "Secrets file missing; an empty template will be created.",
                    event="config.bootstrap.secrets_missing",
                    details={
                        "path": str(self._secrets_path),
                        "first_run": True,
                    },
                )
            )

        sanitized_tree, validation_warnings = coerce_with_defaults(
            raw_cfg, self.default_config_tree
        )
        for warning in validation_warnings:
            logging.warning(warning)

        self.config_tree = sanitized_tree
        self.config = flatten_config_tree(sanitized_tree)
        self._apply_runtime_overrides(
            loaded_config=loaded_config_from_file,
            previous_config=None,
        )
        self.save_config()

    def _apply_runtime_overrides(
        self,
        *,
        loaded_config: dict[str, Any] | None = None,
        applied_updates: dict[str, Any] | None = None,
        previous_config: dict[str, Any] | None = None,
    ) -> None:
        """Apply derived configuration values after schema validation."""

        cfg = self.config

        def _detect_hardware_profile() -> HardwareProfile:
            system_ram_mb: int | None = None
            try:
                total_ram = get_total_memory_mb()
                if isinstance(total_ram, (int, float)):
                    system_ram_mb = int(total_ram)
            except Exception:
                logging.debug("Unable to determine system RAM.", exc_info=True)

            has_cuda = False
            gpu_count = 0
            max_vram_mb = 0
            torch_mod = None

            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is not None:
                try:
                    torch_mod = importlib.import_module("torch")
                except Exception:
                    logging.debug("torch import failed during hardware probing.", exc_info=True)
                    torch_mod = None

            if torch_mod is not None:
                cuda_module = getattr(torch_mod, "cuda", None)
                is_available = getattr(cuda_module, "is_available", None)
                if callable(is_available):
                    try:
                        has_cuda = bool(is_available())
                    except Exception:
                        logging.debug("torch.cuda.is_available() failed.", exc_info=True)
                        has_cuda = False
                if has_cuda:
                    device_count_fn = getattr(cuda_module, "device_count", None)
                    if callable(device_count_fn):
                        try:
                            gpu_count = int(device_count_fn())
                        except Exception:
                            logging.debug("torch.cuda.device_count() failed.", exc_info=True)
                            gpu_count = 0
                    get_props = getattr(cuda_module, "get_device_properties", None)
                    if callable(get_props):
                        for idx in range(max(gpu_count, 0)):
                            try:
                                props = get_props(idx)
                                total_memory = getattr(props, "total_memory", 0)
                                if total_memory:
                                    max_vram_mb = max(
                                        max_vram_mb,
                                        int(int(total_memory) // (1024 * 1024)),
                                    )
                            except Exception:
                                logging.debug("Failed to read CUDA device properties for index %d.", idx, exc_info=True)
                                continue

            return HardwareProfile(
                system_ram_mb=system_ram_mb,
                has_cuda=has_cuda,
                gpu_count=gpu_count,
                max_vram_mb=max_vram_mb,
            )

        def _source_value(key: str, *, default: Any) -> Any:
            if applied_updates and key in applied_updates:
                return applied_updates[key]
            if loaded_config and key in loaded_config:
                return loaded_config[key]
            return default

        # Normalize hotkey fields for internal consumption
        default_record_key = (
            _normalize_key_name(self.default_config["record_key"])
            or self.default_config["record_key"]
        )
        cfg["record_key"] = (
            _normalize_key_name(cfg.get("record_key", default_record_key))
            or default_record_key
        )
        cfg["record_mode"] = str(cfg.get("record_mode", self.default_config["record_mode"])).lower()
        debounce_source = cfg.get(
            HOTKEY_DEBOUNCE_MS_CONFIG_KEY,
            self.default_config[HOTKEY_DEBOUNCE_MS_CONFIG_KEY],
        )
        try:
            debounce_value = float(debounce_source)
        except (TypeError, ValueError):
            logging.warning(
                "Invalid hotkey_debounce_ms value '%s'; falling back to default %.2f ms.",
                debounce_source,
                float(self.default_config[HOTKEY_DEBOUNCE_MS_CONFIG_KEY]),
            )
            debounce_value = float(self.default_config[HOTKEY_DEBOUNCE_MS_CONFIG_KEY])
        else:
            if debounce_value < 0:
                logging.warning(
                    "hotkey_debounce_ms cannot be negative (value=%s); coercing to 0.",
                    debounce_source,
                )
                debounce_value = 0.0
        cfg[HOTKEY_DEBOUNCE_MS_CONFIG_KEY] = int(round(debounce_value))

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
        recordings_dir_value = cfg.get(RECORDINGS_DIR_CONFIG_KEY, self.default_config[RECORDINGS_DIR_CONFIG_KEY])
        recordings_path = Path(str(recordings_dir_value)).expanduser()
        try:
            recordings_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive path
            logging.warning("Failed to create recordings directory '%s': %s", recordings_path, exc)
        cfg[RECORDINGS_DIR_CONFIG_KEY] = str(recordings_path)
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

        def _coerce_path(value: Any, *, default: Path) -> Path:
            if value in (None, ""):
                return default
            if isinstance(value, Path):
                return value.expanduser()
            try:
                candidate = Path(str(value)).expanduser()
            except Exception:
                return default
            return candidate

        def _normalized_str(path: Path) -> str:
            try:
                return str(path.expanduser().resolve())
            except Exception:
                try:
                    return str(path.expanduser().absolute())
                except Exception:
                    return str(path.expanduser())

        def _ensure_directory(path: Path, *, fallback: Path, description: str) -> Path:
            candidates: list[tuple[Path, str]] = [
                (path, "requested"),
                (fallback, "default"),
                (Path.cwd(), "working"),
            ]
            normalized_seen: set[str] = set()
            for candidate, label in candidates:
                normalized = _normalized_str(candidate)
                if not normalized or normalized in normalized_seen:
                    continue
                normalized_seen.add(normalized)
                candidate_path = Path(normalized)
                try:
                    candidate_path.mkdir(parents=True, exist_ok=True)
                except Exception as exc:
                    logging.warning(
                        "%s directory (%s) '%s' is not accessible: %s",
                        description.capitalize(),
                        label,
                        candidate_path,
                        exc,
                    )
                    continue
                if label != "requested":
                    logging.info(
                        "%s directory resolved to '%s' via %s fallback.",
                        description.capitalize(),
                        candidate_path,
                        label,
                    )
                return candidate_path
            logging.error(
                "Unable to secure %s directory; falling back to current working directory.",
                description,
            )
            return Path.cwd()

        default_storage_root_path = Path(
            self.default_config[STORAGE_ROOT_DIR_CONFIG_KEY]
        ).expanduser()
        storage_root_raw = _source_value(
            STORAGE_ROOT_DIR_CONFIG_KEY,
            default=cfg.get(
                STORAGE_ROOT_DIR_CONFIG_KEY,
                self.default_config[STORAGE_ROOT_DIR_CONFIG_KEY],
            ),
        )
        requested_storage_root_path = _coerce_path(
            storage_root_raw,
            default=default_storage_root_path,
        )

        previous_storage_root_path: Path | None = None
        if loaded_config and STORAGE_ROOT_DIR_CONFIG_KEY in loaded_config:
            previous_storage_root_path = _coerce_path(
                loaded_config[STORAGE_ROOT_DIR_CONFIG_KEY],
                default=default_storage_root_path,
            )
        elif previous_config and STORAGE_ROOT_DIR_CONFIG_KEY in previous_config:
            previous_storage_root_path = _coerce_path(
                previous_config[STORAGE_ROOT_DIR_CONFIG_KEY],
                default=default_storage_root_path,
            )

        storage_root_path = _ensure_directory(
            requested_storage_root_path,
            fallback=default_storage_root_path,
            description="storage root",
        )
        cfg[STORAGE_ROOT_DIR_CONFIG_KEY] = str(storage_root_path)

        default_models_storage_path = storage_root_path / "models"
        configured_models_default = self.default_config.get(
            MODELS_STORAGE_DIR_CONFIG_KEY
        )
        if configured_models_default:
            default_models_storage_path = _coerce_path(
                configured_models_default,
                default=default_models_storage_path,
            )

        previous_models_storage_path: Path | None = None
        if previous_config and MODELS_STORAGE_DIR_CONFIG_KEY in previous_config:
            previous_models_storage_path = _coerce_path(
                previous_config[MODELS_STORAGE_DIR_CONFIG_KEY],
                default=default_models_storage_path,
            )
        elif loaded_config and MODELS_STORAGE_DIR_CONFIG_KEY in loaded_config:
            previous_models_storage_path = _coerce_path(
                loaded_config[MODELS_STORAGE_DIR_CONFIG_KEY],
                default=default_models_storage_path,
            )

        models_defaults = {
            _normalized_str(default_models_storage_path),
            _normalized_str(storage_root_path),
        }
        if previous_storage_root_path is not None:
            models_defaults.add(_normalized_str(previous_storage_root_path))
        if previous_models_storage_path is not None:
            models_defaults.add(_normalized_str(previous_models_storage_path))

        models_override = False
        if applied_updates and MODELS_STORAGE_DIR_CONFIG_KEY in applied_updates:
            models_override = True
        elif loaded_config and MODELS_STORAGE_DIR_CONFIG_KEY in loaded_config:
            loaded_models_path = _normalized_str(
                _coerce_path(
                    loaded_config[MODELS_STORAGE_DIR_CONFIG_KEY],
                    default=default_models_storage_path,
                )
            )
            if loaded_models_path not in models_defaults:
                models_override = True

        models_raw = _source_value(
            MODELS_STORAGE_DIR_CONFIG_KEY,
            default=cfg.get(
                MODELS_STORAGE_DIR_CONFIG_KEY,
                str(default_models_storage_path),
            ),
        )
        if models_override:
            requested_models_storage_path = _coerce_path(
                models_raw,
                default=default_models_storage_path,
            )
        else:
            requested_models_storage_path = default_models_storage_path

        models_storage_path = _ensure_directory(
            requested_models_storage_path,
            fallback=default_models_storage_path,
            description="models storage",
        )
        cfg[MODELS_STORAGE_DIR_CONFIG_KEY] = str(models_storage_path)

        derived_asr_path = models_storage_path / "asr"
        default_asr_path = _coerce_path(
            self.default_config.get(ASR_CACHE_DIR_CONFIG_KEY, derived_asr_path),
            default=derived_asr_path,
        )

        previous_asr_path: Path | None = None
        if previous_config and ASR_CACHE_DIR_CONFIG_KEY in previous_config:
            previous_asr_path = _coerce_path(
                previous_config[ASR_CACHE_DIR_CONFIG_KEY],
                default=default_asr_path,
            )
        elif loaded_config and ASR_CACHE_DIR_CONFIG_KEY in loaded_config:
            previous_asr_path = _coerce_path(
                loaded_config[ASR_CACHE_DIR_CONFIG_KEY],
                default=default_asr_path,
            )

        asr_defaults = {
            _normalized_str(derived_asr_path),
            _normalized_str(default_asr_path),
        }
        if previous_storage_root_path is not None:
            asr_defaults.add(_normalized_str(previous_storage_root_path / "asr"))
        if previous_models_storage_path is not None:
            asr_defaults.add(_normalized_str(previous_models_storage_path / "asr"))
        if previous_asr_path is not None:
            asr_defaults.add(_normalized_str(previous_asr_path))

        asr_override = False
        if applied_updates and ASR_CACHE_DIR_CONFIG_KEY in applied_updates:
            asr_override = True
        elif loaded_config and ASR_CACHE_DIR_CONFIG_KEY in loaded_config:
            loaded_asr_path = _normalized_str(
                _coerce_path(
                    loaded_config[ASR_CACHE_DIR_CONFIG_KEY],
                    default=default_asr_path,
                )
            )
            if loaded_asr_path not in asr_defaults:
                asr_override = True

        asr_raw = _source_value(
            ASR_CACHE_DIR_CONFIG_KEY,
            default=cfg.get(
                ASR_CACHE_DIR_CONFIG_KEY,
                self.default_config[ASR_CACHE_DIR_CONFIG_KEY],
            ),
        )
        if asr_override:
            requested_asr_path = _coerce_path(asr_raw, default=derived_asr_path)
            cache_path = _ensure_directory(
                requested_asr_path,
                fallback=derived_asr_path,
                description="ASR cache",
            )
        else:
            cache_path = _ensure_directory(
                derived_asr_path,
                fallback=default_asr_path,
                description="ASR cache",
            )
        cfg[ASR_CACHE_DIR_CONFIG_KEY] = str(cache_path)

        default_deps_path = storage_root_path / "deps"
        configured_deps_default = self.default_config.get(DEPS_INSTALL_DIR_CONFIG_KEY)
        if configured_deps_default:
            default_deps_path = _coerce_path(
                configured_deps_default,
                default=default_deps_path,
            )

        previous_deps_path: Path | None = None
        if previous_config and DEPS_INSTALL_DIR_CONFIG_KEY in previous_config:
            previous_deps_path = _coerce_path(
                previous_config[DEPS_INSTALL_DIR_CONFIG_KEY],
                default=default_deps_path,
            )
        elif loaded_config and DEPS_INSTALL_DIR_CONFIG_KEY in loaded_config:
            previous_deps_path = _coerce_path(
                loaded_config[DEPS_INSTALL_DIR_CONFIG_KEY],
                default=default_deps_path,
            )

        deps_defaults = {
            _normalized_str(default_deps_path),
            _normalized_str(storage_root_path / "deps"),
        }
        if previous_storage_root_path is not None:
            deps_defaults.add(_normalized_str(previous_storage_root_path / "deps"))
        if previous_deps_path is not None:
            deps_defaults.add(_normalized_str(previous_deps_path))

        deps_override = False
        if applied_updates and DEPS_INSTALL_DIR_CONFIG_KEY in applied_updates:
            deps_override = True
        elif loaded_config and DEPS_INSTALL_DIR_CONFIG_KEY in loaded_config:
            loaded_deps_path = _normalized_str(
                _coerce_path(
                    loaded_config[DEPS_INSTALL_DIR_CONFIG_KEY],
                    default=default_deps_path,
                )
            )
            if loaded_deps_path not in deps_defaults:
                deps_override = True

        deps_raw = _source_value(
            DEPS_INSTALL_DIR_CONFIG_KEY,
            default=cfg.get(
                DEPS_INSTALL_DIR_CONFIG_KEY,
                self.default_config[DEPS_INSTALL_DIR_CONFIG_KEY],
            ),
        )
        if deps_override:
            requested_deps_path = _coerce_path(deps_raw, default=default_deps_path)
        else:
            requested_deps_path = default_deps_path

        deps_install_path = _ensure_directory(
            requested_deps_path,
            fallback=default_deps_path,
            description="dependencies",
        )
        cfg[DEPS_INSTALL_DIR_CONFIG_KEY] = str(deps_install_path)

        hf_home_path = _ensure_directory(
            deps_install_path / "huggingface",
            fallback=deps_install_path,
            description="Hugging Face home",
        )
        cfg[HF_HOME_DIR_CONFIG_KEY] = str(hf_home_path)

        derived_recordings_path = storage_root_path / "recordings"
        default_recordings_path = Path(
            self.default_config[RECORDINGS_DIR_CONFIG_KEY]
        ).expanduser()
        recordings_defaults = {
            _normalized_str(derived_recordings_path),
            _normalized_str(default_recordings_path),
        }
        if previous_storage_root_path is not None:
            recordings_defaults.add(
                _normalized_str(previous_storage_root_path / "recordings")
            )

        previous_default_recordings = (
            previous_storage_root_path / "recordings"
            if previous_storage_root_path is not None
            else default_recordings_path
        )
        previous_recordings_path: Path | None = None
        if previous_config and RECORDINGS_DIR_CONFIG_KEY in previous_config:
            previous_recordings_path = _coerce_path(
                previous_config[RECORDINGS_DIR_CONFIG_KEY],
                default=previous_default_recordings,
            )
        elif loaded_config and RECORDINGS_DIR_CONFIG_KEY in loaded_config:
            previous_recordings_path = _coerce_path(
                loaded_config[RECORDINGS_DIR_CONFIG_KEY],
                default=previous_default_recordings,
            )

        recordings_override = False
        if applied_updates and RECORDINGS_DIR_CONFIG_KEY in applied_updates:
            recordings_override = True
        elif loaded_config and RECORDINGS_DIR_CONFIG_KEY in loaded_config:
            loaded_recordings_path = _normalized_str(
                _coerce_path(
                    loaded_config[RECORDINGS_DIR_CONFIG_KEY],
                    default=derived_recordings_path,
                )
            )
            if loaded_recordings_path not in recordings_defaults:
                recordings_override = True

        recordings_raw = _source_value(
            RECORDINGS_DIR_CONFIG_KEY,
            default=cfg.get(
                RECORDINGS_DIR_CONFIG_KEY,
                self.default_config[RECORDINGS_DIR_CONFIG_KEY],
            ),
        )
        if recordings_override:
            requested_recordings_path = _coerce_path(
                recordings_raw,
                default=derived_recordings_path,
            )
            recordings_path = _ensure_directory(
                requested_recordings_path,
                fallback=derived_recordings_path,
                description="recordings",
            )
        else:
            recordings_path = _ensure_directory(
                derived_recordings_path,
                fallback=default_recordings_path,
                description="recordings",
            )
        cfg[RECORDINGS_DIR_CONFIG_KEY] = str(recordings_path)

        self._maybe_migrate_storage_paths(
            previous_storage_root=previous_storage_root_path,
            new_storage_root=storage_root_path,
            previous_asr_path=previous_asr_path,
            new_asr_path=cache_path,
            previous_recordings_path=previous_recordings_path,
            new_recordings_path=recordings_path,
            asr_override=asr_override,
            recordings_override=recordings_override,
            previous_deps_path=previous_deps_path,
            new_deps_path=deps_install_path,
            deps_override=deps_override,
        )

        cfg[ASR_CURATED_CATALOG_CONFIG_KEY] = list_catalog()

        hardware_profile = _detect_hardware_profile()
        runtime_catalog = build_runtime_catalog(hardware_profile)
        recommendation_entry = select_recommended_model(runtime_catalog)

        self._runtime_hardware_profile = hardware_profile
        self._runtime_model_catalog = copy.deepcopy(runtime_catalog)

        default_model_id = str(self.default_config[ASR_MODEL_ID_CONFIG_KEY])
        configured_model_id = str(
            cfg.get(ASR_MODEL_ID_CONFIG_KEY, default_model_id)
        ).strip() or default_model_id

        user_defined_model = False
        if loaded_config and ASR_MODEL_ID_CONFIG_KEY in loaded_config:
            user_defined_model = True
        if applied_updates and ASR_MODEL_ID_CONFIG_KEY in applied_updates:
            user_defined_model = True

        curated_entry = get_curated_entry(configured_model_id)
        auto_selected_model = False

        if not user_defined_model and recommendation_entry is not None:
            configured_model_id = recommendation_entry["id"]
            curated_entry = get_curated_entry(configured_model_id)
            auto_selected_model = True

        if curated_entry is None and recommendation_entry is not None:
            configured_model_id = recommendation_entry["id"]
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

        recommendation_payload = None
        if recommendation_entry is not None:
            recommendation_payload = copy.deepcopy(recommendation_entry)
            recommendation_payload["auto_applied"] = auto_selected_model
            recommendation_payload["hardware_profile"] = hardware_profile.to_dict()
        self._runtime_recommendation = recommendation_payload

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
        raw_chunk_mode = str(
            self.config.get(
                CHUNK_LENGTH_MODE_CONFIG_KEY,
                self.default_config.get(CHUNK_LENGTH_MODE_CONFIG_KEY, "auto"),
            )
        ).lower()
        if raw_chunk_mode not in ["auto", "manual"]:
            logging.warning(
                f"Invalid chunk_length_mode '{raw_chunk_mode}'. Falling back to 'auto'."
            )
            raw_chunk_mode = "auto"
        self.config[CHUNK_LENGTH_MODE_CONFIG_KEY] = raw_chunk_mode

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
        numeric_status_keys = {"bytes_downloaded", "throughput_bps", "duration_seconds"}
        if isinstance(stored_status, dict):
            for key, value in stored_status.items():
                if key not in sanitized_status:
                    sanitized_status[key] = ""
                if value is None:
                    sanitized_status[key] = "" if key not in numeric_status_keys else None
                elif key in numeric_status_keys:
                    try:
                        sanitized_status[key] = (
                            int(value)
                            if key == "bytes_downloaded"
                            else float(value)
                        )
                    except (TypeError, ValueError):
                        sanitized_status[key] = None if key != "bytes_downloaded" else 0
                else:
                    sanitized_status[key] = str(value)
        if not sanitized_status.get("status"):
            sanitized_status["status"] = default_download_status.get("status", "unknown")
        self.config[ASR_LAST_DOWNLOAD_STATUS_KEY] = sanitized_status

        history_entries = self.config.get("asr_download_history", [])
        sanitized_history: list[dict[str, object]] = []
        if isinstance(history_entries, list):
            for entry in history_entries:
                if not isinstance(entry, dict):
                    continue
                normalized_entry: dict[str, object] = {}
                for key, value in entry.items():
                    if key in numeric_status_keys:
                        if value is None:
                            normalized_entry[key] = None
                        else:
                            try:
                                normalized_entry[key] = (
                                    int(value)
                                    if key == "bytes_downloaded"
                                    else float(value)
                                )
                            except (TypeError, ValueError):
                                normalized_entry[key] = None if key != "bytes_downloaded" else 0
                    else:
                        normalized_entry[key] = "" if value is None else str(value)
                sanitized_history.append(normalized_entry)
        history_limit = 50
        if len(sanitized_history) > history_limit:
            sanitized_history = sanitized_history[-history_limit:]
        self.config["asr_download_history"] = sanitized_history

        safe_config = dict(self.config)
        safe_config.pop(GEMINI_API_KEY_CONFIG_KEY, None)
        safe_config.pop(OPENROUTER_API_KEY_CONFIG_KEY, None)
        logging.info("Settings applied: %s", str(safe_config))

        self.config_tree = normalize_payload_tree(self.config)

        try:
            self.apply_environment_overrides()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "Failed to apply environment overrides: %s",
                exc,
                exc_info=True,
            )

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

        sanitized_tree, warnings = coerce_with_defaults(
            candidate, self.default_config_tree
        )
        for warning in warnings:
            logging.warning(warning)

        self.config_tree = sanitized_tree
        self.config = flatten_config_tree(sanitized_tree)
        self._apply_runtime_overrides(
            applied_updates=filtered_updates,
            previous_config=previous_config,
        )
        self.save_config()

        changed_keys = {
            key
            for key, value in self.config.items()
            if previous_config.get(key) != value
        }
        return changed_keys, warnings

    def build_simple_mode_reset_payload(self) -> dict[str, Any]:
        """Return sanitized defaults for reverting to the simplified experience."""

        defaults = self.default_config
        payload: dict[str, Any] = {}
        for key in SIMPLE_MODE_RESET_KEYS:
            if key in defaults:
                payload[key] = copy.deepcopy(defaults[key])

        payload.setdefault(TEXT_CORRECTION_ENABLED_CONFIG_KEY, False)
        payload.setdefault(TEXT_CORRECTION_SERVICE_CONFIG_KEY, SERVICE_NONE)
        payload.setdefault(MAX_MEMORY_SECONDS_MODE_CONFIG_KEY, "auto")
        payload.setdefault(CHUNK_LENGTH_MODE_CONFIG_KEY, "auto")

        return payload

    def reset_to_defaults(self) -> tuple[dict[str, Any], set[str]]:
        """Reset the configuration to the baked-in defaults and persist them."""

        previous_config = copy.deepcopy(self.config)

        sanitized_tree, warnings = coerce_with_defaults(
            copy.deepcopy(self.default_config),
            self.default_config_tree,
        )
        for warning in warnings:
            logging.warning(warning)

        self.config_tree = sanitized_tree
        self.config = flatten_config_tree(sanitized_tree)
        self._apply_runtime_overrides(
            applied_updates=self.default_config,
            previous_config=previous_config,
        )
        self.save_config()

        changed_keys = {
            key
            for key, value in self.config.items()
            if previous_config.get(key) != value
        }

        return copy.deepcopy(self.config), changed_keys

    def _ensure_directory(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.error(
                log_context(
                    "Failed to prepare directory for configuration persistence.",
                    event="config.persistence.mkdir_failed",
                    directory=str(path.parent),
                    error=str(exc),
                ),
                exc_info=True,
            )
            raise

    def _maybe_migrate_profile_file(
        self,
        *,
        target: Path,
        candidates: tuple[Path, ...],
        label: str,
    ) -> str | None:
        for candidate in candidates:
            try:
                if candidate.resolve() == target.resolve():
                    return None
            except Exception:
                if str(candidate) == str(target):
                    return None
            if not candidate.exists():
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                LOGGER.warning(
                    "Unable to prepare destination for %s file at '%s': %s",
                    label,
                    target,
                    exc,
                )
                return None
            try:
                shutil.move(str(candidate), str(target))
            except Exception as exc:
                LOGGER.warning(
                    "Failed to migrate %s file from '%s' to '%s': %s",
                    label,
                    candidate,
                    target,
                    exc,
                )
                return None
            BOOTSTRAP_LOGGER.info(
                StructuredMessage(
                    f"{label.title()} file migrated to profile directory.",
                    event=f"config.bootstrap.{label}_migrated",
                    details={
                        "source": str(candidate),
                        "destination": str(target),
                    },
                )
            )
            return str(candidate)
        return None

    def _maybe_migrate_storage_paths(
        self,
        *,
        previous_storage_root: Path | None,
        new_storage_root: Path,
        previous_asr_path: Path | None,
        new_asr_path: Path,
        previous_recordings_path: Path | None,
        new_recordings_path: Path,
        asr_override: bool,
        recordings_override: bool,
        previous_deps_path: Path | None,
        new_deps_path: Path,
        deps_override: bool,
    ) -> None:
        if previous_storage_root is None:
            return

        try:
            storage_changed = previous_storage_root.resolve() != new_storage_root.resolve()
        except Exception:
            storage_changed = str(previous_storage_root) != str(new_storage_root)

        if not storage_changed:
            return

        if not asr_override:
            source_asr = previous_asr_path or previous_storage_root / "asr"
            self._relocate_directory(source_asr, new_asr_path, "ASR cache")

        if not recordings_override:
            source_recordings = (
                previous_recordings_path or previous_storage_root / "recordings"
            )
            self._relocate_directory(
                source_recordings,
                new_recordings_path,
                "Recordings",
            )

        if not deps_override:
            source_deps = previous_deps_path or previous_storage_root / "deps"
            self._relocate_directory(
                source_deps,
                new_deps_path,
                "Dependencies",
            )

    def _relocate_directory(self, source: Path, destination: Path, label: str) -> None:
        if source is None:
            return

        try:
            if source.resolve() == destination.resolve():
                return
        except Exception:
            if str(source) == str(destination):
                return

        if not source.exists():
            return

        if destination.exists():
            try:
                has_entries = any(destination.iterdir())
            except Exception:
                has_entries = True

            if has_entries:
                logging.info(
                    "%s directory already present at '%s'; skipping migration from '%s'.",
                    label,
                    destination,
                    source,
                )
                return

            try:
                destination.rmdir()
            except OSError:
                logging.info(
                    "%s directory already present at '%s'; skipping migration from '%s'.",
                    label,
                    destination,
                    source,
                )
                return

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logging.warning(
                "Unable to prepare destination for %s directory at '%s': %s",
                label.lower(),
                destination,
                exc,
            )
            return

        try:
            shutil.move(str(source), str(destination))
            logging.info(
                "%s directory migrated from '%s' to '%s'.",
                label,
                source,
                destination,
            )
        except Exception as exc:
            logging.warning(
                "Failed to migrate %s directory from '%s' to '%s': %s",
                label.lower(),
                source,
                destination,
                exc,
            )

    def _verify_persistence(
        self, *, config_changed: bool, secrets_changed: bool
    ) -> dict[str, dict[str, Any]]:
        report: dict[str, dict[str, Any]] = {}
        for changed, label, path in (
            (config_changed, "config", self.config_path),
            (secrets_changed, "secrets", self.secrets_path),
        ):
            status = {"path": str(path), "verified": False, "error": None}
            report[label] = status
            bootstrap_entry = self._bootstrap_state.get(label)
            if bootstrap_entry is not None:
                bootstrap_entry["path"] = path

            if not path.exists():
                status["error"] = "missing"
                LOGGER.error(
                    log_context(
                        f"{label.title()} file missing after save attempt.",
                        event="config.persistence.missing",
                        file=str(path),
                    )
                )
                if bootstrap_entry is not None:
                    bootstrap_entry["error"] = "missing"
                continue

            if not changed:
                status["verified"] = True
                if bootstrap_entry is not None:
                    bootstrap_entry["verified"] = True
                continue

            try:
                with path.open("r", encoding="utf-8") as handle:
                    json.load(handle)
            except json.JSONDecodeError as exc:
                status["error"] = f"invalid_json: {exc}"
                LOGGER.error(
                    log_context(
                        f"{label.title()} file contains invalid JSON after save attempt.",
                        event="config.persistence.invalid_json",
                        file=str(path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
                if bootstrap_entry is not None:
                    bootstrap_entry["error"] = str(exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive path
                status["error"] = str(exc)
                LOGGER.error(
                    log_context(
                        f"Unable to verify {label} file after save attempt.",
                        event="config.persistence.unreadable",
                        file=str(path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
                if bootstrap_entry is not None:
                    bootstrap_entry["error"] = str(exc)
                continue

            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None

            LOGGER.info(
                log_context(
                    f"{label.title()} file persisted successfully.",
                    event="config.persistence.verified",
                    file=str(path),
                    size=size_bytes,
                )
            )
            status["verified"] = True
            if bootstrap_entry is not None:
                bootstrap_entry["verified"] = True
                bootstrap_entry["error"] = None

        return report

    def get_bootstrap_report(self) -> dict[str, dict[str, Any]]:
        """Retorna uma cópia do estado de bootstrap para inspeção externa."""

        report: dict[str, dict[str, Any]] = {}
        for label, payload in self._bootstrap_state.items():
            report[label] = {
                "path": str(payload.get("path", "")),
                "existed": payload.get("existed", False),
                "created": payload.get("created", False),
                "written": payload.get("written", False),
                "verified": payload.get("verified", False),
                "error": payload.get("error"),
            }
        return report

    def register_startup_diagnostics(self, report: "StartupDiagnosticsReport") -> None:
        """Attach startup diagnostics to the bootstrap state and persist the report reference."""

        summary = report.to_dict()
        summary["user_messages"] = report.user_friendly_summary(include_success=False)
        self._bootstrap_state["diagnostics"] = summary
        self._startup_diagnostics_report = report
        BOOTSTRAP_LOGGER.info(
            StructuredMessage(
                "Startup diagnostics registered.",
                event="config.bootstrap.diagnostics_registered",
                has_errors=report.has_errors,
                has_warnings=report.has_warnings,
                has_fatal_errors=report.has_fatal_errors,
            )
        )

    def get_startup_diagnostics(self) -> "StartupDiagnosticsReport | None":
        """Return the latest startup diagnostics report when available."""

        return getattr(self, "_startup_diagnostics_report", None)

    def save_config(self) -> PersistenceOutcome:
        """Salva as configurações não sensíveis no config.json e as sensíveis no secrets.json."""

        config_to_save_flat = copy.deepcopy(self.config)
        secrets_to_save = {}

        secret_keys = [GEMINI_API_KEY_CONFIG_KEY, OPENROUTER_API_KEY_CONFIG_KEY]

        for key in secret_keys:
            if key in config_to_save_flat:
                secrets_to_save[key] = config_to_save_flat.pop(key)

        keys_to_ignore = [
            "tray_menu_items",
            "hotkey_manager",
            "asr_curated_catalog",
        ]
        for key in keys_to_ignore:
            if key in config_to_save_flat:
                del config_to_save_flat[key]

        config_to_save = normalize_payload_tree(config_to_save_flat)
        self.config_tree = copy.deepcopy(config_to_save)

        self._ensure_directory(self.config_path)
        self._ensure_directory(self.secrets_path)

        new_config_hash = self._compute_hash(config_to_save)
        config_changed = False
        config_error: str | None = None
        config_existed_before = self.config_path.exists()

        if new_config_hash != self._config_hash or not config_existed_before:
            temp_file_config = self.config_path.with_name(self.config_path.name + ".tmp")
            try:
                with temp_file_config.open("w", encoding="utf-8") as handle:
                    json.dump(config_to_save, handle, indent=4)
                os.replace(temp_file_config, self.config_path)
                self._config_hash = new_config_hash
                config_changed = True
                LOGGER.info(
                    log_context(
                        "Configuration saved to disk.",
                        event="config.save.success",
                        path=str(self.config_path),
                        first_run=not config_existed_before,
                        keys=len(config_to_save),
                    )
                )
            except Exception as exc:
                config_error = str(exc)
                LOGGER.error(
                    log_context(
                        "Error saving configuration file.",
                        event="config.save.failure",
                        path=str(self.config_path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
                if temp_file_config.exists():
                    try:
                        temp_file_config.unlink()
                    except OSError as cleanup_exc:  # pragma: no cover - defensive path
                        LOGGER.warning(
                            log_context(
                                "Failed to clean up temporary configuration file.",
                                event="config.save.cleanup_failed",
                                path=str(temp_file_config),
                                error=str(cleanup_exc),
                            ),
                            exc_info=True,
                        )
        else:
            LOGGER.debug(
                log_context(
                    "Configuration unchanged; skipping disk write.",
                    event="config.save.skipped",
                    path=str(self.config_path),
                )
            )

        temp_file_secrets = self.secrets_path.with_name(self.secrets_path.name + ".tmp")
        existing_secrets = {}
        if self.secrets_path.exists():
            try:
                with self.secrets_path.open("r", encoding="utf-8") as handle:
                    existing_secrets = json.load(handle)
            except json.JSONDecodeError:
                LOGGER.warning(
                    log_context(
                        "Secrets file is corrupted; overwriting with sanitized values.",
                        event="config.secrets.save.invalid_existing",
                        path=str(self.secrets_path),
                    )
                )
                existing_secrets = {}
            except FileNotFoundError:
                existing_secrets = {}
            except Exception as exc:  # pragma: no cover - defensive path
                LOGGER.error(
                    log_context(
                        "Unexpected error while reading secrets file before save.",
                        event="config.secrets.save.read_failure",
                        path=str(self.secrets_path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
                existing_secrets = {}

        existing_secrets.update(secrets_to_save)
        new_secrets_hash = self._compute_hash(existing_secrets)
        secrets_existed_before = self.secrets_path.exists()
        should_write_secrets = (
            new_secrets_hash != self._secrets_hash or not secrets_existed_before
        )
        secrets_changed = False
        secrets_error: str | None = None

        if should_write_secrets:
            try:
                with temp_file_secrets.open("w", encoding="utf-8") as handle:
                    json.dump(existing_secrets, handle, indent=4)
                os.replace(temp_file_secrets, self.secrets_path)
                self._secrets_hash = new_secrets_hash
                secrets_changed = True
                LOGGER.info(
                    log_context(
                        "Secrets saved to disk.",
                        event="config.secrets.save.success",
                        path=str(self.secrets_path),
                        first_run=not secrets_existed_before,
                        keys=len(existing_secrets),
                    )
                )
            except Exception as exc:
                secrets_error = str(exc)
                LOGGER.error(
                    log_context(
                        "Error saving secrets file.",
                        event="config.secrets.save.failure",
                        path=str(self.secrets_path),
                        error=str(exc),
                    ),
                    exc_info=True,
                )
                if temp_file_secrets.exists():
                    try:
                        temp_file_secrets.unlink()
                    except OSError as cleanup_exc:  # pragma: no cover - defensive path
                        LOGGER.warning(
                            log_context(
                                "Failed to clean up temporary secrets file.",
                                event="config.secrets.save.cleanup_failed",
                                path=str(temp_file_secrets),
                                error=str(cleanup_exc),
                            ),
                            exc_info=True,
                        )
        else:
            LOGGER.debug(
                log_context(
                    "Secrets unchanged; skipping disk write.",
                    event="config.secrets.save.skipped",
                    path=str(self.secrets_path),
                )
            )

        verification = self._verify_persistence(
            config_changed=config_changed,
            secrets_changed=secrets_changed,
        )

        config_entry = self._bootstrap_state["config"]
        secrets_entry = self._bootstrap_state["secrets"]

        config_entry["existed"] = config_entry.get("existed", False) or config_existed_before
        secrets_entry["existed"] = secrets_entry.get("existed", False) or secrets_existed_before

        config_created = (
            not config_existed_before
            and config_changed
            and self.config_path.exists()
        )
        secrets_created = (
            not secrets_existed_before
            and secrets_changed
            and self.secrets_path.exists()
        )

        config_entry["created"] = config_entry.get("created", False) or config_created
        secrets_entry["created"] = secrets_entry.get("created", False) or secrets_created

        config_entry["written"] = config_entry.get("written", False) or config_changed
        secrets_entry["written"] = secrets_entry.get("written", False) or secrets_changed

        config_verification = verification.get("config", {})
        secrets_verification = verification.get("secrets", {})

        config_entry["verified"] = config_entry.get("verified", False) or config_verification.get("verified", False)
        secrets_entry["verified"] = secrets_entry.get("verified", False) or secrets_verification.get("verified", False)

        verification_error_config = config_verification.get("error")
        verification_error_secrets = secrets_verification.get("error")

        if config_error:
            config_entry["error"] = config_error
        elif verification_error_config:
            config_entry["error"] = verification_error_config
        else:
            config_entry["error"] = None

        if secrets_error:
            secrets_entry["error"] = secrets_error
        elif verification_error_secrets:
            secrets_entry["error"] = verification_error_secrets
        else:
            secrets_entry["error"] = None

        return PersistenceOutcome(
            config=PersistenceRecord(
                path=self.config_path,
                existed_before=config_existed_before,
                wrote=config_changed,
                created=config_created,
                verified=config_verification.get("verified", False),
                error=config_entry.get("error"),
            ),
            secrets=PersistenceRecord(
                path=self.secrets_path,
                existed_before=secrets_existed_before,
                wrote=secrets_changed,
                created=secrets_created,
                verified=secrets_verification.get("verified", False),
                error=secrets_entry.get("error"),
            ),
        )

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

    def get_retry_attempts(self, key: str, default: int) -> int:
        """Retorna o número máximo de tentativas configurado para a chave."""
        value = self.get(key, default)
        try:
            attempts = int(value)
            if attempts <= 0:
                raise ValueError
        except (TypeError, ValueError):
            cached_value = self._invalid_retry_cache.get(key)
            if value != cached_value:
                logging.warning(
                    "Invalid retry attempts '%s' for key '%s'; using default %s attempts.",
                    value,
                    key,
                    int(default),
                )
                self._invalid_retry_cache[key] = value
            return int(default)
        else:
            if key in self._invalid_retry_cache:
                self._invalid_retry_cache.pop(key, None)
            return attempts

    def set(self, key, value):
        if key == ASR_BACKEND_CONFIG_KEY:
            value = _normalize_asr_backend(value)
        self.config[key] = value

    def is_first_run(self) -> bool:
        """Return ``True`` when the onboarding wizard must be executed."""

        return not bool(self.config.get(FIRST_RUN_COMPLETED_CONFIG_KEY, False))

    def mark_first_run_complete(self, *, persist: bool = True) -> None:
        """Persistently mark the onboarding flow as completed."""

        self.config[FIRST_RUN_COMPLETED_CONFIG_KEY] = True
        if persist:
            self.save_config()

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

    def get_asr_ct2_compute_type(self):
        return self.config.get(
            ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
            self.default_config[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY],
        )

    def set_asr_ct2_compute_type(self, value: str):
        self.config[ASR_CT2_COMPUTE_TYPE_CONFIG_KEY] = str(value)

    def get_storage_root_dir(self) -> str:
        return self.config.get(
            STORAGE_ROOT_DIR_CONFIG_KEY,
            self.default_config[STORAGE_ROOT_DIR_CONFIG_KEY],
        )

    def set_storage_root_dir(self, value: str):
        old_root = self.config.get(STORAGE_ROOT_DIR_CONFIG_KEY)
        new_root = os.path.expanduser(str(value))
        self.config[STORAGE_ROOT_DIR_CONFIG_KEY] = new_root

        try:
            new_root_path = Path(new_root).expanduser()
        except Exception:
            return

        try:
            old_root_path = Path(old_root).expanduser() if old_root else None
        except Exception:
            old_root_path = None

        derived_dirs = (
            (MODELS_STORAGE_DIR_CONFIG_KEY, Path()),
            (ASR_CACHE_DIR_CONFIG_KEY, Path("asr")),
            (RECORDINGS_DIR_CONFIG_KEY, Path("recordings")),
            (PYTHON_PACKAGES_DIR_CONFIG_KEY, Path("python_packages")),
            (VAD_MODELS_DIR_CONFIG_KEY, Path("vad")),
            (HF_CACHE_DIR_CONFIG_KEY, Path("hf_cache")),
        )

        for key, suffix in derived_dirs:
            current = self.config.get(key)
            if not current:
                continue
            try:
                current_path = Path(str(current)).expanduser()
            except Exception:
                continue
            if old_root_path is not None:
                expected_old = (old_root_path / suffix).resolve() if suffix else old_root_path.resolve()
                if current_path.resolve() != expected_old:
                    continue
            updated_path = (new_root_path / suffix) if suffix else new_root_path
            self.config[key] = str(updated_path)

    def get_models_storage_dir(self) -> str:
        return self.config.get(
            MODELS_STORAGE_DIR_CONFIG_KEY,
            self.default_config.get(
                MODELS_STORAGE_DIR_CONFIG_KEY,
                self.default_config.get(STORAGE_ROOT_DIR_CONFIG_KEY),
            ),
        )

    def set_models_storage_dir(self, value: str):
        self.config[MODELS_STORAGE_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_deps_install_dir(self) -> str:
        return self.config.get(
            DEPS_INSTALL_DIR_CONFIG_KEY,
            self.default_config.get(
                DEPS_INSTALL_DIR_CONFIG_KEY,
                _DEFAULT_DEPS_INSTALL_DIR,
            ),
        )

    def set_deps_install_dir(self, value: str):
        self.config[DEPS_INSTALL_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_python_packages_dir(self) -> str:
        return self.config.get(
            PYTHON_PACKAGES_DIR_CONFIG_KEY,
            self.default_config.get(
                PYTHON_PACKAGES_DIR_CONFIG_KEY,
                _DEFAULT_PYTHON_PACKAGES_DIR,
            ),
        )

    def set_python_packages_dir(self, value: str):
        self.config[PYTHON_PACKAGES_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_hf_home_dir(self) -> str:
        return self.config.get(
            HF_HOME_DIR_CONFIG_KEY,
            self.default_config.get(
                HF_HOME_DIR_CONFIG_KEY,
                _DEFAULT_HF_HOME_DIR,
            ),
        )

    def set_hf_home_dir(self, value: str):
        self.config[HF_HOME_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_hf_cache_dir(self) -> str:
        return self.config.get(
            HF_CACHE_DIR_CONFIG_KEY,
            self.default_config.get(
                HF_CACHE_DIR_CONFIG_KEY,
                _DEFAULT_HF_CACHE_DIR,
            ),
        )

    def set_hf_cache_dir(self, value: str):
        self.config[HF_CACHE_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_recordings_dir(self) -> str:
        return self.config.get(
            RECORDINGS_DIR_CONFIG_KEY,
            self.default_config[RECORDINGS_DIR_CONFIG_KEY],
        )

    def set_recordings_dir(self, value: str):
        self.config[RECORDINGS_DIR_CONFIG_KEY] = os.path.expanduser(str(value))

    def get_environment_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        hf_home = self.get_hf_home_dir()
        if hf_home:
            overrides["HF_HOME"] = os.path.expanduser(str(hf_home))
        models_dir = self.get_models_storage_dir()
        if models_dir:
            overrides["WHISPER_FLASH_MODELS_DIR"] = os.path.expanduser(
                str(models_dir)
            )
        deps_dir = self.get_deps_install_dir()
        if deps_dir:
            overrides["WHISPER_FLASH_DEPS_DIR"] = os.path.expanduser(str(deps_dir))
        return overrides

    def apply_environment_overrides(self) -> dict[str, str]:
        overrides = self.get_environment_overrides()
        applied: dict[str, str] = {}
        for key, value in overrides.items():
            if not value:
                continue
            try:
                Path(value).expanduser().mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                LOGGER.warning(
                    "Unable to prepare directory for environment override %s=%s: %s",
                    key,
                    value,
                    exc,
                )
                continue
            os.environ[key] = value
            applied[key] = value
        if applied:
            summary = ", ".join(f"{k}={v}" for k, v in applied.items())
            LOGGER.info("Environment overrides applied: %s", summary)
        return applied

    def validate_directory_candidate(
        self,
        candidate: str | Path,
        *,
        label: str,
        create: bool = True,
    ) -> tuple[bool, str, Path | None]:
        try:
            path = Path(candidate).expanduser()
        except Exception as exc:
            return False, f"Invalid {label.lower()} directory: {exc}", None

        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                return (
                    False,
                    f"Unable to create {label.lower()} directory at '{path}': {exc}",
                    None,
                )

        if not os.access(path, os.W_OK):
            return (
                False,
                f"No write permission for {label.lower()} directory '{path}'.",
                path,
            )

        try:
            usage = shutil.disk_usage(path)
        except FileNotFoundError:
            usage = shutil.disk_usage(path.parent)

        free_gb = usage.free / (1024 ** 3)
        message = (
            f"{label} directory ready at '{path}'. Free space available: {free_gb:.2f} GB"
        )
        return True, message, path

    def migrate_directory(
        self,
        source: str | Path,
        destination: str | Path,
        *,
        label: str,
    ) -> bool:
        try:
            source_path = Path(source).expanduser()
            destination_path = Path(destination).expanduser()
        except Exception as exc:
            LOGGER.warning("Invalid %s migration paths: %s", label.lower(), exc)
            return False

        try:
            if source_path.resolve() == destination_path.resolve():
                LOGGER.info(
                    "%s migration skipped because source and destination match: %s.",
                    label,
                    source_path,
                )
                return True
        except Exception:
            if str(source_path) == str(destination_path):
                LOGGER.info(
                    "%s migration skipped because source and destination match: %s.",
                    label,
                    source_path,
                )
                return True

        if not source_path.exists():
            LOGGER.info(
                "%s directory '%s' does not exist; nothing to migrate.",
                label,
                source_path,
            )
            return True

        try:
            total_bytes = 0
            for entry in source_path.rglob("*"):
                if entry.is_file():
                    total_bytes += entry.stat().st_size
        except Exception as exc:
            LOGGER.warning(
                "Unable to compute size for %s directory '%s': %s",
                label.lower(),
                source_path,
                exc,
            )
            total_bytes = 0

        try:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            LOGGER.warning(
                "Unable to prepare destination for %s directory at '%s': %s",
                label.lower(),
                destination_path,
                exc,
            )
            return False

        safety_margin = max(int(total_bytes * 0.1), 256 * 1024 * 1024)
        required_bytes = total_bytes + safety_margin
        try:
            usage = shutil.disk_usage(destination_path.parent)
        except FileNotFoundError:
            usage = shutil.disk_usage(destination_path)
        free_bytes = usage.free

        def _fmt_bytes(value: int) -> str:
            units = ["B", "KB", "MB", "GB", "TB"]
            size = float(value)
            for unit in units:
                if size < 1024.0 or unit == units[-1]:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{value} B"

        if total_bytes and free_bytes < required_bytes:
            LOGGER.error(
                "Insufficient free space to migrate %s directory: required %s but only %s available.",
                label.lower(),
                _fmt_bytes(required_bytes),
                _fmt_bytes(free_bytes),
            )
            return False

        if destination_path.exists():
            try:
                has_entries = any(destination_path.iterdir())
            except Exception:
                has_entries = True
            if has_entries:
                LOGGER.warning(
                    "%s directory already exists at '%s'; aborting migration from '%s'.",
                    label,
                    destination_path,
                    source_path,
                )
                return False

        try:
            shutil.move(str(source_path), str(destination_path))
        except Exception as exc:
            LOGGER.warning(
                "Failed to migrate %s directory from '%s' to '%s': %s",
                label.lower(),
                source_path,
                destination_path,
                exc,
            )
            return False

        LOGGER.info(
            "%s directory migrated from '%s' to '%s'.",
            label,
            source_path,
            destination_path,
        )
        return True

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
        target_dir: str | None = None,
        bytes_downloaded: int | None = None,
        throughput_bytes_per_sec: float | None = None,
        duration_seconds: float | None = None,
        task_id: str | None = None,
        timestamp: datetime | None = None,
        save: bool = True,
        history_limit: int = 50,
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
        if target_dir:
            payload["target_dir"] = str(target_dir)
        if bytes_downloaded is not None:
            try:
                payload["bytes_downloaded"] = int(bytes_downloaded)
            except (TypeError, ValueError):
                payload["bytes_downloaded"] = None
        if throughput_bytes_per_sec is not None:
            try:
                payload["throughput_bps"] = float(throughput_bytes_per_sec)
            except (TypeError, ValueError):
                payload["throughput_bps"] = None
        if duration_seconds is not None:
            try:
                payload["duration_seconds"] = float(duration_seconds)
            except (TypeError, ValueError):
                payload["duration_seconds"] = None
        if task_id:
            payload["task_id"] = str(task_id)
        self.set_last_asr_download_status(payload)

        history_payload = dict(payload)
        history_entries = self.config.get("asr_download_history")
        if not isinstance(history_entries, list):
            history_entries = []
        history_entries.append(history_payload)
        if history_limit > 0:
            history_entries = history_entries[-history_limit:]
        self.config["asr_download_history"] = history_entries
        if save:
            self.save_config()

    def get_model_download_history(self, limit: int | None = None) -> list[dict[str, object]]:
        history_entries = self.config.get("asr_download_history", [])
        if not isinstance(history_entries, list):
            return []
        normalized: list[dict[str, object]] = []
        for entry in history_entries:
            if isinstance(entry, dict):
                normalized.append(dict(entry))
        if limit is not None and limit > 0:
            return normalized[-limit:]
        return normalized

    def get_asr_curated_catalog(self):
        return self.config.get(
            ASR_CURATED_CATALOG_CONFIG_KEY,
            self.default_config[ASR_CURATED_CATALOG_CONFIG_KEY],
        )

    def get_max_parallel_downloads(self) -> int:
        value = self.config.get("max_parallel_downloads", self.default_config.get("max_parallel_downloads", 1))
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return int(self.default_config.get("max_parallel_downloads", 1))
        return max(1, normalized)

    def set_max_parallel_downloads(self, value: int) -> None:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = int(self.default_config.get("max_parallel_downloads", 1))
        if normalized < 1:
            normalized = 1
        self.config["max_parallel_downloads"] = normalized

    def set_asr_curated_catalog(self, value: list):
        if isinstance(value, list):
            self.config[ASR_CURATED_CATALOG_CONFIG_KEY] = value
        else:
            self.config[ASR_CURATED_CATALOG_CONFIG_KEY] = self.default_config[
                ASR_CURATED_CATALOG_CONFIG_KEY
            ]

    def get_runtime_model_catalog(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._runtime_model_catalog)

    def get_runtime_recommendation(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._runtime_recommendation) if self._runtime_recommendation else None

    def get_runtime_hardware_profile(self) -> HardwareProfile | None:
        return self._runtime_hardware_profile

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

    def get_recordings_dir(self) -> str:
        return self.config.get(
            RECORDINGS_DIR_CONFIG_KEY,
            self.default_config[RECORDINGS_DIR_CONFIG_KEY],
        )

    def set_recordings_dir(self, value: str) -> None:
        self.config[RECORDINGS_DIR_CONFIG_KEY] = os.path.expanduser(str(value))


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
            self.default_config.get(CHUNK_LENGTH_MODE_CONFIG_KEY, "auto"),
        )

    def set_chunk_length_mode(self, value: str):
        val = str(value).lower()
        if val not in ["auto", "manual"]:
            val = "auto"
        self.config[CHUNK_LENGTH_MODE_CONFIG_KEY] = val

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

    # ------------------------------------------------------------------
    # Runtime notices

    def record_runtime_notice(
        self,
        message: str,
        *,
        category: str = "general",
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persiste um aviso operacional na memória do gerenciador.

        O objetivo é expor mensagens de diagnóstico efêmeras (como o
        relatório de auditoria de dependências) sem poluir o arquivo de
        configuração persistente.
        """

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "category": str(category),
            "message": str(message),
        }
        if details is not None:
            payload["details"] = dict(details)
        self._runtime_notices.append(payload)
        return payload

    def get_runtime_notices(self) -> list[dict[str, Any]]:
        """Retorna uma cópia dos avisos de runtime registrados."""

        return list(self._runtime_notices)
