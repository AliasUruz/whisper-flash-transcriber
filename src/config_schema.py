"""Pydantic schemas for validating application configuration."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from .logging_utils import get_logger, log_context
from .model_manager import CURATED, list_catalog, normalize_backend_label

LOGGER = get_logger(__name__, component='ConfigSchema')


_DEFAULT_STORAGE_ROOT = (Path.home() / ".cache" / "whisper_flash_transcriber").expanduser()
_DEFAULT_MODELS_STORAGE_DIR = str((_DEFAULT_STORAGE_ROOT / "models").expanduser())
_DEFAULT_RECORDINGS_DIR = str((_DEFAULT_STORAGE_ROOT / "recordings").expanduser())
_DEFAULT_ASR_CACHE_DIR = str((_DEFAULT_STORAGE_ROOT / "asr").expanduser())
_DEFAULT_DEPS_INSTALL_DIR = str((_DEFAULT_STORAGE_ROOT / "deps").expanduser())
_DEFAULT_HF_HOME_DIR = str((Path(_DEFAULT_DEPS_INSTALL_DIR) / "huggingface").expanduser())
_DEFAULT_TRANSFORMERS_CACHE_DIR = str(
    (Path(_DEFAULT_DEPS_INSTALL_DIR) / "transformers").expanduser()
)
_DEFAULT_PYTHON_PACKAGES_DIR = str((_DEFAULT_STORAGE_ROOT / "python_packages").expanduser())
_DEFAULT_VAD_MODELS_DIR = str((_DEFAULT_STORAGE_ROOT / "vad").expanduser())
_DEFAULT_HF_CACHE_DIR = str((_DEFAULT_STORAGE_ROOT / "hf_cache").expanduser())
_SUPPORTED_UI_LANGUAGE_MAP = {
    "en": "en-US",
    "en-us": "en-US",
    "english": "en-US",
    "pt": "pt-BR",
    "pt-br": "pt-BR",
    "pt_br": "pt-BR",
    "portuguese": "pt-BR",
    "português": "pt-BR",
}
_DEFAULT_UI_LANGUAGE = "en-US"

_CURATED_MODEL_IDS = {entry["id"] for entry in CURATED}
_ALLOWED_ASR_BACKENDS = {"ctranslate2"}


def _normalize_lower(value: str, *, allowed: set[str], field_name: str) -> str:
    lowered = value.lower()
    if lowered not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return lowered


def _coerce_key(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _normalize_auto_paste_modifier(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or "auto"
    if isinstance(value, (list, tuple, set)):
        normalized_parts: list[str] = []
        for item in value:
            if item is None:
                continue
            item_str = str(item).strip()
            if item_str:
                normalized_parts.append(item_str)
        if not normalized_parts:
            return "auto"
        return "+".join(normalized_parts)
    return str(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError("Value must be boolean or None")


def _expand_path(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return ""
        return str(Path(normalized).expanduser())
    if isinstance(value, Path):
        return str(value.expanduser())
    raise ValueError("Paths must be provided as strings or Path objects")


class ASRDownloadStatus(BaseModel):
    """Structured status for the last ASR download attempt."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    status: str = "unknown"
    timestamp: str = ""
    model_id: str = ""
    backend: str = ""
    message: str = ""
    details: str = ""
    target_dir: str = ""
    bytes_downloaded: int | None = None
    throughput_bps: float | None = None
    duration_seconds: float | None = None
    task_id: str | None = None


class ASRDownloadHistoryEntry(ASRDownloadStatus):
    """Historical record capturing metadata of a download attempt."""

    pass


class ASRPromptDecision(BaseModel):
    """Persisted record of the last prompt decision for model downloads."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    model_id: str = ""
    backend: str = ""
    decision: str = ""
    timestamp: int = 0


class SoundSettings(BaseModel):
    """Top-level audio feedback preferences for the minimal workflow."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    enabled: bool = True
    frequency: int = Field(default=400, ge=0)
    duration: float = Field(default=0.3, ge=0.0)
    volume: float = Field(default=0.5, ge=0.0)


class AdvancedHotkeyConfig(BaseModel):
    """Advanced toggles related to auxiliary hotkeys and modifiers."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    agent_key: str = "F4"
    agent_auto_paste: bool | None = True
    auto_paste_modifier: str = "auto"
    hotkey_stability_service_enabled: bool = True
    keyboard_library: str = "win32"

    @field_validator("agent_key", mode="before")
    @classmethod
    def _coerce_agent_key(cls, value: Any) -> str:
        return _coerce_key(value)

    @field_validator("auto_paste_modifier", mode="before")
    @classmethod
    def _normalize_modifier(cls, value: Any) -> str:
        return _normalize_auto_paste_modifier(value)

    @field_validator("agent_auto_paste", mode="before")
    @classmethod
    def _coerce_optional_bool(cls, value: Any) -> bool | None:
        return _optional_bool(value)


class AdvancedAIConfig(BaseModel):
    """Optional AI post-processing and agent integrations."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    text_correction_enabled: bool = False
    text_correction_service: str = "none"
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324:free"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_agent_model: str = "gemini-2.5-flash-lite"
    openrouter_timeout: int = Field(default=30, ge=1)
    gemini_timeout: int = Field(default=120, ge=1)
    openrouter_prompt: str = ""
    prompt_agentico: str = (
        "You are an AI assistant that executes text commands. "
        "The user will provide an instruction followed by the text to be processed. "
        "Your task is to execute the instruction on the text and return ONLY the final result. "
        "Do not add explanations, greetings, or any extra text. "
        "The output language should match the main language of the provided text."
    )
    gemini_prompt: str = (
        "You are a meticulous speech-to-text correction AI. "
        "Your primary task is to correct punctuation, capitalization, and minor transcription errors in the text below "
        "while preserving the original content and structure as closely as possible. "
        "Key instructions: - Correct punctuation, such as adding commas, periods, and question marks. "
        "- Fix capitalization at the beginning of sentences. "
        "- Remove only obvious speech disfluencies (e.g., \"I-I mean\"). "
        "- DO NOT summarize, paraphrase, or change the original meaning. "
        "- Return ONLY the corrected text, with no additional comments or explanations. "
        "Transcribed speech: {text}"
    )
    gemini_model_options: list[str] = Field(
        default_factory=lambda: [
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
    )

    @field_validator("text_correction_service", mode="before")
    @classmethod
    def _validate_text_service(cls, value: Any) -> str:
        if isinstance(value, str):
            return _normalize_lower(
                value,
                allowed={"none", "openrouter", "gemini"},
                field_name="text_correction_service",
            )
        raise ValueError("text_correction_service must be a string")

    @field_validator("gemini_model_options", mode="before")
    @classmethod
    def _coerce_model_options(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            coerced: list[str] = []
            for item in value:
                if item is None:
                    continue
                coerced.append(str(item).strip())
            return coerced
        return [str(value)]


class AdvancedPerformanceConfig(BaseModel):
    """Performance tunables kept behind the advanced namespace."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    batch_size: int = Field(default=16, ge=1)
    batch_size_mode: str = "auto"
    manual_batch_size: int = Field(default=8, ge=1)
    gpu_index: int = Field(default=0, ge=-1)
    chunk_length_sec: float = Field(default=30.0, ge=0.0)
    chunk_length_mode: str = "manual"
    enable_torch_compile: bool = False
    clear_gpu_cache: bool = True
    asr_compute_device: str = "auto"
    asr_dtype: str = "float16"
    asr_ct2_compute_type: str = "int8_float16"
    asr_ct2_cpu_threads: int | None = None
    max_parallel_downloads: int = Field(default=1, ge=1, le=8)

    @field_validator("batch_size_mode", "chunk_length_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: Any, info: ValidationInfo) -> str:
        field_name = "batch_size_mode"
        if info is not None and info.field_name is not None:
            field_name = info.field_name
        if isinstance(value, str):
            return _normalize_lower(value, allowed={"auto", "manual"}, field_name=field_name)
        raise ValueError(f"{field_name} must be a string")


class AdvancedStorageConfig(BaseModel):
    """Extended storage policies and cache placement overrides."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    record_storage_mode: str = "auto"
    record_storage_limit: int = Field(default=0, ge=0)
    max_memory_seconds_mode: str = "manual"
    max_memory_seconds: float = Field(default=30.0, ge=0.0)
    min_free_ram_mb: int = Field(default=1000, ge=0)
    auto_ram_threshold_percent: int = Field(default=10, ge=1, le=50)
    save_temp_recordings: bool = False
    storage_root_dir: str = str(_DEFAULT_STORAGE_ROOT)
    models_storage_dir: str = _DEFAULT_MODELS_STORAGE_DIR
    recordings_dir: str = _DEFAULT_RECORDINGS_DIR
    asr_cache_dir: str = _DEFAULT_ASR_CACHE_DIR
    deps_install_dir: str = _DEFAULT_DEPS_INSTALL_DIR
    hf_home_dir: str = _DEFAULT_HF_HOME_DIR
    transformers_cache_dir: str = _DEFAULT_TRANSFORMERS_CACHE_DIR
    python_packages_dir: str = _DEFAULT_PYTHON_PACKAGES_DIR
    vad_models_dir: str = _DEFAULT_VAD_MODELS_DIR
    hf_cache_dir: str = _DEFAULT_HF_CACHE_DIR

    @field_validator("record_storage_mode", mode="before")
    @classmethod
    def _validate_storage_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "hybrid":
                LOGGER.info(
                    log_context(
                        "Mapping legacy record_storage_mode 'hybrid' to 'auto'.",
                        event="config.legacy_storage_mode_mapped",
                    )
                )
                lowered = "auto"
            return _normalize_lower(
                lowered,
                allowed={"disk", "memory", "auto"},
                field_name="record_storage_mode",
            )
        raise ValueError("record_storage_mode must be a string")

    @field_validator("max_memory_seconds_mode", mode="before")
    @classmethod
    def _validate_memory_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return _normalize_lower(
                value,
                allowed={"manual", "auto"},
                field_name="max_memory_seconds_mode",
            )
        raise ValueError("max_memory_seconds_mode must be a string")

    @field_validator(
        "storage_root_dir",
        "models_storage_dir",
        "recordings_dir",
        "asr_cache_dir",
        "deps_install_dir",
        "hf_home_dir",
        "transformers_cache_dir",
        "python_packages_dir",
        "vad_models_dir",
        "hf_cache_dir",
        mode="before",
    )
    @classmethod
    def _expand_dirs(cls, value: Any) -> str:
        return _expand_path(value)


class AdvancedVADConfig(BaseModel):
    """Voice activity detection parameters exposed to power users."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    use_vad: bool = False
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_silence_duration: float = Field(default=1.0, ge=0.0)
    vad_pre_speech_padding_ms: int = Field(default=150, ge=0)
    vad_post_speech_padding_ms: int = Field(default=300, ge=0)

    @field_validator("vad_pre_speech_padding_ms", "vad_post_speech_padding_ms", mode="before")
    @classmethod
    def _coerce_padding(cls, value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(float(value))
        raise ValueError("Padding value must be numeric")


class AdvancedWorkflowConfig(BaseModel):
    """UI-level toggles that are not required for the minimal workflow."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    display_transcripts_in_terminal: bool = False


class AdvancedSystemConfig(BaseModel):
    """System integration flags that remain opt-in."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    launch_at_startup: bool = False


class AdvancedConfig(BaseModel):
    """Namespace that groups every optional/advanced configuration knob."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    hotkeys: AdvancedHotkeyConfig = Field(default_factory=AdvancedHotkeyConfig)
    ai: AdvancedAIConfig = Field(default_factory=AdvancedAIConfig)
    performance: AdvancedPerformanceConfig = Field(default_factory=AdvancedPerformanceConfig)
    storage: AdvancedStorageConfig = Field(default_factory=AdvancedStorageConfig)
    vad: AdvancedVADConfig = Field(default_factory=AdvancedVADConfig)
    workflow: AdvancedWorkflowConfig = Field(default_factory=AdvancedWorkflowConfig)
    system: AdvancedSystemConfig = Field(default_factory=AdvancedSystemConfig)


class AppConfig(BaseModel):
    """Application configuration validated via Pydantic."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    record_key: str = "F3"
    record_mode: str = "toggle"
    auto_paste: bool = True
    min_record_duration: float = Field(default=0.5, ge=0.0)
    min_transcription_duration: float = Field(default=1.0, ge=0.0)
    sound: SoundSettings = Field(default_factory=SoundSettings)
    asr_model_id: str = "distil-whisper/distil-large-v3"
    asr_backend: str = "ctranslate2"
    ui_language: str = _DEFAULT_UI_LANGUAGE
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    asr_installed_models: list[str] = Field(default_factory=list)
    asr_curated_catalog: list[dict[str, Any]] = Field(default_factory=list_catalog)
    asr_curated_catalog_url: str = ""
    asr_last_download_status: ASRDownloadStatus = Field(default_factory=ASRDownloadStatus)
    asr_download_history: list[ASRDownloadHistoryEntry] = Field(default_factory=list)
    asr_last_prompt_decision: ASRPromptDecision = Field(default_factory=ASRPromptDecision)
    first_run_completed: bool = False

    @field_validator("record_key", mode="before")
    @classmethod
    def _coerce_record_key(cls, value: Any) -> str:
        return _coerce_key(value)

    @field_validator("ui_language", mode="before")
    @classmethod
    def _normalize_ui_language(cls, value: Any) -> str:
        if value is None:
            return _DEFAULT_UI_LANGUAGE
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return _DEFAULT_UI_LANGUAGE
            mapped = _SUPPORTED_UI_LANGUAGE_MAP.get(normalized.lower())
            if mapped:
                return mapped
            raise ValueError(
                f"ui_language must be one of {sorted(set(_SUPPORTED_UI_LANGUAGE_MAP.values()))}"
            )
        raise ValueError("ui_language must be a string")

    @field_validator("record_mode", mode="before")
    @classmethod
    def _validate_record_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return _normalize_lower(value, allowed={"toggle", "press"}, field_name="record_mode")
        raise ValueError("record_mode must be a string")


    @field_validator("asr_installed_models", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if item is not None]
        return [str(value)]

    @field_validator("asr_model_id", mode="before")
    @classmethod
    def _validate_asr_model_id(cls, value: Any, info: ValidationInfo) -> str:
        if not isinstance(value, str):
            raise ValueError("asr_model_id must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("asr_model_id must not be empty")

        allowed_ids = set(_CURATED_MODEL_IDS)
        runtime_ids: set[str] = set()

        def _extract_ids(source: Any) -> None:
            if isinstance(source, list):
                for entry in source:
                    if isinstance(entry, dict):
                        candidate = entry.get("model_id") or entry.get("id")
                        if isinstance(candidate, str):
                            candidate = candidate.strip()
                            if candidate:
                                runtime_ids.add(candidate)
                    elif isinstance(entry, str):
                        candidate = entry.strip()
                        if candidate:
                            runtime_ids.add(candidate)

        if info is not None:
            data = info.data or {}
            _extract_ids(data.get("asr_curated_catalog"))
            _extract_ids(data.get("asr_installed_models"))

            context = getattr(info, "context", None) or {}
            if isinstance(context, dict):
                _extract_ids(context.get("asr_curated_catalog"))
                _extract_ids(context.get("asr_installed_models"))

        if normalized in allowed_ids or normalized in runtime_ids:
            return normalized

        LOGGER.debug(
            "Accepting non-curated ASR model id '%s' during config validation.",
            normalized,
        )
        return normalized

    @field_validator("asr_backend", mode="before")
    @classmethod
    def _validate_asr_backend(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("asr_backend must be a string")
        normalized = normalize_backend_label(value)
        if normalized == "auto":
            normalized = "ctranslate2"
        if normalized not in _ALLOWED_ASR_BACKENDS:
            raise ValueError(
                f"asr_backend must be one of {sorted(_ALLOWED_ASR_BACKENDS)}"
            )
        return normalized


KEY_PATH_OVERRIDES: dict[str, tuple[str, ...]] = {
    # Sound namespace
    "sound_enabled": ("sound", "enabled"),
    "sound_frequency": ("sound", "frequency"),
    "sound_duration": ("sound", "duration"),
    "sound_volume": ("sound", "volume"),
    # Advanced hotkeys
    "agent_key": ("advanced", "hotkeys", "agent_key"),
    "agent_auto_paste": ("advanced", "hotkeys", "agent_auto_paste"),
    "auto_paste_modifier": ("advanced", "hotkeys", "auto_paste_modifier"),
    "hotkey_stability_service_enabled": ("advanced", "hotkeys", "hotkey_stability_service_enabled"),
    "keyboard_library": ("advanced", "hotkeys", "keyboard_library"),
    # Advanced AI
    "text_correction_enabled": ("advanced", "ai", "text_correction_enabled"),
    "text_correction_service": ("advanced", "ai", "text_correction_service"),
    "openrouter_api_key": ("advanced", "ai", "openrouter_api_key"),
    "openrouter_model": ("advanced", "ai", "openrouter_model"),
    "openrouter_timeout": ("advanced", "ai", "openrouter_timeout"),
    "openrouter_prompt": ("advanced", "ai", "openrouter_prompt"),
    "gemini_api_key": ("advanced", "ai", "gemini_api_key"),
    "gemini_model": ("advanced", "ai", "gemini_model"),
    "gemini_agent_model": ("advanced", "ai", "gemini_agent_model"),
    "gemini_timeout": ("advanced", "ai", "gemini_timeout"),
    "gemini_prompt": ("advanced", "ai", "gemini_prompt"),
    "prompt_agentico": ("advanced", "ai", "prompt_agentico"),
    "gemini_model_options": ("advanced", "ai", "gemini_model_options"),
    # Advanced performance
    "batch_size": ("advanced", "performance", "batch_size"),
    "batch_size_mode": ("advanced", "performance", "batch_size_mode"),
    "manual_batch_size": ("advanced", "performance", "manual_batch_size"),
    "gpu_index": ("advanced", "performance", "gpu_index"),
    "chunk_length_sec": ("advanced", "performance", "chunk_length_sec"),
    "chunk_length_mode": ("advanced", "performance", "chunk_length_mode"),
    "enable_torch_compile": ("advanced", "performance", "enable_torch_compile"),
    "clear_gpu_cache": ("advanced", "performance", "clear_gpu_cache"),
    "asr_compute_device": ("advanced", "performance", "asr_compute_device"),
    "asr_dtype": ("advanced", "performance", "asr_dtype"),
    "asr_ct2_compute_type": ("advanced", "performance", "asr_ct2_compute_type"),
    "asr_ct2_cpu_threads": ("advanced", "performance", "asr_ct2_cpu_threads"),
    "max_parallel_downloads": ("advanced", "performance", "max_parallel_downloads"),
    # Advanced storage
    "record_storage_mode": ("advanced", "storage", "record_storage_mode"),
    "record_storage_limit": ("advanced", "storage", "record_storage_limit"),
    "max_memory_seconds_mode": ("advanced", "storage", "max_memory_seconds_mode"),
    "max_memory_seconds": ("advanced", "storage", "max_memory_seconds"),
    "min_free_ram_mb": ("advanced", "storage", "min_free_ram_mb"),
    "auto_ram_threshold_percent": ("advanced", "storage", "auto_ram_threshold_percent"),
    "save_temp_recordings": ("advanced", "storage", "save_temp_recordings"),
    "storage_root_dir": ("advanced", "storage", "storage_root_dir"),
    "models_storage_dir": ("advanced", "storage", "models_storage_dir"),
    "recordings_dir": ("advanced", "storage", "recordings_dir"),
    "asr_cache_dir": ("advanced", "storage", "asr_cache_dir"),
    "deps_install_dir": ("advanced", "storage", "deps_install_dir"),
    "hf_home_dir": ("advanced", "storage", "hf_home_dir"),
    "transformers_cache_dir": ("advanced", "storage", "transformers_cache_dir"),
    "python_packages_dir": ("advanced", "storage", "python_packages_dir"),
    "vad_models_dir": ("advanced", "storage", "vad_models_dir"),
    "hf_cache_dir": ("advanced", "storage", "hf_cache_dir"),
    # Advanced VAD
    "use_vad": ("advanced", "vad", "use_vad"),
    "vad_threshold": ("advanced", "vad", "vad_threshold"),
    "vad_silence_duration": ("advanced", "vad", "vad_silence_duration"),
    "vad_pre_speech_padding_ms": ("advanced", "vad", "vad_pre_speech_padding_ms"),
    "vad_post_speech_padding_ms": ("advanced", "vad", "vad_post_speech_padding_ms"),
    # Advanced workflow/system
    "display_transcripts_in_terminal": ("advanced", "workflow", "display_transcripts_in_terminal"),
    "launch_at_startup": ("advanced", "system", "launch_at_startup"),
}


PATH_TO_KEY: dict[tuple[str, ...], str] = {
    path: key for key, path in KEY_PATH_OVERRIDES.items()
}
PATH_TO_KEY.update({(field_name,): field_name for field_name in AppConfig.model_fields})


def path_for_key(key: str) -> tuple[str, ...]:
    return KEY_PATH_OVERRIDES.get(key, (key,))


def set_path_value(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = target
    for part in path[:-1]:
        existing = node.get(part)
        if not isinstance(existing, dict):
            existing = {}
            node[part] = existing
        node = existing
    node[path[-1]] = value


def get_path_value(source: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    node: Any = source
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            deep_merge_dict(target[key], value)
        else:
            target[key] = value


def normalize_payload_tree(payload: dict[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "advanced" and isinstance(value, dict):
            advanced_section = tree.setdefault("advanced", {})
            deep_merge_dict(advanced_section, value)
            continue
        path = path_for_key(key)
        set_path_value(tree, path, value)
    return tree


def flatten_config_tree(config_tree: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in config_tree.items():
        if key in {"advanced", "sound"}:
            continue
        flat[key] = copy.deepcopy(value)
    for key, path in KEY_PATH_OVERRIDES.items():
        value = get_path_value(config_tree, path)
        if value is not None:
            flat[key] = copy.deepcopy(value)
    return flat


def coerce_with_defaults(payload: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate ``payload`` merging it with ``defaults``.

    Returns the sanitized configuration dictionary and a list of warning
    messages produced while coercing invalid fields back to their defaults.
    """
    normalized_payload = dict(payload)
    warnings: list[str] = []

    record_mode_value = normalized_payload.get("record_mode")
    if isinstance(record_mode_value, str) and record_mode_value.strip().lower() == "hold":
        warnings.append("Legacy record_mode 'hold' mapped to 'press'.")
        LOGGER.info(
            log_context(
                "Mapping legacy record_mode 'hold' to 'press'.",
                event="config.legacy_record_mode_mapped",
            )
        )
        normalized_payload["record_mode"] = "press"

    defaults_tree = copy.deepcopy(defaults)
    merged_tree = copy.deepcopy(defaults_tree)
    normalized_tree = normalize_payload_tree(normalized_payload)
    deep_merge_dict(merged_tree, normalized_tree)

    while True:
        try:
            validated = AppConfig.model_validate(merged_tree)
        except ValidationError as exc:  # pragma: no cover - rare paths
            for error in exc.errors():
                loc = error.get("loc", ())
                if not loc:
                    continue
                path = tuple(loc)
                default_value = get_path_value(defaults_tree, path)
                if default_value is None and len(path) == 1:
                    default_value = defaults_tree.get(path[0])
                if default_value is None:
                    continue
                set_path_value(merged_tree, path, copy.deepcopy(default_value))
                field_name = PATH_TO_KEY.get(path, ".".join(str(part) for part in path))
                warnings.append(
                    f"Invalid value for '{field_name}': {error.get('msg')}. Using default instead."
                )
            continue
        else:
            data = validated.model_dump()
            return data, warnings
