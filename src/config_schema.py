"""Pydantic schemas for validating application configuration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .logging_utils import get_logger, log_context

LOGGER = get_logger(__name__, component='ConfigSchema')


_DEFAULT_STORAGE_ROOT = (Path.home() / ".cache" / "whisper_flash_transcriber").expanduser()
_DEFAULT_MODELS_STORAGE_DIR = str((_DEFAULT_STORAGE_ROOT / "models").expanduser())
_DEFAULT_DEPS_STORAGE_DIR = str((_DEFAULT_STORAGE_ROOT / "deps").expanduser())


class ASRDownloadStatus(BaseModel):
    """Structured status for the last ASR download attempt."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    status: str = "unknown"
    timestamp: str = ""
    model_id: str = ""
    backend: str = ""
    message: str = ""
    details: str = ""


class ASRPromptDecision(BaseModel):
    """Persisted record of the last prompt decision for model downloads."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    model_id: str = ""
    backend: str = ""
    decision: str = ""
    timestamp: int = 0


class AppConfig(BaseModel):
    """Application configuration validated via Pydantic."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    record_key: str = "F3"
    record_mode: str = "toggle"
    auto_paste: bool = True
    auto_paste_modifier: str = "auto"
    agent_auto_paste: bool | None = None
    min_record_duration: float = Field(default=0.5, ge=0.0)
    sound_enabled: bool = True
    sound_frequency: int = Field(default=400, ge=0)
    sound_duration: float = Field(default=0.3, ge=0.0)
    sound_volume: float = Field(default=0.5, ge=0.0)
    agent_key: str = "F4"
    keyboard_library: str = "win32"
    text_correction_enabled: bool = False
    text_correction_service: str = "none"
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-chat-v3-0324:free"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-lite"
    gemini_agent_model: str = "gemini-2.5-flash-lite"
    openrouter_timeout: int = Field(default=30, ge=1)
    gemini_timeout: int = Field(default=120, ge=1)
    ai_provider: str = "gemini"
    openrouter_prompt: str = ""
    prompt_agentico: str = ""
    gemini_prompt: str = ""
    batch_size: int = Field(default=16, ge=1)
    batch_size_mode: str = "auto"
    manual_batch_size: int = Field(default=8, ge=1)
    gpu_index: int = Field(default=0, ge=-1)
    hotkey_stability_service_enabled: bool = True
    use_vad: bool = False
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_silence_duration: float = Field(default=1.0, ge=0.0)
    vad_pre_speech_padding_ms: int = Field(default=150, ge=0)
    vad_post_speech_padding_ms: int = Field(default=300, ge=0)
    display_transcripts_in_terminal: bool = False
    gemini_model_options: list[str] = Field(default_factory=list)
    save_temp_recordings: bool = False
    record_storage_mode: str = "auto"
    record_storage_limit: int = Field(default=0, ge=0)
    max_memory_seconds_mode: str = "manual"
    max_memory_seconds: float = Field(default=30.0, ge=0.0)
    min_free_ram_mb: int = Field(default=1000, ge=0)
    auto_ram_threshold_percent: int = Field(default=10, ge=1, le=50)
    min_transcription_duration: float = Field(default=1.0, ge=0.0)
    chunk_length_sec: float = Field(default=30.0, ge=0.0)
    chunk_length_mode: str = "manual"
    enable_torch_compile: bool = False
    launch_at_startup: bool = False
    clear_gpu_cache: bool = True
    models_storage_dir: str = _DEFAULT_MODELS_STORAGE_DIR
    deps_install_dir: str = _DEFAULT_DEPS_STORAGE_DIR
    hf_home_dir: str = str((Path(_DEFAULT_DEPS_STORAGE_DIR) / "huggingface").expanduser())
    transformers_cache_dir: str = str((Path(_DEFAULT_DEPS_STORAGE_DIR) / "transformers").expanduser())
    storage_root_dir: str = str(_DEFAULT_STORAGE_ROOT)
    recordings_dir: str = str((_DEFAULT_STORAGE_ROOT / "recordings").expanduser())
    asr_model_id: str = "openai/whisper-large-v3-turbo"
    asr_backend: str = "ctranslate2"
    asr_compute_device: str = "auto"
    asr_dtype: str = "float16"
    asr_ct2_compute_type: str = "int8_float16"
    asr_ct2_cpu_threads: int | None = None
    asr_cache_dir: str = str((_DEFAULT_STORAGE_ROOT / "asr").expanduser())
    asr_installed_models: list[str] = Field(default_factory=list)
    asr_curated_catalog: list[str] = Field(default_factory=list)
    asr_curated_catalog_url: str = ""
    asr_last_download_status: ASRDownloadStatus = Field(default_factory=ASRDownloadStatus)
    asr_last_prompt_decision: ASRPromptDecision = Field(default_factory=ASRPromptDecision)

    @staticmethod
    def _normalize_lower(value: str, *, allowed: set[str], field_name: str) -> str:
        lowered = value.lower()
        if lowered not in allowed:
            raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
        return lowered

    @field_validator("record_key", "agent_key", mode="before")
    @classmethod
    def _coerce_key(cls, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return str(value)

    @field_validator("auto_paste_modifier", mode="before")
    @classmethod
    def _normalize_auto_paste_modifier(cls, value: Any) -> str:
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

    @field_validator("record_mode", mode="before")
    @classmethod
    def _validate_record_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return cls._normalize_lower(value, allowed={"toggle", "press"}, field_name="record_mode")
        raise ValueError("record_mode must be a string")

    @field_validator("batch_size_mode", mode="before")
    @classmethod
    def _validate_batch_size_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return cls._normalize_lower(value, allowed={"auto", "manual"}, field_name="batch_size_mode")
        raise ValueError("batch_size_mode must be a string")

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
            return cls._normalize_lower(
                lowered,
                allowed={"disk", "memory", "auto"},
                field_name="record_storage_mode",
            )
        raise ValueError("record_storage_mode must be a string")

    @field_validator("max_memory_seconds_mode", mode="before")
    @classmethod
    def _validate_memory_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return cls._normalize_lower(
                value,
                allowed={"manual", "auto"},
                field_name="max_memory_seconds_mode",
            )
        raise ValueError("max_memory_seconds_mode must be a string")

    @field_validator("chunk_length_mode", mode="before")
    @classmethod
    def _validate_chunk_mode(cls, value: Any) -> str:
        if isinstance(value, str):
            return cls._normalize_lower(
                value,
                allowed={"manual", "auto"},
                field_name="chunk_length_mode",
            )
        raise ValueError("chunk_length_mode must be a string")

    @field_validator("text_correction_service", mode="before")
    @classmethod
    def _validate_text_service(cls, value: Any) -> str:
        if isinstance(value, str):
            return cls._normalize_lower(
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

    @field_validator(
        "storage_root_dir",
        "models_storage_dir",
        "deps_install_dir",
        "hf_home_dir",
        "transformers_cache_dir",
        "recordings_dir",
        "asr_cache_dir",
        mode="before",
    )
    @classmethod
    def _expand_cache_dir(cls, value: Any) -> str:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return ""
            return str(Path(normalized).expanduser())
        if isinstance(value, Path):
            return str(value.expanduser())
        raise ValueError("Storage paths must be provided as strings or Path objects")

    @field_validator("asr_installed_models", "asr_curated_catalog", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if item is not None]
        return [str(value)]

    @field_validator("agent_auto_paste", mode="before")
    @classmethod
    def _optional_bool(cls, value: Any) -> bool | None:
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
        raise ValueError("agent_auto_paste must be boolean or None")


def coerce_with_defaults(payload: dict[str, Any], defaults: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate ``payload`` merging it with ``defaults``.

    Returns the sanitized configuration dictionary and a list of warning
    messages produced while coercing invalid fields back to their defaults.
    """

    merged = {**defaults, **payload}
    warnings: list[str] = []

    while True:
        try:
            validated = AppConfig.model_validate(merged)
        except ValidationError as exc:  # pragma: no cover - rare paths
            for error in exc.errors():
                loc = error.get("loc", ())
                if not loc:
                    continue
                field_name = loc[0]
                warnings.append(
                    f"Invalid value for '{field_name}': {error.get('msg')}. Using default instead."
                )
                merged[field_name] = defaults.get(field_name)
            continue
        else:
            data = validated.model_dump()
            return data, warnings
