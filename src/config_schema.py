from pydantic import BaseModel, conint, confloat
from typing import List, Literal

class SoundConfig(BaseModel):
    sound_enabled: bool
    sound_frequency: conint(gt=0)
    sound_duration: confloat(gt=0.0)
    sound_volume: confloat(ge=0.0, le=1.0)

class ASRLastDownloadStatus(BaseModel):
    status: str
    timestamp: str
    model_id: str
    backend: str
    message: str
    details: str

class ASRLastPromptDecision(BaseModel):
    model_id: str
    backend: str
    decision: str
    timestamp: int

class AppConfig(BaseModel):
    record_key: str
    record_mode: Literal["toggle", "press", "hold"]
    auto_paste: bool
    min_record_duration: confloat(ge=0.0)
    sound: SoundConfig
    agent_key: str
    keyboard_library: str
    text_correction_enabled: bool
    text_correction_service: Literal["none", "openrouter", "gemini"]
    openrouter_model: str
    gemini_model: str
    gemini_agent_model: str
    openrouter_timeout: int
    gemini_timeout: int
    ai_provider: str
    openrouter_prompt: str
    prompt_agentico: str
    gemini_prompt: str
    batch_size: int
    batch_size_mode: str
    manual_batch_size: int
    gpu_index: int
    hotkey_stability_service_enabled: bool
    use_vad: bool
    vad_threshold: confloat(ge=0.0, le=1.0)
    vad_silence_duration: confloat(ge=0.0)
    display_transcripts_in_terminal: bool
    gemini_model_options: List[str]
    save_temp_recordings: bool
    record_storage_mode: str
    record_storage_limit: int
    max_memory_seconds_mode: str
    max_memory_seconds: float
    min_free_ram_mb: int
    auto_ram_threshold_percent: int
    min_transcription_duration: float
    chunk_length_sec: float
    chunk_length_mode: str
    enable_torch_compile: bool
    launch_at_startup: bool
    clear_gpu_cache: bool
    asr_model_id: str
    asr_backend: str
    asr_compute_device: str
    asr_dtype: str
    asr_ct2_compute_type: str
    asr_cache_dir: str
    asr_installed_models: List[str]
    asr_curated_catalog_url: str
    asr_last_download_status: ASRLastDownloadStatus
    asr_last_prompt_decision: ASRLastPromptDecision
    agent_auto_paste: bool
    batch_size_specified: bool
    gpu_index_specified: bool
    record_to_memory: bool
