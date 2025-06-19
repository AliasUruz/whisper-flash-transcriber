import importlib.machinery
import types
import concurrent.futures

# Stub simples de torch
fake_torch = types.ModuleType("torch")
fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
fake_torch.__version__ = "0.0"
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

import sys
sys.modules["torch"] = fake_torch

from src.transcription_handler import TranscriptionHandler
from src.config_manager import (
    BATCH_SIZE_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY,
    MANUAL_BATCH_SIZE_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE,
    OPENROUTER_API_KEY_CONFIG_KEY,
    OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
)


class DummyConfig:
    def __init__(self):
        self.data = {
            BATCH_SIZE_CONFIG_KEY: 16,
            BATCH_SIZE_MODE_CONFIG_KEY: "auto",
            MANUAL_BATCH_SIZE_CONFIG_KEY: 8,
            GPU_INDEX_CONFIG_KEY: -1,
            "batch_size_specified": False,
            "gpu_index_specified": False,
            TEXT_CORRECTION_ENABLED_CONFIG_KEY: False,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY: SERVICE_NONE,
            OPENROUTER_API_KEY_CONFIG_KEY: "",
            OPENROUTER_MODEL_CONFIG_KEY: "",
            GEMINI_API_KEY_CONFIG_KEY: "",
            "gemini_agent_model": "",
            "gemini_prompt": "",
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY: 1.0,
            DISPLAY_TRANSCRIPTS_KEY: False,
        }

    def get(self, key):
        return self.data.get(key)


noop = lambda *a, **k: None


class DummyPipe:
    def __call__(self, *a, **k):
        return {"text": "dummy"}


def test_transcription_task_handles_missing_callback(monkeypatch):
    cfg = DummyConfig()
    results = []

    def result_callback(text, original):
        results.append(text)

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=result_callback,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: True,
    )
    handler.pipe = DummyPipe()
    handler.transcription_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    monkeypatch.setattr(handler, "_get_dynamic_batch_size", lambda: 1)
    monkeypatch.setattr(handler, "_async_text_correction", lambda text, service, ev: result_callback(text, text))

    handler._transcription_task(None, agent_mode=False)

    assert results == ["dummy"]
