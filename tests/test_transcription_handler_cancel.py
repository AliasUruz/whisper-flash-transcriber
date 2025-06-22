import importlib.machinery
import types
import time
import numpy as np
import concurrent.futures
from unittest.mock import MagicMock

# Stub simples de torch
fake_torch = types.ModuleType("torch")
fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
fake_torch.__version__ = "0.0"
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

import sys
sys.modules["torch"] = fake_torch

fake_transformers = types.ModuleType("transformers")
fake_transformers.pipeline = MagicMock()
fake_transformers.AutoProcessor = MagicMock()
fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
sys.modules["transformers"] = fake_transformers

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
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
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
            SAVE_TEMP_RECORDINGS_CONFIG_KEY: False,
        }

    def get(self, key):
        return self.data.get(key)


noop = lambda *a, **k: None


class SlowPipe:
    def __call__(self, *a, **k):
        time.sleep(0.1)
        return {"text": "dummy"}


def test_callback_called_on_cancel(monkeypatch):
    cfg = DummyConfig()
    called = []

    def cancelled():
        called.append(True)

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: True,
        on_transcription_cancelled_callback=cancelled,
    )
    handler.pipe = SlowPipe()
    handler.transcription_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    monkeypatch.setattr(handler, "_get_dynamic_batch_size", lambda: 1)

    handler.transcribe_audio_segment(np.zeros((1, 1), dtype=np.float32), agent_mode=False)
    time.sleep(0.02)
    handler.cancel_transcription()
    handler.transcription_future.result(timeout=1)

    assert called == [True]
