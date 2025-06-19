import importlib.machinery
import types
from types import SimpleNamespace
import os
from unittest.mock import MagicMock

# Stub simples de torch para evitar importacoes pesadas
fake_torch = types.ModuleType("torch")
fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
fake_torch.__version__ = "0.0"
fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

import sys  # noqa: E402
sys.modules["torch"] = fake_torch

fake_transformers = types.ModuleType("transformers")
fake_transformers.pipeline = MagicMock()
fake_transformers.AutoProcessor = MagicMock()
fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
sys.modules["transformers"] = fake_transformers

fake_requests = types.ModuleType("requests")
sys.modules["requests"] = fake_requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


# Funções de callback dummy

def noop(*_a, **_k):
    return None


def test_state_check_callback_attribute():
    cfg = DummyConfig()
    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=noop,
        is_state_transcribing_fn=noop,
    )
    assert hasattr(handler, "state_check_callback")
    assert handler.state_check_callback is handler.is_state_transcribing_fn
