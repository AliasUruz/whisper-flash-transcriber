import importlib.machinery
import types
from unittest.mock import MagicMock
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

# Stub torch and transformers
fake_torch = types.ModuleType("torch")
fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
fake_torch.__version__ = "0.0"
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = fake_torch

fake_transformers = types.ModuleType("transformers")
fake_transformers.pipeline = MagicMock()
fake_transformers.AutoProcessor = MagicMock()
fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
sys.modules["transformers"] = fake_transformers

from src.transcription_handler import TranscriptionHandler  # noqa: E402
from src.config_manager import (  # noqa: E402
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
    TEXT_CORRECTION_TIMEOUT_CONFIG_KEY,
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
            "use_turbo": False,
            OPENROUTER_API_KEY_CONFIG_KEY: "",
            OPENROUTER_MODEL_CONFIG_KEY: "",
            GEMINI_API_KEY_CONFIG_KEY: "",
            "gemini_agent_model": "",
            "gemini_prompt": "",
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY: 1.0,
            DISPLAY_TRANSCRIPTS_KEY: False,
            SAVE_TEMP_RECORDINGS_CONFIG_KEY: False,
            TEXT_CORRECTION_TIMEOUT_CONFIG_KEY: 30,
        }

    def get(self, key, default=None):
        return self.data.get(key, default)


def test_executor_shutdown_parameters():
    cfg = DummyConfig()
    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=lambda: None,
        on_model_error_callback=lambda *_: None,
        on_transcription_result_callback=lambda *_: None,
        on_agent_result_callback=lambda *_: None,
        on_segment_transcribed_callback=lambda *_: None,
        is_state_transcribing_fn=lambda: False,
    )
    dummy_exec = MagicMock()
    handler.transcription_executor = dummy_exec

    handler.shutdown()

    dummy_exec.shutdown.assert_called_once_with(
        wait=False,
        cancel_futures=True,
    )
    assert handler.transcription_cancel_event.is_set()


def test_correction_thread_join_called_when_alive():
    cfg = DummyConfig()
    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=lambda: None,
        on_model_error_callback=lambda *_: None,
        on_transcription_result_callback=lambda *_: None,
        on_agent_result_callback=lambda *_: None,
        on_segment_transcribed_callback=lambda *_: None,
        is_state_transcribing_fn=lambda: False,
    )

    dummy_exec = MagicMock()
    handler.transcription_executor = dummy_exec

    dummy_thread = MagicMock()
    dummy_thread.is_alive.return_value = True
    handler.correction_thread = dummy_thread

    handler.shutdown()

    dummy_thread.join.assert_called_once()
