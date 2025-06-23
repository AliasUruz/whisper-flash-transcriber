import importlib.machinery
import types
import concurrent.futures
import threading
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
    SERVICE_GEMINI,
    SERVICE_OPENROUTER,
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


def test_async_text_correction_service_selection(monkeypatch):
    cfg = DummyConfig()
    cfg.data[TEXT_CORRECTION_ENABLED_CONFIG_KEY] = True

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )

    handler.openrouter_client = MagicMock()
    handler.gemini_client = MagicMock(is_valid=True)

    monkeypatch.setattr(handler, "_correct_text_with_gemini", MagicMock())
    monkeypatch.setattr(handler, "_correct_text_with_openrouter", MagicMock())

    scenarios = [SERVICE_GEMINI, SERVICE_OPENROUTER, SERVICE_NONE]
    for service in scenarios:
        handler.text_correction_service = service
        selected = handler._get_text_correction_service()
        handler._correct_text_with_gemini.reset_mock()
        handler._correct_text_with_openrouter.reset_mock()
        handler._async_text_correction("txt", selected, threading.Event())

        if service == SERVICE_GEMINI:
            assert handler._correct_text_with_gemini.called
            assert not handler._correct_text_with_openrouter.called
        elif service == SERVICE_OPENROUTER:
            assert handler._correct_text_with_openrouter.called
            assert not handler._correct_text_with_gemini.called
        else:
            assert not handler._correct_text_with_gemini.called
            assert not handler._correct_text_with_openrouter.called


def test_async_text_correction_is_cancelled(monkeypatch):
    cfg = DummyConfig()
    cfg.data[TEXT_CORRECTION_ENABLED_CONFIG_KEY] = True

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )

    handler.gemini_client = MagicMock(is_valid=True)

    called = []
    def fake_correct(text):
        called.append(True)
        return text

    monkeypatch.setattr(handler, "_correct_text_with_gemini", fake_correct)

    cancel_event = threading.Event()
    cancel_event.set()
    handler._async_text_correction("txt", SERVICE_GEMINI, cancel_event)

    assert not called


def test_get_dynamic_batch_size_for_cpu_and_gpu(monkeypatch):
    cfg = DummyConfig()
    cfg.data[GPU_INDEX_CONFIG_KEY] = 0
    cfg.data[BATCH_SIZE_MODE_CONFIG_KEY] = "manual"
    cfg.data[MANUAL_BATCH_SIZE_CONFIG_KEY] = 8

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )

    monkeypatch.setattr(fake_torch.cuda, "is_available", lambda: True)
    assert handler._get_dynamic_batch_size() == 8

    monkeypatch.setattr(fake_torch.cuda, "is_available", lambda: False)
    assert handler._get_dynamic_batch_size() == 4
