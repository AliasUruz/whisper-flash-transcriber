import importlib
import importlib.machinery
import types
import concurrent.futures
import numpy as np
import threading
import time
import sys
from unittest.mock import MagicMock

# Stub simples de torch
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

if "src.transcription_handler" in sys.modules:
    importlib.reload(sys.modules["src.transcription_handler"])

from src.transcription_handler import TranscriptionHandler  # noqa: E402
from src.config_manager import (  # noqa: E402
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
            OPENROUTER_API_KEY_CONFIG_KEY: "dummy",
            OPENROUTER_MODEL_CONFIG_KEY: "",
            GEMINI_API_KEY_CONFIG_KEY: "dummy",
            "gemini_agent_model": "",
            "gemini_prompt": "",
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY: 1.0,
            DISPLAY_TRANSCRIPTS_KEY: False,
            SAVE_TEMP_RECORDINGS_CONFIG_KEY: False,
        }

    def get(self, key):
        return self.data.get(key)

    def get_api_key(self, provider):
        if provider == SERVICE_GEMINI:
            return self.data.get(GEMINI_API_KEY_CONFIG_KEY)
        if provider == SERVICE_OPENROUTER:
            return self.data.get(OPENROUTER_API_KEY_CONFIG_KEY)
        return ""


def noop(*_a, **_k):
    return None


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
    handler.transcription_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )

    monkeypatch.setattr(handler, "_get_dynamic_batch_size", lambda: 1)

    def fake_correction(
        text,
        agent_mode,
        g_prompt,
        o_prompt,
        was_transcribing,
    ):
        return result_callback(text, text)

    monkeypatch.setattr(handler, "_async_text_correction", fake_correction)

    handler._transcription_task("dummy.wav", agent_mode=False)

    assert results == ["dummy"]


def test_transcription_task_accepts_numpy_array(monkeypatch):
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

    audio_array = np.zeros(16000, dtype=np.float32)
    handler._transcription_task(audio_array, agent_mode=False)

    assert results == ["dummy"]


def test_transcription_task_accepts_file_path(monkeypatch):
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

    handler._transcription_task("audio.wav", agent_mode=False)

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
    handler.openrouter_api = handler.openrouter_client
    handler.gemini_client = MagicMock(is_valid=True)
    handler.gemini_api = handler.gemini_client
    handler.gemini_api = handler.gemini_client

    monkeypatch.setattr(handler.gemini_api, "correct_text_async", MagicMock())
    monkeypatch.setattr(
        handler.openrouter_api,
        "correct_text_async",
        MagicMock(),
    )

    scenarios = [SERVICE_GEMINI, SERVICE_OPENROUTER, SERVICE_NONE]
    for service in scenarios:
        handler.text_correction_service = service
        cfg.data[TEXT_CORRECTION_SERVICE_CONFIG_KEY] = service
        handler.gemini_api.correct_text_async.reset_mock()
        handler.openrouter_api.correct_text_async.reset_mock()
        handler._async_text_correction("txt", False, "", "", True)

        if service == SERVICE_GEMINI:
            assert handler.gemini_api.correct_text_async.called
            assert not handler.openrouter_api.correct_text_async.called
        elif service == SERVICE_OPENROUTER:
            assert handler.openrouter_api.correct_text_async.called
            assert not handler.gemini_api.correct_text_async.called
        else:
            assert not handler.gemini_api.correct_text_async.called
            assert not handler.openrouter_api.correct_text_async.called


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

    import src.transcription_handler as th_module
    monkeypatch.setattr(th_module.torch.cuda, "is_available", lambda: True)
    assert handler._get_dynamic_batch_size() == 8
    monkeypatch.setattr(th_module.torch.cuda, "is_available", lambda: False)
    assert handler._get_dynamic_batch_size() == 4


def test_text_correction_preserves_result_when_state_changes(monkeypatch):
    cfg = DummyConfig()
    cfg.data[TEXT_CORRECTION_ENABLED_CONFIG_KEY] = True
    cfg.data[TEXT_CORRECTION_SERVICE_CONFIG_KEY] = SERVICE_GEMINI
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
    handler.gemini_client = MagicMock(is_valid=True)
    handler.gemini_api = handler.gemini_client

    def delayed_correct(text):
        time.sleep(0.05)
        return "corrigido"

    monkeypatch.setattr(
        handler.gemini_api,
        "correct_text_async",
        lambda *a, **k: delayed_correct(a[0]),
    )

    thread = threading.Thread(
        target=handler._async_text_correction,
        args=("texto", False, "", "", True),
        daemon=True,
    )
    thread.start()
    time.sleep(0.01)
    handler.is_state_transcribing_fn = lambda: False
    thread.join()

    assert results == ["corrigido"]
