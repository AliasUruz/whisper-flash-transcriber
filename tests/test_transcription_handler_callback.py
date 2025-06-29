import importlib
import importlib.machinery
import types
import concurrent.futures
import threading
import time
import numpy as np
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
    USE_FLASH_ATTENTION_2_CONFIG_KEY,
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
            OPENROUTER_API_KEY_CONFIG_KEY: "dummy",
            OPENROUTER_MODEL_CONFIG_KEY: "",
            GEMINI_API_KEY_CONFIG_KEY: "dummy",
            "gemini_agent_model": "",
            "gemini_prompt": "",
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY: 1.0,
            DISPLAY_TRANSCRIPTS_KEY: False,
            SAVE_TEMP_RECORDINGS_CONFIG_KEY: False,
            USE_FLASH_ATTENTION_2_CONFIG_KEY: False,
            TEXT_CORRECTION_TIMEOUT_CONFIG_KEY: 30,
        }

    def get(self, key, default=None):
        return self.data.get(key, default)

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


def test_transcribe_audio_chunk_handles_missing_callback(monkeypatch):
    cfg = DummyConfig()
    results = []
    mock_on_model_error = MagicMock()

    def result_callback(text, original):
        results.append(text)

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=mock_on_model_error,
        on_transcription_result_callback=result_callback,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: True,
    )
    handler.transcription_pipeline = None  # Simulate missing pipeline
    handler.model_loaded_event.set()
    handler.transcription_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )

    monkeypatch.setattr(handler, "_get_dynamic_batch_size", lambda: 1)

    def fake_correction(
        text,
        agent_mode,
        g_prompt,
        o_prompt,
    ):
        return result_callback(text, text)

    monkeypatch.setattr(handler, "_async_text_correction", fake_correction)

    # Garante que o evento esteja acionado para simular modelo previamente carregado
    handler.model_loaded_event.set()

    handler._transcribe_audio_chunk(None, agent_mode=False)

    mock_on_model_error.assert_not_called()  # Callback should not trigger when model was never loaded
    assert not results  # No transcription result should be added


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
    handler.gemini_api = MagicMock(is_valid=True)

    monkeypatch.setattr(handler.gemini_api, "correct_text_async", MagicMock())

    def _openrouter_async(*a, **k):
        return handler.openrouter_api.correct_text(*a, **k)

    monkeypatch.setattr(
        handler.openrouter_api,
        "correct_text_async",
        MagicMock(wraps=_openrouter_async),
    )

    scenarios = [SERVICE_GEMINI, SERVICE_OPENROUTER, SERVICE_NONE]
    for service in scenarios:
        handler.text_correction_service = service
        cfg.data[TEXT_CORRECTION_SERVICE_CONFIG_KEY] = service
        handler.gemini_api.correct_text_async.reset_mock()
        handler.openrouter_api.correct_text.reset_mock()
        handler._async_text_correction("txt", False, "", "")

        if service == SERVICE_OPENROUTER:
            handler.openrouter_api.correct_text_async.assert_called_once_with(
                "txt",
                "",
                cfg.get(OPENROUTER_API_KEY_CONFIG_KEY),
                cfg.get(OPENROUTER_MODEL_CONFIG_KEY),
            )

        if service == SERVICE_GEMINI:
            assert handler.gemini_api.correct_text_async.called
            assert not handler.openrouter_api.correct_text.called
        elif service == SERVICE_OPENROUTER:
            assert handler.openrouter_api.correct_text.called
            assert not handler.gemini_api.correct_text_async.called
        else:
            assert not handler.gemini_api.correct_text_async.called
            assert not handler.openrouter_api.correct_text.called


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
    handler.gemini_api = MagicMock(is_valid=True)

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
        args=("texto", False, "", ""),
        daemon=True,
    )
    thread.start()
    time.sleep(0.01)
    handler.is_state_transcribing_fn = lambda: False
    thread.join()

    assert results == ["corrigido"]

def test_transcribe_audio_segment_waits_for_model(monkeypatch):
    cfg = DummyConfig()
    
    # Initially, model is not loaded
    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=noop,
        is_state_transcribing_fn=lambda: True,
    )
    handler.transcription_pipeline = DummyPipe()
    handler.model_loaded_event.clear() # Ensure it's not set

    # Mock the transcription function to check if it's called
    mock_transcribe_audio_chunk = MagicMock()
    monkeypatch.setattr(handler, "_transcribe_audio_chunk", mock_transcribe_audio_chunk)

    # Start transcription in a separate thread, it should block
    transcription_thread = threading.Thread(
        target=handler.transcribe_audio_segment,
        args=(None, False),
        daemon=True,
    )
    transcription_thread.start()

    # Give it a moment to potentially block
    time.sleep(0.1)
    assert not mock_transcribe_audio_chunk.called # Should not be called yet

    # Now, simulate model loading completion
    handler.model_loaded_event.set()

    # Give it a moment to unblock and call the task
    time.sleep(0.1)
    assert mock_transcribe_audio_chunk.called # Should be called now

    transcription_thread.join(timeout=1) # Clean up the thread


def test_text_correction_timeout(monkeypatch):
    cfg = DummyConfig()
    cfg.data[TEXT_CORRECTION_ENABLED_CONFIG_KEY] = True
    cfg.data[TEXT_CORRECTION_SERVICE_CONFIG_KEY] = SERVICE_GEMINI
    cfg.data[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY] = 0.01
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
    handler.gemini_api = MagicMock(is_valid=True)

    def slow_correction(*_a, **_k):
        time.sleep(0.05)
        return "corrigido"

    monkeypatch.setattr(handler.gemini_api, "correct_text_async", slow_correction)

    handler._async_text_correction("texto", False, "", "")

    assert results == ["texto"]


def test_transcribe_audio_chunk_uses_audio_input(monkeypatch):
    cfg = DummyConfig()

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
    dummy_result = {"text": "ok"}
    pipeline_mock = MagicMock(return_value=dummy_result)
    handler.transcription_pipeline = pipeline_mock
    monkeypatch.setattr(handler, "_async_text_correction", lambda *_: None)

    audio = np.zeros(16000, dtype=float)

    handler._transcribe_audio_chunk(audio, agent_mode=False)

    pipeline_mock.assert_called_once()
    np.testing.assert_array_equal(pipeline_mock.call_args[0][0], audio)


def test_optimization_fallback_callback(monkeypatch):
    cfg = DummyConfig()
    cfg.data[USE_FLASH_ATTENTION_2_CONFIG_KEY] = True
    cfg.data[GPU_INDEX_CONFIG_KEY] = 0

    messages = []

    handler = TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_optimization_fallback_callback=lambda msg: messages.append(msg),
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )

    import src.transcription_handler as th_module

    monkeypatch.setattr(th_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        th_module.torch.cuda,
        "get_device_capability",
        lambda _=0: (8, 0),
        raising=False,
    )
    monkeypatch.setattr(th_module.torch, "float16", 1, raising=False)
    monkeypatch.setattr(th_module.torch, "float32", 2, raising=False)

    class DummyModel:
        def to_bettertransformer(self):
            raise RuntimeError("fail")

    class DummyPipeline:
        def __init__(self):
            self.model = DummyModel()

    monkeypatch.setattr(th_module, "pipeline", lambda *a, **k: DummyPipeline())

    handler._load_model_task()

    assert messages
    assert "otimização 'Turbo'" in messages[0]
