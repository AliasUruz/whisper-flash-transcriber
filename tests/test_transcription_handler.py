import sys
import types
import contextlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_torch = types.SimpleNamespace(
    no_grad=lambda: contextlib.nullcontext(),
    cuda=types.SimpleNamespace(is_available=lambda: False),
)
sys.modules.setdefault("torch", fake_torch)

from src.config_manager import ConfigManager
from src.transcription_handler import TranscriptionHandler


def _make_handler(tmp_path, error_cb):
    cfg = ConfigManager(config_file=str(tmp_path / "config.json"))

    def _noop(*args, **kwargs):
        pass

    return TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=_noop,
        on_model_error_callback=error_cb,
        on_transcription_result_callback=_noop,
        on_agent_result_callback=_noop,
        on_segment_transcribed_callback=_noop,
        is_state_transcribing_fn=lambda: False,
    )


def test_load_model_failure_triggers_callback(monkeypatch, tmp_path):
    errors = []
    handler = _make_handler(tmp_path, errors.append)

    def failing_make_backend(_):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.transcription_handler.make_backend", failing_make_backend)
    handler._load_model_task()
    assert not handler._asr_loaded
    assert any("Falha ao carregar backend" in msg for msg in errors)


def test_transcription_without_model(tmp_path):
    errors = []
    handler = _make_handler(tmp_path, errors.append)
    handler._asr_backend = None
    handler._asr_loaded = False
    handler._transcription_task(np.zeros(16000), agent_mode=False)
    assert any("Backend ASR indisponível" in msg for msg in errors)
