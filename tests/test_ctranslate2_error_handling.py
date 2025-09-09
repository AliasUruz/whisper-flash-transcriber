import contextlib
import threading
import types
import sys
import numpy as np
import pytest

# Stub torch to avoid heavy dependency
fake_torch = types.SimpleNamespace(
    no_grad=lambda: contextlib.nullcontext(),
    cuda=types.SimpleNamespace(is_available=lambda: False),
)
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

import src.config_manager as _cfg
_cfg.ASR_CACHE_DIR = ""
_cfg.ASR_MODEL_CONFIG_KEY = "asr_model"
_cfg.list_catalog = lambda: []
_cfg.list_installed = lambda path: []

from src.transcription_handler import TranscriptionHandler


class FailingPipe:
    def transcribe(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_ctranslate2_exception_no_unboundlocalerror():
    handler = TranscriptionHandler.__new__(TranscriptionHandler)
    handler.backend_resolved = "ctranslate2"
    handler.pipe = FailingPipe()
    handler.transcription_cancel_event = threading.Event()
    handler.on_model_error_callback = lambda *args, **kwargs: None
    handler.config_manager = types.SimpleNamespace(get=lambda key: False)
    handler.on_segment_transcribed_callback = lambda *args, **kwargs: None
    handler.on_transcription_result_callback = lambda *args, **kwargs: None
    handler.is_state_transcribing_fn = lambda: False
    handler.text_correction_enabled = False
    handler.gemini_prompt = ""

    # Should handle exception without raising UnboundLocalError
    handler._transcription_task(np.zeros(16000), agent_mode=False)
