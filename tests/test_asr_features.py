import types
import importlib.util
import pathlib
import sys
import json
import hmac
import hashlib
import requests

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
sys.modules.setdefault("pyperclip", types.ModuleType("pyperclip"))
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))
torch_stub = types.ModuleType("torch")
torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", torch_stub)
transformers_stub = types.ModuleType("transformers")
transformers_stub.pipeline = lambda *a, **k: None
class _DummyAuto:
    @staticmethod
    def from_pretrained(*a, **k):
        return object()
transformers_stub.AutoProcessor = _DummyAuto
transformers_stub.AutoModelForSpeechSeq2Seq = _DummyAuto
sys.modules.setdefault("transformers", transformers_stub)
sys.modules.setdefault("keyboard", types.ModuleType("keyboard"))
google_stub = types.ModuleType("google")
generativeai_stub = types.ModuleType("google.generativeai")
generativeai_stub.types = types.ModuleType("google.generativeai.types")
generativeai_stub.types.helper_types = types.SimpleNamespace()
generativeai_stub.types.BrokenResponseError = Exception
generativeai_stub.types.IncompleteIterationError = Exception
google_stub.generativeai = generativeai_stub
sys.modules.setdefault("google", google_stub)
sys.modules.setdefault("google.generativeai", generativeai_stub)
sys.modules.setdefault("google.generativeai.types", generativeai_stub.types)

from src.config_manager import (
    ConfigManager,
    EMBEDDED_ASR_CATALOG,
    CATALOG_SIGNATURE_KEY,
)
from src.core import AppCore
from src.transcription_handler import TranscriptionHandler


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_curated_catalog_update(monkeypatch):
    models = [{"model_id": "m1"}, {"model_id": "m2"}]
    sig = hmac.new(
        CATALOG_SIGNATURE_KEY,
        json.dumps(models, sort_keys=True).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    data = {"models": models, "signature": sig}
    monkeypatch.setattr(
        "requests.get", lambda url, timeout=10: DummyResponse(data)
    )
    cm = ConfigManager()
    assert cm.get_asr_curated_catalog() == models


def test_resolve_backend_ct2(monkeypatch):
    monkeypatch.setattr("requests.get", lambda url, timeout=10: DummyResponse([]))
    cm = ConfigManager()
    th = TranscriptionHandler(
        config_manager=cm,
        gemini_api_client=None,
        on_model_ready_callback=lambda: None,
        on_model_error_callback=lambda e: None,
        on_transcription_result_callback=lambda r: None,
        on_agent_result_callback=lambda r: None,
        on_segment_transcribed_callback=lambda s: None,
        is_state_transcribing_fn=lambda: False,
    )

    th.asr_backend = "auto"

    def fake_find_spec(name):
        if name == "faster_whisper":
            return object()
        return importlib.util.find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    backend, *_ = th._resolve_asr_settings()
    assert backend == "ctranslate2"


def test_apply_settings_updates_trans_handler(monkeypatch):
    data = [{"model_id": "m1"}]
    monkeypatch.setattr(
        "requests.get", lambda url, timeout=10: DummyResponse(data)
    )

    class DummyAudioHandler:
        def __init__(self, *_, **__):
            self.is_recording = False

        def update_config(self):
            pass

        def shutdown(self):
            pass

    class DummyTransHandler:
        def __init__(self, config_manager, **kwargs):
            self.config_manager = config_manager
            self.gemini_client = None
            self.openrouter_client = None

        def start_model_loading(self):
            pass

        def update_config(self):
            pass

        def shutdown(self):
            pass

    class DummyKeyboardHotkeyManager:
        def __init__(self, *_, **__):
            pass

        def update_config(self, **kwargs):
            pass

    class DummyGeminiAPI:
        def __init__(self, *_):
            pass

        def reinitialize_client(self):
            pass

    monkeypatch.setattr("src.core.AudioHandler", DummyAudioHandler)
    monkeypatch.setattr("src.core.TranscriptionHandler", DummyTransHandler)
    monkeypatch.setattr("src.core.KeyboardHotkeyManager", DummyKeyboardHotkeyManager)
    monkeypatch.setattr("src.core.GeminiAPI", DummyGeminiAPI)

    updated = {}
    def mark_update(self):
        updated["called"] = True
    monkeypatch.setattr(DummyTransHandler, "update_config", mark_update)

    app = AppCore(None)
    updated.clear()
    app.apply_settings_from_external(new_asr_model_id="model-x")
    app.transcription_handler.update_config()
    assert updated.get("called")


def test_scan_cache_and_remove(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda url, timeout=10: DummyResponse([]))
    cache_dir = tmp_path / "cache"
    (cache_dir / "m1").mkdir(parents=True)
    (cache_dir / "m2").mkdir()
    cm = ConfigManager()
    cm.set_asr_cache_dir(str(cache_dir))
    cm.scan_asr_cache_dir()
    assert set(cm.get_asr_installed_models()) == {"m1", "m2"}
    cm.remove_installed_model("m1")
    assert cm.get_asr_installed_models() == ["m2"]


def test_ct2_advanced_params(monkeypatch):
    class DummyModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, language=None, beam_size=0, temperature=0.0):
            self.beam_size = beam_size
            self.temperature = temperature
            return [], None

    monkeypatch.setattr("requests.get", lambda url, timeout=10: DummyResponse([]))
    cm = ConfigManager()
    th = TranscriptionHandler(
        config_manager=cm,
        gemini_api_client=None,
        on_model_ready_callback=lambda: None,
        on_model_error_callback=lambda e: None,
        on_transcription_result_callback=lambda r: None,
        on_agent_result_callback=lambda r: None,
        on_segment_transcribed_callback=lambda s: None,
        is_state_transcribing_fn=lambda: False,
    )
    th.backend_resolved = "ctranslate2"
    th.pipe = DummyModel()
    th.asr_ct2_beam_size = 7
    th.asr_ct2_temperature = 0.5
    th._transcription_task("dummy", False)
    assert th.pipe.beam_size == 7
    assert th.pipe.temperature == 0.5


def test_catalog_fallback(monkeypatch):
    models = [{"model_id": "m1"}]
    sig = hmac.new(
        CATALOG_SIGNATURE_KEY,
        json.dumps(models, sort_keys=True).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    monkeypatch.setattr(
        "requests.get", lambda url, timeout=10: DummyResponse({"models": models, "signature": sig})
    )
    cm = ConfigManager()
    assert cm.get_asr_curated_catalog() == models

    def _fail(*a, **k):
        raise requests.RequestException("fail")

    monkeypatch.setattr("requests.get", _fail)
    cm2 = ConfigManager()
    assert cm2.get_asr_curated_catalog() == EMBEDDED_ASR_CATALOG

