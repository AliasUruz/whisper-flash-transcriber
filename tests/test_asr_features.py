import types
import importlib.util
import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
sys.modules.setdefault("pyperclip", types.ModuleType("pyperclip"))
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
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

from src.config_manager import ConfigManager
from src.core import AppCore
from src.transcription_handler import TranscriptionHandler


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data
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

