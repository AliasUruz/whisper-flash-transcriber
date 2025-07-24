import os
import sys
import types
import time
import numpy as np
from unittest.mock import MagicMock

# Stub de dependências antes de importar o módulo principal
fake_pyautogui = types.ModuleType("pyautogui")
fake_pyautogui.hotkey = MagicMock()
fake_pyperclip = types.ModuleType("pyperclip")
fake_pyperclip.copy = MagicMock()
fake_sd = types.SimpleNamespace(PortAudioError=Exception, InputStream=MagicMock())
fake_onnx = types.ModuleType("onnxruntime")
fake_onnx.InferenceSession = MagicMock()
fake_torch = types.ModuleType("torch")
fake_torch.from_numpy = MagicMock(return_value=types.SimpleNamespace())
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
fake_transformers = types.ModuleType("transformers")
fake_transformers.pipeline = MagicMock()
fake_transformers.AutoProcessor = MagicMock()
fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
fake_keyboard = types.ModuleType("keyboard")

sys.modules.setdefault("pyautogui", fake_pyautogui)
sys.modules.setdefault("pyperclip", fake_pyperclip)
sys.modules.setdefault("sounddevice", fake_sd)
sys.modules.setdefault("onnxruntime", fake_onnx)
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("transformers", fake_transformers)
sys.modules.setdefault("keyboard", fake_keyboard)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import src.core as core_module
from src.audio_handler import AUDIO_SAMPLE_RATE
from src.config_manager import (
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE,
)

# Dummy classes based on test_appcore_state
class DummyConfig:
    def __init__(self):
        self.data = {
            "record_key": "F3",
            "record_mode": "toggle",
            "auto_paste": False,
            "agent_key": "F4",
            "hotkey_stability_service_enabled": False,
            "keyboard_library": "win32",
            "min_record_duration": 0.0,
            "min_transcription_duration": 1.0,
            TEXT_CORRECTION_ENABLED_CONFIG_KEY: False,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY: SERVICE_NONE,
        }

    def get(self, key, default=None):
        return self.data.get(key, default)

class DummyAudioHandler:
    def __init__(self, config_manager, on_audio_segment_ready_callback, on_recording_state_change_callback):
        self.config_manager = config_manager
        self.on_audio_segment_ready_callback = on_audio_segment_ready_callback
        self.on_recording_state_change_callback = on_recording_state_change_callback
        self.is_recording = False

    def start_recording(self):
        self.is_recording = True
        self.on_recording_state_change_callback(core_module.STATE_RECORDING)

    def stop_recording(self):
        self.is_recording = False
        self.on_recording_state_change_callback(core_module.STATE_TRANSCRIBING)
        audio = np.zeros(int(0.5 * AUDIO_SAMPLE_RATE), dtype=np.float32)
        self.on_audio_segment_ready_callback(audio)
        return True

class DummyTranscriptionHandler:
    def __init__(self, config_manager, gemini_api_client, on_model_ready_callback,
                 on_model_error_callback, on_transcription_result_callback,
                 on_agent_result_callback, on_segment_transcribed_callback,
                 is_state_transcribing_fn):
        self.pipe = True
        self.on_transcription_result_callback = on_transcription_result_callback

    def start_model_loading(self):
        pass

    def transcribe_audio_segment(self, audio_source, agent_mode=False):
        self.on_transcription_result_callback("done", "raw")

    def stop_transcription(self):
        pass

    def shutdown(self):
        pass

class DummyGeminiAPI:
    def __init__(self, *a, **k):
        self.is_valid = True

class DummyHotkeyManager:
    def __init__(self, *a, **k):
        pass
    def start(self):
        return True
    def stop(self):
        pass
    def update_config(self, *a, **k):
        pass
    def set_callbacks(self, *a, **k):
        pass
    def detect_single_key(self):
        return None


def setup_app(monkeypatch):
    fake_pyautogui = types.ModuleType("pyautogui")
    fake_pyautogui.hotkey = MagicMock()
    fake_pyperclip = types.ModuleType("pyperclip")
    fake_pyperclip.copy = MagicMock()
    fake_sd = types.SimpleNamespace(PortAudioError=Exception, InputStream=MagicMock())
    fake_onnx = types.ModuleType("onnxruntime")
    fake_onnx.InferenceSession = MagicMock()
    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = MagicMock(return_value=types.SimpleNamespace())
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = MagicMock()
    fake_transformers.AutoProcessor = MagicMock()
    fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
    fake_keyboard = types.ModuleType("keyboard")

    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)
    monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "keyboard", fake_keyboard)
    monkeypatch.setattr(core_module, "AudioHandler", DummyAudioHandler)
    monkeypatch.setattr(core_module, "TranscriptionHandler", DummyTranscriptionHandler)
    monkeypatch.setattr(core_module, "GeminiAPI", DummyGeminiAPI)
    monkeypatch.setattr(core_module, "KeyboardHotkeyManager", DummyHotkeyManager)
    monkeypatch.setattr(core_module, "ConfigManager", DummyConfig)
    monkeypatch.setattr(core_module.atexit, "register", lambda *a, **k: None)
    dummy_root = types.SimpleNamespace(after=lambda *a, **k: None)
    app = core_module.AppCore(dummy_root)
    app.current_state = core_module.STATE_IDLE
    if hasattr(app.transcription_handler, "pipe"):
        app.transcription_handler.pipe = True
    return app


def test_state_idle_when_segment_below_min_duration(monkeypatch):
    app = setup_app(monkeypatch)
    app.transcription_handler.transcribe_audio_segment = MagicMock()

    app.start_recording()
    app.stop_recording()

    time.sleep(0.01)
    assert app.current_state == core_module.STATE_IDLE
    app.transcription_handler.transcribe_audio_segment.assert_not_called()


def test_state_idle_when_recording_invalid(monkeypatch):
    app = setup_app(monkeypatch)
    app.transcription_handler.transcribe_audio_segment = MagicMock()

    def stop_invalid(self):
        self.is_recording = False
        self.on_recording_state_change_callback(core_module.STATE_IDLE)
        return False

    app.audio_handler.stop_recording = types.MethodType(stop_invalid, app.audio_handler)

    app.start_recording()
    app.stop_recording()

    assert app.current_state == core_module.STATE_IDLE
    app.transcription_handler.transcribe_audio_segment.assert_not_called()
