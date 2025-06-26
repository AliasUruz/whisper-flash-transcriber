import types
import time
import threading
import os
import sys
import numpy as np
from unittest.mock import MagicMock

# Stub external dependencies before importing core module
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
fake_keyboard.unhook_all = lambda *a, **k: None
fake_google = types.ModuleType("google")
fake_genai = types.ModuleType("generativeai")
fake_genai.configure = lambda api_key=None: None
fake_genai.GenerativeModel = MagicMock()
fake_types = types.ModuleType("types")
fake_helper = types.ModuleType("helper_types")
fake_helper.RequestOptions = MagicMock()
fake_types.helper_types = fake_helper
fake_types.BrokenResponseError = type("BrokenResponseError", (Exception,), {})
fake_types.IncompleteIterationError = type("IncompleteIterationError", (Exception,), {})
fake_genai.types = fake_types
fake_google.generativeai = fake_genai

sys.modules.setdefault("pyautogui", fake_pyautogui)
sys.modules.setdefault("pyperclip", fake_pyperclip)
sys.modules.setdefault("sounddevice", fake_sd)
sys.modules.setdefault("onnxruntime", fake_onnx)
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("transformers", fake_transformers)
sys.modules.setdefault("keyboard", fake_keyboard)
sys.modules.setdefault("google", fake_google)
sys.modules.setdefault("google.generativeai", fake_genai)
sys.modules.setdefault("google.generativeai.types", fake_types)
sys.modules.setdefault("google.generativeai.types.helper_types", fake_helper)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src import core as core_module

from src.config_manager import (
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE,
)

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
            "min_transcription_duration": 0.0,
            TEXT_CORRECTION_ENABLED_CONFIG_KEY: True,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY: SERVICE_NONE,
        }

    def get(self, key, default=None):
        return self.data.get(key, default)

class DummyAudioHandler:
    def __init__(self, config, on_audio_segment_ready_callback, on_recording_state_change_callback):
        self.config_manager = config
        self.on_audio_segment_ready_callback = on_audio_segment_ready_callback
        self.on_recording_state_change_callback = on_recording_state_change_callback
        self.is_recording = False

    def start_recording(self):
        self.is_recording = True
        self.on_recording_state_change_callback(core_module.STATE_RECORDING)

    def stop_recording(self):
        self.is_recording = False
        self.on_recording_state_change_callback(core_module.STATE_TRANSCRIBING)
        audio = np.zeros(1600, dtype=np.float32)
        self.on_audio_segment_ready_callback(audio)

class DummyTranscriptionHandler:
    def __init__(self, config_manager, gemini_api_client, on_model_ready_callback,
                 on_model_error_callback, on_transcription_result_callback,
                 on_agent_result_callback, on_segment_transcribed_callback,
                 is_state_transcribing_fn):
        self.pipe = True
        self.on_transcription_result_callback = on_transcription_result_callback
        self.config_manager = config_manager

    def start_model_loading(self):
        pass

    def transcribe_audio_segment(self, audio_data, agent_mode=False):
        if self.config_manager.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY):
            def _run():
                time.sleep(0.01)
                self.on_transcription_result_callback("fixed", "raw")
            threading.Thread(target=_run).start()
        else:
            self.on_transcription_result_callback("raw", "raw")

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
    fake_keyboard.unhook_all = lambda *a, **k: None
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("generativeai")
    fake_genai.configure = lambda api_key=None: None
    fake_genai.GenerativeModel = MagicMock()
    fake_types = types.ModuleType("types")
    fake_helper = types.ModuleType("helper_types")
    fake_helper.RequestOptions = MagicMock()
    fake_types.helper_types = fake_helper
    fake_types.BrokenResponseError = type("BrokenResponseError", (Exception,), {})
    fake_types.IncompleteIterationError = type("IncompleteIterationError", (Exception,), {})
    fake_genai.types = fake_types
    fake_google.generativeai = fake_genai

    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)
    monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "keyboard", fake_keyboard)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.generativeai.types", fake_types)
    monkeypatch.setitem(
        sys.modules,
        "google.generativeai.types.helper_types",
        fake_helper,
    )

    monkeypatch.setattr(core_module, "AudioHandler", DummyAudioHandler)
    monkeypatch.setattr(core_module, "TranscriptionHandler", DummyTranscriptionHandler)
    monkeypatch.setattr(core_module, "GeminiAPI", DummyGeminiAPI)
    monkeypatch.setattr(core_module, "KeyboardHotkeyManager", DummyHotkeyManager)
    monkeypatch.setattr(core_module, "ConfigManager", DummyConfig)
    monkeypatch.setattr(core_module.atexit, "register", lambda *a, **k: None)

    dummy_root = types.SimpleNamespace(after=lambda *a, **k: None)
    return core_module.AppCore(dummy_root)

def test_no_recording_when_transcribing(monkeypatch):
    app = setup_app(monkeypatch)
    app.current_state = core_module.STATE_TRANSCRIBING
    app.audio_handler.start_recording = MagicMock()

    app.start_recording()
    assert not app.audio_handler.start_recording.called

def test_state_idle_after_text_correction(monkeypatch):
    app = setup_app(monkeypatch)
    app.current_state = core_module.STATE_IDLE

    app.start_recording()
    app.stop_recording()

    assert app.current_state == core_module.STATE_TRANSCRIBING

    time.sleep(0.05)
    assert app.current_state == core_module.STATE_IDLE
