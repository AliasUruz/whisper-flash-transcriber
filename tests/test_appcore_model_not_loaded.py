import types
import os
import sys
from unittest.mock import MagicMock

# Patch heavy dependencies before importing the core module
fake_pyautogui = types.ModuleType("pyautogui")
fake_pyautogui.hotkey = MagicMock()
fake_pyperclip = types.ModuleType("pyperclip")
fake_pyperclip.copy = MagicMock()
fake_sd = types.SimpleNamespace(PortAudioError=Exception, InputStream=MagicMock())
fake_onnx = types.ModuleType("onnxruntime")
fake_onnx.InferenceSession = MagicMock()
fake_torch = types.ModuleType("torch")
fake_torch.__spec__ = types.SimpleNamespace()
fake_torch.from_numpy = MagicMock(return_value=types.SimpleNamespace())
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
fake_transformers = types.ModuleType("transformers")
fake_transformers.pipeline = MagicMock()
fake_transformers.AutoProcessor = MagicMock()
fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
fake_transformers.__version__ = "0.0"
fake_optimum = types.ModuleType("optimum")
fake_optimum.__version__ = "1.26.1"
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
sys.modules.setdefault("optimum", fake_optimum)
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
            "use_turbo": False,
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

audio_dummy = DummyAudioHandler

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
    monkeypatch.setattr(core_module, "AudioHandler", DummyAudioHandler)
    monkeypatch.setattr(core_module, "GeminiAPI", DummyGeminiAPI)
    monkeypatch.setattr(core_module, "KeyboardHotkeyManager", DummyHotkeyManager)
    monkeypatch.setattr(core_module, "ConfigManager", DummyConfig)
    monkeypatch.setattr(core_module.TranscriptionHandler, "start_model_loading", lambda self: None)
    monkeypatch.setattr(core_module.AppCore, "_cleanup_old_audio_files_on_startup", lambda self: None)
    monkeypatch.setattr(core_module.atexit, "register", lambda *a, **k: None)
    dummy_root = types.SimpleNamespace(after=lambda *a, **k: None)
    app = core_module.AppCore(dummy_root)
    # Simula modelo não carregado
    app.transcription_handler.pipe = None
    app.current_state = core_module.STATE_IDLE
    return app, core_module

def test_start_recording_without_model(monkeypatch):
    app, core = setup_app(monkeypatch)
    app.audio_handler.start_recording = MagicMock()
    mock_log = MagicMock()
    monkeypatch.setattr(app, "_log_status", mock_log)

    app.start_recording()

    mock_log.assert_called_once_with("Cannot record: Model not loaded.", error=True)
    assert not app.audio_handler.start_recording.called
    assert app.current_state == core.STATE_IDLE

def test_start_agent_command_without_model(monkeypatch):
    app, core = setup_app(monkeypatch)
    app.audio_handler.start_recording = MagicMock()
    mock_log = MagicMock()
    monkeypatch.setattr(app, "_log_status", mock_log)

    app.start_agent_command()

    mock_log.assert_called_once_with("Model not loaded.", error=True)
    assert not app.audio_handler.start_recording.called
    assert app.current_state == core.STATE_IDLE
