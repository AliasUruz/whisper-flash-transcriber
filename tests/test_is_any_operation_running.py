import os
import types
from unittest.mock import MagicMock
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def setup_fake_modules(monkeypatch):
    fake_pyautogui = types.ModuleType("pyautogui")
    fake_pyautogui.hotkey = MagicMock()
    fake_pyperclip = types.ModuleType("pyperclip")
    fake_pyperclip.copy = MagicMock()
    fake_sd = types.SimpleNamespace(PortAudioError=Exception, InputStream=MagicMock())
    fake_sf = types.ModuleType("soundfile")
    fake_sf.write = MagicMock()
    fake_onnx = types.ModuleType("onnxruntime")
    fake_onnx.InferenceSession = MagicMock()
    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = MagicMock(return_value=types.SimpleNamespace())
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = MagicMock()
    fake_transformers.AutoProcessor = MagicMock()
    fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
    fake_keyboard = types.ModuleType("keyboard")
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("generativeai")
    fake_google.generativeai = fake_genai

    monkeypatch.setitem(sys.modules, "pyautogui", fake_pyautogui)
    monkeypatch.setitem(sys.modules, "pyperclip", fake_pyperclip)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setitem(sys.modules, "soundfile", fake_sf)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnx)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "keyboard", fake_keyboard)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)


class DummyAudioHandler:
    def __init__(self, config, on_audio_segment_ready_callback, on_recording_state_change_callback):
        self.config_manager = config
        self.on_audio_segment_ready_callback = on_audio_segment_ready_callback
        self.on_recording_state_change_callback = on_recording_state_change_callback
        self.is_recording = False


class DummyTranscriptionHandler:
    def __init__(self, *a, **k):
        self.correction_in_progress = False
        self.transcription_in_progress = False

    def start_model_loading(self):
        pass

    def is_text_correction_running(self):
        return self.correction_in_progress

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


def test_is_any_operation_running(monkeypatch):
    setup_fake_modules(monkeypatch)
    from src import core as core_module

    monkeypatch.setattr(core_module, "AudioHandler", DummyAudioHandler)
    monkeypatch.setattr(core_module, "TranscriptionHandler", DummyTranscriptionHandler)
    monkeypatch.setattr(core_module, "GeminiAPI", DummyGeminiAPI)
    monkeypatch.setattr(core_module, "KeyboardHotkeyManager", DummyHotkeyManager)
    monkeypatch.setattr(core_module.AppCore, "_cleanup_old_audio_files_on_startup", lambda self: None)

    dummy_root = types.SimpleNamespace(after=lambda *a, **k: None)
    app = core_module.AppCore(dummy_root)
    app.current_state = core_module.STATE_IDLE

    assert not app.is_any_operation_running()

    app.audio_handler.is_recording = True
    assert app.is_any_operation_running()
    app.audio_handler.is_recording = False

    app.current_state = core_module.STATE_TRANSCRIBING
    assert app.is_any_operation_running()
    app.current_state = core_module.STATE_IDLE

    app.transcription_handler.correction_in_progress = True
    assert app.is_any_operation_running()
    app.transcription_handler.correction_in_progress = False

    app.current_state = core_module.STATE_LOADING_MODEL
    assert app.is_any_operation_running()
