import os
import types
from unittest.mock import MagicMock
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_temp_recording_cleanup(tmp_path, monkeypatch):
    # Stubs para dependências ausentes
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

    from src import core as core_module

    class DummyAudioHandler:
        def __init__(self, config, on_audio_segment_ready_callback, on_recording_state_change_callback):
            self.config_manager = config
            self.on_audio_segment_ready_callback = on_audio_segment_ready_callback
            self.on_recording_state_change_callback = on_recording_state_change_callback
            self.is_recording = False
            self.temp_file_path = None

        def start_recording(self):
            self.is_recording = True

        def stop_recording(self):
            self.is_recording = False
            path = tmp_path / "temp_recording_test.wav"
            path.write_text("data")
            self.temp_file_path = str(path)
            audio = np.zeros(1600, dtype=np.float32)
            self.on_audio_segment_ready_callback(audio)

    class DummyTranscriptionHandler:
        def __init__(self, *a, **k):
            self.pipe = True
            self.transcription_future = None

        def start_model_loading(self):
            pass

        def is_transcription_running(self):
            return False

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

    monkeypatch.setattr(core_module, "AudioHandler", DummyAudioHandler)
    monkeypatch.setattr(core_module, "TranscriptionHandler", DummyTranscriptionHandler)
    monkeypatch.setattr(core_module, "GeminiAPI", DummyGeminiAPI)
    monkeypatch.setattr(core_module, "KeyboardHotkeyManager", DummyHotkeyManager)

    dummy_root = types.SimpleNamespace(after=lambda *a, **k: None)
    app = core_module.AppCore(dummy_root)
    app.current_state = core_module.STATE_IDLE

    app.start_recording()
    app.stop_recording()

    file_path = app.audio_handler.temp_file_path
    assert file_path and os.path.exists(file_path)

    app._handle_transcription_result("final", "raw")

    assert not os.path.exists(file_path)
    assert app.audio_handler.temp_file_path is None
