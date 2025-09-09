import sys
import types
from pathlib import Path

# Stub external modules required by src.core and its dependencies
sys.modules.setdefault("pyautogui", types.ModuleType("pyautogui"))
sys.modules.setdefault("pyperclip", types.ModuleType("pyperclip"))
sys.modules.setdefault("soundfile", types.ModuleType("soundfile"))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
sys.modules.setdefault("numpy", numpy_stub)

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    no_grad=lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc, val, tb: None),
)
sys.modules.setdefault("torch", torch_stub)

# Stub tkinter messagebox
tk_stub = types.ModuleType("tkinter")
tk_stub.messagebox = types.SimpleNamespace(showerror=lambda *a, **k: None, showinfo=lambda *a, **k: None, showwarning=lambda *a, **k: None)
sys.modules.setdefault("tkinter", tk_stub)

sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

# Stub google generative AI library
google_mod = types.ModuleType("google")
genai_stub = types.ModuleType("google.generativeai")
setattr(genai_stub, "configure", lambda **kwargs: None)
class _DummyModel:
    def __init__(self, *a, **k):
        pass
setattr(genai_stub, "GenerativeModel", _DummyModel)

genai_types_stub = types.ModuleType("google.generativeai.types")
genai_types_stub.helper_types = None
genai_types_stub.BrokenResponseError = Exception
genai_types_stub.IncompleteIterationError = Exception

sys.modules.setdefault("google", google_mod)
sys.modules.setdefault("google.generativeai", genai_stub)
sys.modules.setdefault("google.generativeai.types", genai_types_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_manager import (
    ASR_BACKEND_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
)
from src.core import AppCore


import pytest


@pytest.mark.parametrize(
    "kw, cfg_key",
    [
        ("new_asr_backend", ASR_BACKEND_CONFIG_KEY),
        ("new_asr_model_id", ASR_MODEL_ID_CONFIG_KEY),
        ("new_ct2_quantization", ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
    ],
)
def test_start_model_loading_triggered(monkeypatch, tmp_path, kw, cfg_key):

    class DummyConfig:
        def __init__(self):
            self.store = {}
        def get(self, k, default=None):
            return self.store.get(k, default)
        def set(self, k, v):
            self.store[k] = v
        def save_config(self):
            pass

    cfg = DummyConfig()
    cfg.set(cfg_key, "old")

    core = AppCore.__new__(AppCore)
    core.config_manager = cfg
    core._apply_initial_config_to_core_attributes = lambda: None
    core.audio_handler = types.SimpleNamespace(update_config=lambda: None, config_manager=cfg)
    start_called = {"count": 0}
    core.transcription_handler = types.SimpleNamespace(
        update_config=lambda: None,
        start_model_loading=lambda: start_called.__setitem__("count", start_called["count"] + 1),
        config_manager=cfg,
        gemini_client=None,
        openrouter_client=None,
    )
    core.gemini_api = types.SimpleNamespace(reinitialize_client=lambda: None)
    core.register_hotkeys = lambda: None
    core._log_status = lambda *a, **k: None
    core.ahk_running = False
    core.reregister_timer_thread = None
    core.health_check_thread = None
    dummy_event = types.SimpleNamespace(set=lambda: None, clear=lambda: None)
    core.stop_reregister_event = dummy_event
    core.stop_health_check_event = dummy_event

    core.apply_settings_from_external(**{kw: "new"})
    assert start_called["count"] == 1
