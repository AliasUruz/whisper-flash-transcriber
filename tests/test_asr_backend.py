import types
import sys

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    float16=object(),
    float32=object(),
)
sys.modules["torch"] = torch_stub
numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
numpy_stub.zeros = lambda n, *a, **k: [0] * n
sys.modules["numpy"] = numpy_stub
sys.modules["soundfile"] = types.ModuleType("soundfile")
sys.modules["requests"] = types.ModuleType("requests")
sys.modules["sounddevice"] = types.ModuleType("sounddevice")
sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")
sys.modules["psutil"] = types.ModuleType("psutil")
transformers_stub = types.ModuleType("transformers")
transformers_stub.pipeline = lambda *args, **kwargs: None
transformers_stub.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
transformers_stub.AutoModelForSpeechSeq2Seq = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
transformers_stub.BitsAndBytesConfig = object
sys.modules["transformers"] = transformers_stub
distutils_util = types.ModuleType("distutils.util")
def _strtobool(val):
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    if val in ("n", "no", "f", "false", "off", "0"):
        return 0
    raise ValueError(f"invalid truth value {val}")
distutils_util.strtobool = _strtobool
sys.modules["distutils"] = types.ModuleType("distutils")
sys.modules["distutils.util"] = distutils_util
import torch
from src.config_manager import ConfigManager, DEFAULT_CONFIG, CLEAR_GPU_CACHE_CONFIG_KEY
from src.transcription_handler import TranscriptionHandler
from src.asr_backends import backend_registry, _AdapterBackend


def _build_handler(monkeypatch, clear_gpu_cache=True):
    monkeypatch.setattr(ConfigManager, "save_config", lambda self: None)
    cfg = ConfigManager(config_file="nonexistent.json", default_config=DEFAULT_CONFIG)
    cfg.config[CLEAR_GPU_CACHE_CONFIG_KEY] = clear_gpu_cache
    handler = TranscriptionHandler(
        cfg,
        None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda: False,
    )
    return handler


def test_reload_clears_gpu_cache(monkeypatch):
    handler = _build_handler(monkeypatch)
    class Dummy:
        def __init__(self):
            self.unloaded = False
        def load(self):
            pass
        def unload(self):
            self.unloaded = True
    dummy = Dummy()
    handler._asr_backend = dummy
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    empty_called = {"called": False}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_called.__setitem__("called", True))
    monkeypatch.setattr(handler, "_initialize_model_and_processor", lambda: None)
    handler.reload_asr()
    assert dummy.unloaded
    assert empty_called["called"]


def test_switches_backend(monkeypatch):
    handler = _build_handler(monkeypatch, clear_gpu_cache=False)
    monkeypatch.setattr(handler, "_initialize_model_and_processor", lambda: None)
    class SecondBackend:
        def __init__(self, _handler):
            self.loaded = False
            self.unloaded = False
        def load(self):
            self.loaded = True
        def unload(self):
            self.unloaded = True
    backend_registry["second"] = SecondBackend
    handler.asr_backend = "second"
    assert isinstance(handler._asr_backend, SecondBackend)
    assert handler._asr_backend.loaded


def test_adapter_backend_respects_config_transformers(monkeypatch):
    class DummyBackend:
        def __init__(self):
            self.model_id = None
            self.device = None
            self.loaded_kwargs = None

        def load(self, **kwargs):
            self.loaded_kwargs = kwargs

    config = {
        "asr_model_id": "custom-model",
        "asr_compute_device": "cpu",
        "asr_dtype": "float32",
        "asr_cache_dir": "/tmp",
        "asr_ct2_compute_type": "int8",
    }

    class Cfg:
        def get(self, key):
            return config.get(key)

    handler = types.SimpleNamespace(config_manager=Cfg())
    monkeypatch.setattr(
        "src.asr_backends._make_asr_backend", lambda name: DummyBackend()
    )
    adapter = _AdapterBackend(handler, "transformers")
    adapter.load()
    backend = adapter._backend
    assert backend.model_id == "custom-model"
    assert backend.device == "cpu"
    assert backend.loaded_kwargs == {
        "device": "cpu",
        "dtype": "float32",
        "cache_dir": "/tmp",
    }


def test_adapter_backend_respects_config_faster_whisper(monkeypatch):
    class DummyBackend:
        def __init__(self):
            self.model_id = None
            self.device = None
            self.loaded_kwargs = None

        def load(self, **kwargs):
            self.loaded_kwargs = kwargs

    config = {
        "asr_model_id": "custom-model",
        "asr_compute_device": "cuda",
        "asr_dtype": "float16",
        "asr_cache_dir": "/cache",
        "asr_ct2_compute_type": "int8",
    }

    class Cfg:
        def get(self, key):
            return config.get(key)

    handler = types.SimpleNamespace(config_manager=Cfg())
    monkeypatch.setattr(
        "src.asr_backends._make_asr_backend", lambda name: DummyBackend()
    )
    adapter = _AdapterBackend(handler, "faster-whisper")
    adapter.load()
    backend = adapter._backend
    assert backend.model_id == "custom-model"
    assert backend.device == "cuda"
    assert backend.loaded_kwargs == {
        "cache_dir": "/cache",
        "ct2_compute_type": "int8",
    }
