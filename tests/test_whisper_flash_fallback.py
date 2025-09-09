import types
import sys

# Stubs to avoid heavy dependencies
torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        device_count=lambda: 0,
    ),
    float16=object(),
    float32=object(),
    no_grad=lambda: types.SimpleNamespace(__enter=lambda self: None, __exit=lambda self, exc_type, exc, tb: False)(),
)
sys.modules['torch'] = torch_stub

numpy_stub = types.ModuleType('numpy')
numpy_stub.ndarray = object
numpy_stub.float32 = float
numpy_stub.linspace = lambda start, stop, num, endpoint, dtype: [0.0] * num
numpy_stub.sin = lambda x: x
numpy_stub.pi = 3.141592653589793
sys.modules['numpy'] = numpy_stub

sys.modules['soundfile'] = types.ModuleType('soundfile')
sys.modules['requests'] = types.ModuleType('requests')
sys.modules['sounddevice'] = types.ModuleType('sounddevice')
sys.modules['onnxruntime'] = types.ModuleType('onnxruntime')
sys.modules['psutil'] = types.ModuleType('psutil')

pipeline_called = {"value": False}

class DummyProcessor:
    tokenizer = object()
    feature_extractor = object()

def pipeline_stub(*args, **kwargs):
    pipeline_called["value"] = True
    class DummyPipe:
        def __call__(self, *args, **kwargs):
            return {"text": "ok"}
    return DummyPipe()

distutils_util = types.ModuleType('distutils.util')
def _strtobool(val):
    return 1 if val.lower() in ('y','yes','t','true','on','1') else 0
sys.modules['distutils'] = types.ModuleType('distutils')
sys.modules['distutils.util'] = distutils_util

from src.config_manager import ConfigManager, DEFAULT_CONFIG
import src.transcription_handler as th


def test_fallback_pipeline_loads(monkeypatch):
    monkeypatch.setattr(ConfigManager, 'save_config', lambda self: None)
    monkeypatch.setattr(th, 'ensure_download', lambda *a, **k: None)
    monkeypatch.setattr(th, 'pipeline', pipeline_stub)
    monkeypatch.setattr(th, 'AutoProcessor', types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyProcessor()))
    monkeypatch.setattr(th, 'AutoModelForSpeechSeq2Seq', types.SimpleNamespace(from_pretrained=lambda *a, **k: object()))
    cfg = ConfigManager(config_file='nonexistent.json', default_config=DEFAULT_CONFIG)
    handler = th.TranscriptionHandler(
        cfg,
        None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: None,
        lambda: False,
    )
    handler._load_model_task()
    assert pipeline_called["value"]
    assert handler._asr_loaded
