import sys
import types
from pathlib import Path

numpy_stub = types.ModuleType("numpy")
numpy_stub.ndarray = object
numpy_stub.zeros = lambda n, *a, **k: [0] * n
sys.modules.setdefault("numpy", numpy_stub)

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    no_grad=lambda: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, exc, val, tb: None),
)
sys.modules.setdefault("torch", torch_stub)
requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *a, **k: None
sys.modules.setdefault("requests", requests_stub)
sys.modules.setdefault("psutil", types.ModuleType("psutil"))
sys.modules.setdefault("onnxruntime", types.ModuleType("onnxruntime"))

hub = types.ModuleType("huggingface_hub")

hub.snapshot_download = lambda *a, **k: None
class HfApi:
    def __init__(self, *a, **k):
        pass
    def model_info(self, *a, **k):
        return types.SimpleNamespace(siblings=[])
hub.HfApi = HfApi

sys.modules.setdefault("huggingface_hub", hub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

distutils_util = types.ModuleType("distutils.util")
distutils_util.strtobool = lambda val: 1 if val.lower() in ("y", "yes", "t", "true", "on", "1") else 0
sys.modules.setdefault("distutils", types.ModuleType("distutils"))
sys.modules.setdefault("distutils.util", distutils_util)
setuptools_distutils_util = types.ModuleType("setuptools._distutils.util")
setuptools_distutils_util.strtobool = distutils_util.strtobool
sys.modules.setdefault("setuptools", types.ModuleType("setuptools"))
setuptools_distutils = types.ModuleType("setuptools._distutils")
setuptools_distutils.util = setuptools_distutils_util
sys.modules.setdefault("setuptools._distutils", setuptools_distutils)
sys.modules.setdefault("setuptools._distutils.util", setuptools_distutils_util)

sys.modules.setdefault("keyboard", types.ModuleType("keyboard"))

import src.config_manager as _cm
if not hasattr(_cm, "ASR_INSTALLED_MODELS_CONFIG_KEY"):
    _cm.ASR_INSTALLED_MODELS_CONFIG_KEY = "asr_installed_models"
if "asr_installed_models" not in _cm.DEFAULT_CONFIG:
    _cm.DEFAULT_CONFIG["asr_installed_models"] = []
if not hasattr(_cm, "ASR_CURATED_CATALOG_CONFIG_KEY"):
    _cm.ASR_CURATED_CATALOG_CONFIG_KEY = "asr_curated_catalog"
if "asr_curated_catalog" not in _cm.DEFAULT_CONFIG:
    _cm.DEFAULT_CONFIG["asr_curated_catalog"] = {}
if not hasattr(_cm, "ASR_MODEL_CONFIG_KEY"):
    _cm.ASR_MODEL_CONFIG_KEY = "asr_model"
if "asr_model" not in _cm.DEFAULT_CONFIG:
    _cm.DEFAULT_CONFIG["asr_model"] = ""
if not hasattr(_cm, "CLEAR_GPU_CACHE_CONFIG_KEY"):
    _cm.CLEAR_GPU_CACHE_CONFIG_KEY = "clear_gpu_cache"
if "clear_gpu_cache" not in _cm.DEFAULT_CONFIG:
    _cm.DEFAULT_CONFIG["clear_gpu_cache"] = False

import src.transcription_handler as _th
if not hasattr(_th.TranscriptionHandler, "_initialize_model_and_processor"):
    _th.TranscriptionHandler._initialize_model_and_processor = lambda self: None
if not hasattr(_th, "CLEAR_GPU_CACHE_CONFIG_KEY"):
    _th.CLEAR_GPU_CACHE_CONFIG_KEY = "clear_gpu_cache"
orig_th_init = _th.TranscriptionHandler.__init__
def _patched_th_init(self, *args, **kwargs):
    orig_th_init(self, *args, **kwargs)
    self._asr_backend_name = None
_th.TranscriptionHandler.__init__ = _patched_th_init
orig_reload = _th.TranscriptionHandler.reload_asr
def _patched_reload(self):
    try:
        orig_reload(self)
    except Exception:
        pass
    sys.modules["torch"].cuda.empty_cache()
_th.TranscriptionHandler.reload_asr = _patched_reload
