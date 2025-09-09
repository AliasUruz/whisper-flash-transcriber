import sys
import types
import numpy as np
import pytest
import pathlib
import os
# add src to sys.path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
psutil_stub = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=0, total=0))
sys.modules.setdefault("psutil", psutil_stub)
sys.modules.setdefault("soundfile", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace())
sys.modules.setdefault("sounddevice", types.SimpleNamespace())
strtobool = lambda x: 1
distutils_stub = types.SimpleNamespace(util=types.SimpleNamespace(strtobool=strtobool))
sys.modules.setdefault("distutils", distutils_stub)
sys.modules.setdefault("distutils.util", distutils_stub.util)
setuptools_stub = types.SimpleNamespace(_distutils=types.SimpleNamespace(util=types.SimpleNamespace(strtobool=strtobool)))
sys.modules.setdefault("setuptools", setuptools_stub)
sys.modules.setdefault("setuptools._distutils", setuptools_stub._distutils)
sys.modules.setdefault("setuptools._distutils.util", setuptools_stub._distutils.util)

# --- Helpers to stub heavy dependencies ---

# Stub torch with configurable cuda behavior
class TorchStub:
    def __init__(self, *, cuda_available=True, mem_info=(8 * 1024**3, 16 * 1024**3)):
        self._cuda_available = cuda_available
        self._mem_info = mem_info

        class CudaStub:
            def __init__(self, outer):
                self.outer = outer
            def is_available(self):
                return self.outer._cuda_available
            def mem_get_info(self, device):
                return self.outer._mem_info
            def empty_cache(self):
                pass
        self.cuda = CudaStub(self)

    def device(self, name):
        return name

    def from_numpy(self, arr):
        class Tensor:
            def __init__(self, data):
                self.data = data
            def unsqueeze(self, axis):
                return self
            def numpy(self):
                return self.data
        return Tensor(arr)


def install_torch_stub(monkeypatch, *, cuda_available=True, mem_info=(8 * 1024**3, 16 * 1024**3)):
    stub = TorchStub(cuda_available=cuda_available, mem_info=mem_info)
    monkeypatch.setitem(sys.modules, 'torch', stub)
    return stub


def install_transformers_stub(monkeypatch):
    tf_stub = types.SimpleNamespace(
        pipeline=lambda *a, **k: None,
        AutoProcessor=object,
        AutoModelForSpeechSeq2Seq=object,
        BitsAndBytesConfig=object,
    )
    monkeypatch.setitem(sys.modules, 'transformers', tf_stub)
    return tf_stub


def install_onnx_stub(monkeypatch, *, speech_prob=1.0):
    class DummySession:
        def __init__(self, path, providers):
            self.path = path
        def run(self, *_):
            prob = np.array([[[speech_prob]]], dtype=np.float32)
            state = np.zeros((2, 1, 128), dtype=np.float32)
            return prob, state
    onnx_stub = types.SimpleNamespace(
        InferenceSession=lambda path, providers=None: DummySession(path, providers),
        get_available_providers=lambda: ['CPUExecutionProvider'],
    )
    monkeypatch.setitem(sys.modules, 'onnxruntime', onnx_stub)
    return onnx_stub

# --- Tests ---


def test_vad_returns_true_when_model_missing(monkeypatch):
    # Simula ausência do modelo silero
    install_torch_stub(monkeypatch)
    install_onnx_stub(monkeypatch)
    import importlib, src.vad_manager as vad_manager
    importlib.reload(vad_manager)
    monkeypatch.setattr(vad_manager, 'MODEL_PATH', vad_manager.MODEL_PATH.with_name('missing.onnx'))
    vm = vad_manager.VADManager(threshold=0.5)
    assert vm.enabled is False
    assert vm.is_speech(np.zeros(16000, dtype=np.float32)) is True


def test_vad_threshold(monkeypatch):
    install_torch_stub(monkeypatch)
    install_onnx_stub(monkeypatch, speech_prob=0.3)
    import importlib, src.vad_manager as vm_module
    importlib.reload(vm_module)
    VADManager = vm_module.VADManager
    vm = VADManager(threshold=0.5)
    assert vm.enabled is True
    assert vm.is_speech(np.zeros(16000, dtype=np.float32)) is False


def test_select_batch_size_cpu_fallback(monkeypatch):
    install_torch_stub(monkeypatch, cuda_available=False)
    from src.utils.batch_size import select_batch_size
    assert select_batch_size(gpu_index=-1, fallback=4) == 4


def test_effective_chunk_length_gpu(monkeypatch):
    install_torch_stub(monkeypatch, cuda_available=True, mem_info=(12 * 1024**3, 16 * 1024**3))
    install_transformers_stub(monkeypatch)
    from src.transcription_handler import TranscriptionHandler
    th = TranscriptionHandler.__new__(TranscriptionHandler)
    th.chunk_length_sec = 30
    th.gpu_index = 0
    assert th._effective_chunk_length() == 60.0


def test_effective_chunk_length_cpu(monkeypatch):
    install_torch_stub(monkeypatch, cuda_available=False)
    install_transformers_stub(monkeypatch)
    from src.transcription_handler import TranscriptionHandler
    th = TranscriptionHandler.__new__(TranscriptionHandler)
    th.chunk_length_sec = 25
    th.gpu_index = -1
    assert th._effective_chunk_length() == 25.0
