import os
import sys
import importlib
import numpy as np
from types import SimpleNamespace
from pathlib import Path


class DummyTensor(SimpleNamespace):
    def unsqueeze(self, *_):
        return self

    def numpy(self):
        return np.zeros((1,), dtype=np.float32)


# Evita dependência real de torch durante os testes
sys.modules.setdefault(
    "torch",
    SimpleNamespace(
        from_numpy=lambda *_: DummyTensor(),
        cuda=SimpleNamespace(is_available=lambda: False),
    ),
)

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)


def test_initialization_disables_vad_if_model_not_found(monkeypatch):
    """Se o modelo não existir, o VAD deve ser desabilitado."""
    sys.modules.pop("onnxruntime", None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda *_: DummyTensor()),
    )
    real_onnx = importlib.import_module("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", real_onnx)
    vad_module = importlib.reload(importlib.import_module("src.vad_manager"))
    monkeypatch.setattr(
        vad_module.MODEL_PATH.__class__,
        "exists",
        lambda self: False,
    )
    vad = vad_module.VADManager()
    assert not vad.enabled


def test_is_speech_detects_speech(monkeypatch):
    """Confere se is_speech retorna True quando a probabilidade é alta."""
    sys.modules.pop("onnxruntime", None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda *_: DummyTensor()),
    )
    real_onnx = importlib.import_module("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", real_onnx)
    vad_module = importlib.reload(importlib.import_module("src.vad_manager"))
    # Garante que o modelo "exista" para inicialização
    monkeypatch.setattr(
        vad_module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    class DummySession:
        def run(self, *_):
            return [np.array([[0.9]]), np.zeros((2, 1, 128), dtype=np.float32)]

    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
        lambda *a, **k: DummySession(),
    )

    vad = vad_module.VADManager()
    audio = np.zeros(160, dtype=np.float32)
    assert vad.is_speech(audio) is True


def test_is_speech_handles_invalid_input(monkeypatch):
    """Entradas inválidas devem resultar em False sem exceção."""
    sys.modules.pop("onnxruntime", None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda *_: DummyTensor()),
    )
    real_onnx = importlib.import_module("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", real_onnx)
    vad_module = importlib.reload(importlib.import_module("src.vad_manager"))
    monkeypatch.setattr(
        vad_module.MODEL_PATH.__class__,
        "exists",
        lambda self: True,
    )

    class DummySession:
        def run(self, *_):
            return [np.array([[0.1]]), np.zeros((2, 1, 128), dtype=np.float32)]

    monkeypatch.setattr(
        "onnxruntime.InferenceSession",
        lambda *a, **k: DummySession(),
    )

    vad = vad_module.VADManager()
    assert vad.is_speech(None) is False
    assert vad.is_speech([1, 2, 3]) is False


def test_model_path_resolves_with_meipass(monkeypatch):
    """Se sys._MEIPASS estiver definido, MODEL_PATH deve usar esse caminho."""
    tmp_meipass = os.path.abspath("/tmp/meipass")
    monkeypatch.setattr(sys, "_MEIPASS", tmp_meipass, raising=False)

    sys.modules.pop("onnxruntime", None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda *_: DummyTensor()),
    )
    real_onnx = importlib.import_module("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", real_onnx)
    vad_module = importlib.reload(importlib.import_module("src.vad_manager"))

    expected_path = Path(tmp_meipass) / "models" / "silero_vad.onnx"
    assert vad_module.MODEL_PATH == expected_path

