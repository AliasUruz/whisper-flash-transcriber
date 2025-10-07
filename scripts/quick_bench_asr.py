"""Quick benchmarking utility for ASR backends."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:  # pragma: no cover - apenas para ambientes de teste sem PortAudio
    import sounddevice  # type: ignore  # noqa: F401
except Exception:
    class _DummyStream:
        def __init__(self, *_, **__):
            self.active = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def close(self):
            self.active = False

    sys.modules["sounddevice"] = SimpleNamespace(
        InputStream=_DummyStream,
        OutputStream=_DummyStream,
        sleep=lambda _ms: None,
        PortAudioError=RuntimeError,
        CallbackStop=Exception,
    )

try:  # pragma: no cover - ambientes de CI sem libsndfile
    import soundfile  # type: ignore  # noqa: F401
except Exception:
    class _DummySoundFile:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def write(self, *_args, **_kwargs):
            return None

        def close(self):
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _dummy_sf_write(filename, *_args, **_kwargs):
        try:
            Path(filename).touch()
        except Exception:
            pass

    sys.modules["soundfile"] = SimpleNamespace(
        SoundFile=_DummySoundFile,
        write=_dummy_sf_write,
    )

try:  # pragma: no cover - ambientes de CI sem onnxruntime
    import onnxruntime  # type: ignore  # noqa: F401
except Exception:
    class _DummyInferenceSession:
        def __init__(self, *_args, **_kwargs):
            self._state = np.zeros((2, 1, 128), dtype=np.float32)

        def run(self, _outputs=None, inputs=None):
            if isinstance(inputs, dict) and "state" in inputs:
                self._state = inputs["state"]
            prob = np.array([[[0.0]]], dtype=np.float32)
            return [prob, self._state]

    sys.modules["onnxruntime"] = SimpleNamespace(
        InferenceSession=_DummyInferenceSession,
        get_available_providers=lambda: ["CPUExecutionProvider"],
    )

try:  # pragma: no cover - ambientes headless sem customtkinter
    import customtkinter  # type: ignore  # noqa: F401
except Exception:
    class _DummyVar:
        def __init__(self, value=None):
            self._value = value

        def get(self):
            return self._value

        def set(self, value):
            self._value = value

    class _DummyWidget:
        def __init__(self, *args, **kwargs):
            self._value = kwargs.get("value")
            self._text = ""

        # Tk-like geometry methods -------------------------------------------------
        def pack(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def place(self, *args, **kwargs):
            return None

        # Widget lifecycle --------------------------------------------------------
        def destroy(self):
            return None

        def withdraw(self):
            return None

        def lift(self):
            return None

        def focus_force(self):
            return None

        def update_idletasks(self):
            return None

        def winfo_exists(self):
            return False

        def winfo_screenwidth(self):
            return 1920

        def winfo_screenheight(self):
            return 1080

        # Configuration -----------------------------------------------------------
        def configure(self, *args, **kwargs):
            if "state" in kwargs:
                self._state = kwargs["state"]
            return None

        config = configure

        def set(self, value):
            self._value = value

        def get(self, *args, **kwargs):
            if self._text:
                return self._text
            return getattr(self, "_value", None)

        def insert(self, *args, **kwargs):
            if args:
                self._text += str(args[-1])
            return None

        def delete(self, *args, **kwargs):
            self._text = ""
            return None

        def see(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def resizable(self, *args, **kwargs):
            return None

        def attributes(self, *args, **kwargs):
            return None

        def geometry(self, *args, **kwargs):
            return None

        def after(self, *args, **kwargs):
            return None

        def update_menu(self):
            return None

    class _DummyScrollableFrame(_DummyWidget):
        def pack(self, *args, **kwargs):
            return None

    class _DummyOptionMenu(_DummyWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._command = kwargs.get("command")

        def configure(self, *args, **kwargs):
            if "command" in kwargs:
                self._command = kwargs["command"]
            return super().configure(*args, **kwargs)

        config = configure

        def set(self, value):
            super().set(value)
            if callable(self._command):
                self._command(value)

    sys.modules["customtkinter"] = SimpleNamespace(
        CTk=_DummyWidget,
        CTkToplevel=_DummyWidget,
        CTkFrame=_DummyWidget,
        CTkLabel=_DummyWidget,
        CTkButton=_DummyWidget,
        CTkSwitch=_DummyWidget,
        CTkTextbox=_DummyWidget,
        CTkOptionMenu=_DummyOptionMenu,
        CTkEntry=_DummyWidget,
        CTkScrollableFrame=_DummyScrollableFrame,
        CTkSlider=_DummyWidget,
        BooleanVar=_DummyVar,
        StringVar=_DummyVar,
        DoubleVar=_DummyVar,
        IntVar=_DummyVar,
        set_appearance_mode=lambda *args, **kwargs: None,
        set_default_color_theme=lambda *args, **kwargs: None,
        CTkFont=lambda *args, **kwargs: None,
    )

try:  # pragma: no cover - ambientes headless sem pystray
    import pystray  # type: ignore  # noqa: F401
except Exception:
    class _DummyMenuItem:
        def __init__(self, text, action=None, *, default=False, enabled=True, radio=False, checked=None):
            self.text = text
            self.action = action
            self.default = default
            self.enabled = enabled
            self.radio = radio
            self.checked = checked

        def __call__(self, icon=None, item=None):  # pragma: no cover - compatibility helper
            if callable(self.action):
                return self.action(icon, item)
            return None

    class _DummyMenu(tuple):
        SEPARATOR = object()

        def __new__(cls, *items):
            return super().__new__(cls, items)

    class _DummyIcon:
        def __init__(self, name, icon, title, menu=None):
            self.name = name
            self.icon = icon
            self.title = title
            self.menu = menu

        def run_detached(self):
            return None

        def stop(self):
            return None

        def update_menu(self):
            return None

    sys.modules["pystray"] = SimpleNamespace(
        Icon=_DummyIcon,
        MenuItem=_DummyMenuItem,
        Menu=_DummyMenu,
    )

try:  # pragma: no cover - ambientes headless sem Pillow
    from PIL import Image as _PILImage  # type: ignore  # noqa: F401
    from PIL import ImageDraw as _PILImageDraw  # type: ignore  # noqa: F401
except Exception:
    pil_module = ModuleType("PIL")

    class _DummyImageModule(ModuleType):
        def __init__(self):
            super().__init__("PIL.Image")

        @staticmethod
        def new(mode, size, color):
            return {"mode": mode, "size": size, "color": color}

    class _DummyImageDrawModule(ModuleType):
        def __init__(self):
            super().__init__("PIL.ImageDraw")

        class Draw:
            def __init__(self, _image):
                self._image = _image

            def rectangle(self, *_args, **_kwargs):
                return None

    image_module = _DummyImageModule()
    image_draw_module = _DummyImageDrawModule()

    pil_module.Image = image_module
    pil_module.ImageDraw = image_draw_module

    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module
    sys.modules["PIL.ImageDraw"] = image_draw_module

if TYPE_CHECKING:  # pragma: no cover - apenas para tipagem estática
    from src.config_manager import ConfigManager as _ConfigManager
    from src.transcription_handler import TranscriptionHandler as _TranscriptionHandler
else:  # pragma: no cover - evita NameError em tempo de execução
    _ConfigManager = Any  # type: ignore[assignment]
    _TranscriptionHandler = Any  # type: ignore[assignment]

_CONFIG_MANAGER_CLS: type[_ConfigManager] | None = None
_TRANSCRIPTION_HANDLER_CLS: type[_TranscriptionHandler] | None = None
_TRANSCRIPTION_HANDLER_MODULE: ModuleType | None = None


def _ensure_app_dependencies() -> None:
    """Importa dependências do app somente após configurar dummies."""

    global _CONFIG_MANAGER_CLS, _TRANSCRIPTION_HANDLER_CLS, _TRANSCRIPTION_HANDLER_MODULE

    if (
        _CONFIG_MANAGER_CLS is not None
        and _TRANSCRIPTION_HANDLER_CLS is not None
        and _TRANSCRIPTION_HANDLER_MODULE is not None
    ):
        return

    from src.config_manager import ConfigManager as _LoadedConfigManager
    from src.transcription_handler import TranscriptionHandler as _LoadedTranscriptionHandler
    import src.transcription_handler as loaded_transcription_handler_module

    _CONFIG_MANAGER_CLS = _LoadedConfigManager
    _TRANSCRIPTION_HANDLER_CLS = _LoadedTranscriptionHandler
    _TRANSCRIPTION_HANDLER_MODULE = loaded_transcription_handler_module


def main() -> None:
    _ensure_app_dependencies()
    assert _CONFIG_MANAGER_CLS is not None
    assert _TRANSCRIPTION_HANDLER_CLS is not None

    cfg = _CONFIG_MANAGER_CLS()

    handler = _TRANSCRIPTION_HANDLER_CLS(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=lambda: None,
        on_model_error_callback=lambda e: print(f"error: {e}"),
        on_transcription_result_callback=lambda text, _orig: print(text),
        on_agent_result_callback=None,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )

    handler.reload_asr()
    audio = np.zeros(int(16000 * 15), dtype="float32")
    times = []
    for _ in range(3):
        start = time.perf_counter()
        handler._asr_backend.transcribe(audio, chunk_length_s=30, batch_size=1)
        times.append(time.perf_counter() - start)
    median = sorted(times)[len(times) // 2]
    print(f"median_time={median:.2f}s")


# --- Tests ---------------------------------------------------------------


def _make_handler_for_tests(chunk: float = 30.0, gpu_index: int = 0) -> "_TranscriptionHandler":
    _ensure_app_dependencies()
    assert _TRANSCRIPTION_HANDLER_CLS is not None

    handler = _TRANSCRIPTION_HANDLER_CLS.__new__(_TRANSCRIPTION_HANDLER_CLS)
    handler.chunk_length_sec = chunk
    handler.gpu_index = gpu_index
    return handler


def _mock_cuda_env(
    monkeypatch,
    *,
    available: bool,
    free_gb: float = 0.0,
    total_gb: float = 16.0,
    raise_error: bool = False,
) -> None:
    """Configure torch.cuda mocks for TranscriptionHandler tests."""

    fake_cuda = SimpleNamespace()
    fake_cuda.is_available = lambda: available

    if raise_error:
        def _fail(_device):
            raise RuntimeError("mock mem_get_info failure")

        fake_cuda.mem_get_info = _fail
    else:
        def _mem_info(_device):
            return (int(free_gb * (1024 ** 3)), int(total_gb * (1024 ** 3)))

        fake_cuda.mem_get_info = _mem_info

    _ensure_app_dependencies()
    assert _TRANSCRIPTION_HANDLER_MODULE is not None

    monkeypatch.setattr(_TRANSCRIPTION_HANDLER_MODULE.torch, "cuda", fake_cuda)
    monkeypatch.setattr(_TRANSCRIPTION_HANDLER_MODULE.torch, "device", lambda spec: spec)


def test_effective_chunk_length_cpu(monkeypatch):
    handler = _make_handler_for_tests(chunk=37.0, gpu_index=-1)
    _mock_cuda_env(monkeypatch, available=False)
    assert handler._effective_chunk_length() == 37.0


def test_effective_chunk_length_gpu_buckets(monkeypatch):
    handler = _make_handler_for_tests(chunk=30.0, gpu_index=0)
    scenarios = [
        (12.5, 60.0),
        (8.2, 45.0),
        (6.4, 30.0),
        (4.3, 20.0),
        (3.0, 15.0),
    ]
    for free_gb, expected in scenarios:
        _mock_cuda_env(monkeypatch, available=True, free_gb=free_gb)
        assert handler._effective_chunk_length() == expected


def test_effective_chunk_length_meminfo_failure(monkeypatch):
    handler = _make_handler_for_tests(chunk=42.0, gpu_index=0)
    _mock_cuda_env(monkeypatch, available=True, raise_error=True)
    assert handler._effective_chunk_length() == 42.0


def test_apply_settings_payload_headless():
    from src.ui_manager import UIManager

    class CoreStub:
        def __init__(self):
            self.received = None

        def apply_settings_from_external(self, **kwargs):
            self.received = kwargs

    core_stub = CoreStub()
    config_stub = SimpleNamespace()
    root_stub = SimpleNamespace()
    ui = UIManager(root_stub, config_stub, core_stub, is_running_as_admin=False)

    payload = {
        "new_key": "f5",
        "new_mode": "press",
        "new_chunk_length_sec": 48.0,
    }

    ui.apply_settings_payload(payload)

    assert core_stub.received == payload


if __name__ == "__main__":
    main()
