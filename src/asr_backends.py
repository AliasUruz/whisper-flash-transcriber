from __future__ import annotations

from typing import Any, Callable, Protocol

from .asr import make_backend as _make_asr_backend


class ASRBackend(Protocol):
    """Interface mínima que os backends de ASR devem implementar."""

    def load(self, *args, **kwargs) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int): ...


class DummyBackend:
    """Backend de exemplo sem funcionalidade real."""

    def __init__(self, handler):
        self.handler = handler

    def load(self, *args, **kwargs) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        return {"text": ""}


backend_registry: dict[str, Callable[[Any], ASRBackend]] = {
    "dummy": DummyBackend,
}


class _AdapterBackend:
    """Adaptador para o backend CTranslate2 nativo do aplicativo."""

    def __init__(self, handler):
        self._handler = handler
        self._backend: ASRBackend | None = None

    def load(self) -> None:
        cfg = self._handler.config_manager
        cache_dir = cfg.get("asr_cache_dir") or None
        cpu_threads = cfg.get("asr_ct2_cpu_threads")
        _, model_id, device, compute_type = self._handler._resolve_asr_settings()

        backend = _make_asr_backend("ctranslate2")
        if hasattr(backend, "model_id"):
            backend.model_id = model_id
        if hasattr(backend, "device"):
            backend.device = device

        kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "ct2_compute_type": compute_type,
        }
        if cpu_threads:
            kwargs["cpu_threads"] = cpu_threads

        device_index = getattr(self._handler, "gpu_index", None)
        if isinstance(device_index, int) and device_index >= 0:
            kwargs["device_index"] = device_index

        backend.load(**{k: v for k, v in kwargs.items() if v not in (None, "")})
        self._backend = backend

    def unload(self) -> None:
        if self._backend:
            self._backend.unload()
            self._backend = None

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        if not self._backend:
            raise RuntimeError("Backend não carregado")
        return self._backend.transcribe(
            audio_source, chunk_length_s=chunk_length_s, batch_size=batch_size
        )


def register_default_backends():
    backend_registry.update(
        {
            "ct2": _AdapterBackend,
            "ctranslate2": _AdapterBackend,
            "faster-whisper": _AdapterBackend,
        }
    )


register_default_backends()
