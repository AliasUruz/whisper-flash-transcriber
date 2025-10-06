from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

from .asr.backends import make_backend as _make_asr_backend


class ASRBackend(Protocol):
    """Interface mínima que os backends de ASR devem implementar."""

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int): ...


class WhisperBackend:
    """Backend padrão utilizando pipeline Hugging Face."""

    def __init__(self, handler):
        self.handler = handler

    def load(self) -> None:
        self.handler._initialize_model_and_processor()

    def unload(self) -> None:
        self.handler.unload()

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        generate_kwargs = {"task": "transcribe", "language": None}
        return self.handler.pipe(
            audio_source,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )


class DummyBackend:
    """Backend de exemplo sem funcionalidade real."""

    def __init__(self, handler):
        self.handler = handler

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        return {"text": ""}


backend_registry: Dict[str, Callable[[Any], ASRBackend]] = {
    "whisper": WhisperBackend,
    "dummy": DummyBackend,
}


class _AdapterBackend:
    """Adaptador que integra os novos backends definidos em ``src/asr``."""

    def __init__(self, handler, name: str):
        self._handler = handler
        self._name = name
        self._backend: ASRBackend | None = None

    def load(self) -> None:
        cfg = self._handler.config_manager
        model_id = cfg.get("asr_model_id")
        device = cfg.get("asr_compute_device") or "auto"
        dtype = cfg.get("asr_dtype") or "auto"
        cache = cfg.get("asr_cache_dir") or None
        ct2_type = cfg.get("asr_ct2_compute_type") or "default"

        backend = _make_asr_backend(self._name)
        if hasattr(backend, "model_id"):
            backend.model_id = model_id
        if hasattr(backend, "device"):
            backend.device = device

        if self._name == "transformers":
            backend.load(device=device, dtype=dtype, cache_dir=cache)
        elif self._name == "faster-whisper":
            backend.load(cache_dir=cache, ct2_compute_type=ct2_type)
        else:
            backend.load()
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


backend_registry.update(
    {
        "transformers": lambda h: _AdapterBackend(h, "transformers"),
        "ct2": lambda h: _AdapterBackend(h, "faster-whisper"),
        "ctranslate2": lambda h: _AdapterBackend(h, "faster-whisper"),
        "faster-whisper": lambda h: _AdapterBackend(h, "faster-whisper"),
    }
)
