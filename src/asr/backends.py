from __future__ import annotations

from typing import Protocol


class ASRBackend(Protocol):
    """Minimal interface for ASR backends."""

    def load(self, *args, **kwargs) -> None: ...

    def warmup(self) -> None: ...

    def transcribe(self, audio: str | bytes | list[float] | None, **kwargs) -> dict: ...

    def unload(self) -> None: ...


def make_backend(name: str) -> ASRBackend:
    """Factory that returns an ASR backend by name."""
    normalized = name.strip().lower()
    if normalized == "transformers":
        from .backend_transformers import TransformersBackend
        return TransformersBackend()
    if normalized in {"faster-whisper", "faster_whisper"}:
        from .backend_faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend()
    raise ValueError(f"Unknown ASR backend: {name}")
