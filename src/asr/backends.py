from __future__ import annotations

from typing import Protocol


class ASRBackend(Protocol):
    """Minimal interface for ASR backends."""

    def load(self, *args, **kwargs) -> None:
        ...

    def warmup(self) -> None:
        ...

    def transcribe(
        self, audio: str | bytes | list[float] | None, **kwargs
    ) -> dict:
        ...

    def unload(self) -> None:
        ...


def make_backend(name: str) -> ASRBackend:
    """Factory that returns an ASR backend by name."""
    normalized = name.strip().lower()
    if normalized == "transformers":
        raise ValueError(
            "The legacy Transformers backend is no longer bundled. Configure the "
            "application to use the CTranslate2 runtime or reinstall a fork that "
            "ships the Transformers pipeline."
        )
    if normalized in {"faster-whisper", "faster_whisper", "ct2", "ctranslate2"}:
        from .backend_faster_whisper import FasterWhisperBackend
        return FasterWhisperBackend()
    raise ValueError(f"Unknown ASR backend: {name}")
