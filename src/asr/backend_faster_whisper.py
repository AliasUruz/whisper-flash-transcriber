from __future__ import annotations

from typing import Any


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    def __init__(self) -> None:
        self.model_id = ""
        self.device = "auto"
        self.model = None

    def load(
        self,
        *,
        model_id: str,
        device: str | None = None,
        ct2_compute_type: str = "default",
        cache_dir: str | None = None,
        dtype: str | None = None,
        **kwargs,
    ) -> None:
        """Load the WhisperModel with the given compute type."""
        from faster_whisper import WhisperModel

        self.model_id = model_id
        self.device = device or "auto"

        device = self.device
        if device in (None, "auto"):
            device = "cuda" if _has_cuda() else "cpu"
        if ct2_compute_type in ("default", "auto"):
            ct2_compute_type = "int8_float16" if device == "cuda" else "int8"

        self.model = WhisperModel(
            self.model_id,
            device=device,
            compute_type=ct2_compute_type,
            download_root=cache_dir or None,
            **kwargs,
        )

    def warmup(self) -> None:
        """Run a dummy inference to initialize the model."""
        if not self.model:
            return
        import numpy as np

        audio = np.zeros(int(16000 * 0.015), dtype="float32")
        segments, _ = self.model.transcribe(audio)
        next(segments, None)

    def transcribe(self, audio: Any, **kwargs) -> dict:
        """Transcribe the provided audio and return just the text."""
        if not self.model:
            raise RuntimeError("Backend not loaded")
        segments, _ = self.model.transcribe(audio, **kwargs)
        text = " ".join(segment.text for segment in segments)
        return {"text": text}

    def unload(self) -> None:
        """Release model resources."""
        self.model = None


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
