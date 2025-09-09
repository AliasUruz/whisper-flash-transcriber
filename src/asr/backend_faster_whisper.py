from __future__ import annotations

from typing import Any


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    def __init__(self, model_id: str = "whisper-large-v3", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.model = None

    def load(self, ct2_compute_type: str = "default", **kwargs) -> None:
        """Load the WhisperModel with the given compute type."""
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=ct2_compute_type,
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
        text = "".join(segment.text for segment in segments)
        return {"text": text}

    def unload(self) -> None:
        """Release model resources."""
        self.model = None
