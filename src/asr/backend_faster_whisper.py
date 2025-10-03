from __future__ import annotations

from typing import Any

import math


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    def __init__(self, model_id: str = "whisper-large-v3", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.model = None

    def load(self, ct2_compute_type: str = "default", cache_dir: str | None = None, **kwargs) -> None:
        """Load the WhisperModel with the given compute type."""
        from faster_whisper import WhisperModel

        if "model_id" in kwargs:
            self.model_id = kwargs.pop("model_id")

        device = self.device
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"
        if ct2_compute_type == "default":
            ct2_compute_type = "int8_float16" if device == "cuda" else "int8"

        model_name = self.model_id
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        if model_name.startswith("whisper-"):
            model_name = model_name.replace("whisper-", "", 1)

        self.model = WhisperModel(
            model_name,
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
        if "chunk_length_s" in kwargs:
            kwargs["chunk_length"] = kwargs.pop("chunk_length_s")
        # ``WhisperModel.transcribe`` does not accept ``batch_size``. The unified
        # transcription handler still provides the value for backends that use it,
        # so we discard it here to avoid ``TypeError``.
        kwargs.pop("batch_size", None)

        language_segments = kwargs.get("language_detection_segments")
        if language_segments is not None:
            try:
                normalized = math.ceil(float(language_segments))
                kwargs["language_detection_segments"] = max(1, normalized)
            except (TypeError, ValueError):
                # Faster-whisper expects an integer. If coercion fails, fall back to
                # the library default by dropping the argument entirely.
                kwargs.pop("language_detection_segments", None)

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
