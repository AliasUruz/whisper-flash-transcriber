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
        chunk_length = _coerce_chunk_length(kwargs.pop("chunk_length_s", None))
        if chunk_length is not None:
            kwargs["chunk_length"] = chunk_length
        # ``WhisperModel.transcribe`` does not accept ``batch_size``. The unified
        # transcription handler still provides the value for backends that use it,
        # so we discard it here to avoid ``TypeError``.
        kwargs.pop("batch_size", None)
        sanitized_segments = _sanitize_language_detection_segments(
            kwargs.get("language_detection_segments")
        )
        if sanitized_segments is not None:
            kwargs["language_detection_segments"] = sanitized_segments
        else:
            computed_segments = _segments_from_chunk_length(kwargs.get("chunk_length"))
            if computed_segments is not None:
                kwargs["language_detection_segments"] = computed_segments
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


def _coerce_chunk_length(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        chunk = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(chunk) or math.isinf(chunk):
        return None
    return max(0.0, chunk)


def _sanitize_language_detection_segments(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(round(float(raw)))
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return 1
    return value


def _segments_from_chunk_length(chunk_length: Any) -> int | None:
    if chunk_length is None:
        return None
    try:
        chunk = float(chunk_length)
    except (TypeError, ValueError):
        return None
    if chunk <= 0:
        return 1
    approx_segments = chunk / 30.0
    return _sanitize_language_detection_segments(approx_segments)
