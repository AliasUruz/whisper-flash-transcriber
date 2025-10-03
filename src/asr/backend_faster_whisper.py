from __future__ import annotations

from typing import Any


# Mapping between Hugging Face repository identifiers and the canonical
# identifiers expected by faster-whisper / CTranslate2.
_MODEL_ID_ALIASES: dict[str, str] = {
    "whisper-tiny": "tiny",
    "whisper-tiny.en": "tiny.en",
    "whisper-base": "base",
    "whisper-base.en": "base.en",
    "whisper-small": "small",
    "whisper-small.en": "small.en",
    "whisper-medium": "medium",
    "whisper-medium.en": "medium.en",
    "whisper-large-v1": "large-v1",
    "whisper-large-v2": "large-v2",
    "whisper-large-v3": "large-v3",
    "whisper-large": "large",
    "whisper-large-v3-turbo": "large-v3-turbo",
    "whisper-turbo": "turbo",
    "whisper-distil-large-v2": "distil-large-v2",
    "whisper-distil-medium.en": "distil-medium.en",
    "whisper-distil-small.en": "distil-small.en",
    "whisper-distil-large-v3": "distil-large-v3",
    "whisper-distil-large-v3.5": "distil-large-v3.5",
}


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.model = None

    def load(self, ct2_compute_type: str = "default", cache_dir: str | None = None, **kwargs) -> None:
        """Load the WhisperModel with the given compute type."""
        from faster_whisper import WhisperModel

        device = self.device
        if device == "auto":
            device = "cuda" if _has_cuda() else "cpu"
        if ct2_compute_type == "default":
            ct2_compute_type = "int8_float16" if device == "cuda" else "int8"

        model_id = _resolve_model_identifier(self.model_id)

        self.model = WhisperModel(
            model_id,
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


def _resolve_model_identifier(model_id: str) -> str:
    """Resolve the identifier passed to faster-whisper."""

    if _looks_like_local_path(model_id):
        return model_id

    repo_id = model_id.split("/")[-1]
    lowered = repo_id.lower()

    if lowered in _MODEL_ID_ALIASES:
        return _MODEL_ID_ALIASES[lowered]

    if lowered.startswith("whisper-"):
        candidate = repo_id[len("whisper-") :]
        if candidate:
            return candidate

    return repo_id


def _looks_like_local_path(value: str) -> bool:
    """Best-effort heuristic to detect local filesystem paths."""

    if not value:
        return False
    if value.startswith(("./", "../", "/", "~")):
        return True
    if "\\" in value or ":" in value:
        return True
    return False
