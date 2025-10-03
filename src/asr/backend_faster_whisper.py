from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "large-v3-turbo"


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

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.model = None
        self._resolved_model_id: str | None = None

    def load(
        self,
        *,
        model_id: str | None = None,
        device: str | None = None,
        ct2_compute_type: str = "default",
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        """Load the WhisperModel with the given compute type."""
        from faster_whisper import WhisperModel

        if model_id:
            self.model_id = model_id
        if device:
            self.device = device

        requested_device = self.device or "auto"
        device_kind, device_index = self._parse_device(requested_device)
        if device_kind == "auto":
            device_kind = "cuda" if _has_cuda() else "cpu"
        if ct2_compute_type == "default":
            ct2_compute_type = "int8_float16" if device_kind == "cuda" else "int8"

        resolved_id = self._normalize_model_id(self.model_id)
        self._resolved_model_id = resolved_id

        init_kwargs = {
            "device": device_kind,
            "compute_type": ct2_compute_type,
            "download_root": cache_dir or None,
            **kwargs,
        }
        if device_index is not None:
            init_kwargs["device_index"] = device_index

        logging.info(
            "Inicializando WhisperModel (requested_id=%s, resolved_id=%s, device=%s, device_index=%s, compute_type=%s)",
            self.model_id,
            resolved_id,
            device_kind,
            device_index,
            ct2_compute_type,
        )

        self.model = WhisperModel(
            resolved_id,
            **init_kwargs,
        )
        self.device = self._compose_device(device_kind, device_index)

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
        chunk_length_s = kwargs.pop("chunk_length_s", None)
        if chunk_length_s is not None:
            kwargs.setdefault("chunk_length", chunk_length_s)
        segments, _ = self.model.transcribe(audio, **kwargs)
        text = " ".join(segment.text for segment in segments)
        return {"text": text}

    def unload(self) -> None:
        """Release model resources."""
        self.model = None

    def _normalize_model_id(self, raw_id: str | None) -> str:
        candidate = (raw_id or "").strip()
        if not candidate:
            candidate = DEFAULT_MODEL_ID

        path_candidate = Path(candidate).expanduser()
        if path_candidate.exists():
            return str(path_candidate)

        normalized = candidate.replace("\\", "/")
        if "/" in normalized:
            normalized = normalized.split("/")[-1]
        lowered = normalized.lower()

        if lowered == "auto":
            return DEFAULT_MODEL_ID

        if lowered.startswith("whisper-"):
            normalized = normalized[len("whisper-") :]
            lowered = normalized.lower()

        return normalized

    @staticmethod
    def _parse_device(device: str) -> tuple[str, int | None]:
        value = (device or "auto").strip()
        if value.startswith("cuda:"):
            try:
                return "cuda", int(value.split(":", 1)[1])
            except ValueError:
                return "cuda", None
        return value or "auto", None

    @staticmethod
    def _compose_device(device: str, index: int | None) -> str:
        if index is None or device in {"cpu", "auto"}:
            return device
        return f"{device}:{index}"


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
