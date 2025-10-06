from __future__ import annotations

import math
from pathlib import Path
from typing import Any


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

        model_source: str | None = None
        download_root: str | None = cache_dir or None
        if cache_dir:
            try:
                cache_path = Path(cache_dir).expanduser()
                from .. import model_manager as model_manager_module

                canonical_dir = model_manager_module.get_installation_dir(
                    cache_path,
                    "ctranslate2",
                    self.model_id,
                )
                candidate_dirs = [canonical_dir]
                for alias in ("ctranslate2", "faster-whisper"):
                    candidate = cache_path / alias / self.model_id
                    if candidate not in candidate_dirs:
                        candidate_dirs.append(candidate)

                for candidate in candidate_dirs:
                    if _looks_like_ct2_installation(candidate):
                        model_source = str(candidate)
                        download_root = None
                        break
            except Exception:
                model_source = None

        self.model = WhisperModel(
            model_source or model_name,
            device=device,
            compute_type=ct2_compute_type,
            download_root=download_root,
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


def _looks_like_ct2_installation(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
    except Exception:
        return False

    has_config = False
    has_weights = False
    try:
        iterator = path.rglob("*")
    except Exception:
        return False

    for candidate in iterator:
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        if name == "config.json":
            has_config = True
            continue
        if name in {"model.bin", "model.onnx", "model.safetensors"} or (
            name.endswith((".bin", ".onnx", ".safetensors")) and "model" in name
        ):
            has_weights = True
        if has_config and has_weights:
            return True
    return has_config and has_weights
