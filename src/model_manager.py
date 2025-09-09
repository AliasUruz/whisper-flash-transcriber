"""Utilities for curated ASR model management."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download

CURATED: List[Dict[str, str]] = [
    {"id": "openai/whisper-large-v3", "backend": "transformers"},
    {"id": "openai/whisper-large-v3-turbo", "backend": "transformers"},
    {"id": "distil-whisper/distil-large-v3", "backend": "transformers"},
    {"id": "Systran/faster-whisper-large-v3", "backend": "ct2"},
    {"id": "h2oai/faster-whisper-large-v3-turbo", "backend": "ct2"},
    {"id": "Systran/faster-distil-whisper-large-v3", "backend": "ct2"},
]


def list_catalog() -> List[Dict[str, str]]:
    """Return curated catalog entries."""
    return CURATED


def list_installed(cache_dir: str | Path) -> List[Dict[str, str]]:
    """List models already downloaded in ``cache_dir``."""
    installed: List[Dict[str, str]] = []
    cache_dir = Path(cache_dir)
    for backend in ("transformers", "ct2"):
        backend_path = cache_dir / backend
        if not backend_path.is_dir():
            continue
        for model_dir in backend_path.iterdir():
            if model_dir.is_dir():
                installed.append(
                    {"id": model_dir.name, "backend": backend, "path": str(model_dir)}
                )
    return installed


def ensure_download(
    model_id: str,
    backend: str,
    cache_dir: str | Path,
    quant: str | None = None,
) -> str:
    """Ensure that the given model is present locally.

    Parameters
    ----------
    model_id: str
        Full model identifier as in the curated catalog.
    backend: str
        Either ``"transformers"`` or ``"ct2"``.
    cache_dir: str | Path
        Root directory where models are cached.
    quant: str | None
        Quantization branch for CT2 models. Ignored for Transformers.
    """

    cache_dir = Path(cache_dir)
    local_dir = cache_dir / backend / model_id
    if local_dir.is_dir() and any(local_dir.iterdir()):
        return str(local_dir)

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    if backend == "transformers":
        snapshot_download(repo_id=model_id, local_dir=str(local_dir), allow_patterns=None)
    elif backend == "ct2":
        from faster_whisper import WhisperModel

        WhisperModel(model_id, device="cpu", compute_type=quant or "int8", download_root=str(local_dir))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return str(local_dir)
