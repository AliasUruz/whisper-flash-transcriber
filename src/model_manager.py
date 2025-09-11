"""Utilities for curated ASR model management."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download


class DownloadCancelledError(Exception):
    """Raised when a model download is cancelled by the user."""

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
    """Discover models available on disk and in the shared HF cache.

    Detection covers curated backends (``transformers`` and ``ct2``), custom
    directory layouts (e.g., Silero VAD), single model files, and the global
    Hugging Face cache. Any model found is returned with its backend (or
    ``"custom"`` when unknown) and resolved path.
    """

    installed: List[Dict[str, str]] = []
    seen = set()

    cache_dir = Path(cache_dir)
    if cache_dir.is_dir():
        for item in cache_dir.iterdir():
            if item.is_dir():
                subdirs = [p for p in item.iterdir() if p.is_dir()]
                if subdirs:
                    backend = item.name
                    for model_dir in subdirs:
                        mid = model_dir.name
                        installed.append({"id": mid, "backend": backend, "path": str(model_dir)})
                        seen.add(mid)
                else:
                    mid = item.name
                    installed.append({"id": mid, "backend": "custom", "path": str(item)})
                    seen.add(mid)
            elif item.is_file():
                mid = item.stem
                installed.append({"id": mid, "backend": "custom", "path": str(item)})
                seen.add(mid)

    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id not in seen:
                installed.append(
                    {
                        "id": repo.repo_id,
                        "backend": "transformers",
                        "path": str(repo.repo_path),
                    }
                )
    except Exception:  # pragma: no cover - best effort
        pass

    return installed


def get_model_download_size(model_id: str) -> int:
    """Return the total download size in bytes for ``model_id``."""

    api = HfApi()
    info = api.model_info(model_id)
    return sum((s.size or 0) for s in getattr(info, "siblings", []))


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

    try:
        if backend == "transformers":
            snapshot_download(repo_id=model_id, local_dir=str(local_dir), allow_patterns=None)
        elif backend == "ct2":
            from faster_whisper import WhisperModel

            WhisperModel(
                model_id,
                device="cpu",
                compute_type=quant or "int8",
                download_root=str(local_dir),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    except KeyboardInterrupt as exc:
        raise DownloadCancelledError("Model download cancelled by user.") from exc

    return str(local_dir)
