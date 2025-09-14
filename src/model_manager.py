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
    """Discover curated models available on disk and in the shared HF cache.

    Only models listed in :data:`CURATED` and containing essential files
    (``config.json`` together with ``model.bin`` or ``model.onnx``) are
    returned. Any other directories or isolated files found in ``cache_dir``
    are ignored. The shared Hugging Face cache is queried as a fallback.
    """

    curated_ids = {c["id"] for c in CURATED}
    installed: List[Dict[str, str]] = []
    seen = set()

    cache_dir = Path(cache_dir)
    if cache_dir.is_dir():
        for backend_dir in cache_dir.iterdir():
            if not backend_dir.is_dir():
                continue
            backend = backend_dir.name
            for model_dir in backend_dir.rglob("*"):
                if not model_dir.is_dir():
                    continue
                rel_id = model_dir.relative_to(backend_dir).as_posix()
                if rel_id in seen or rel_id not in curated_ids:
                    continue
                files_present = {f.name for f in model_dir.iterdir() if f.is_file()}
                if "config.json" not in files_present or not (
                    "model.bin" in files_present or "model.onnx" in files_present
                ):
                    continue
                installed.append({"id": rel_id, "backend": backend, "path": str(model_dir)})
                seen.add(rel_id)

    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id in seen or repo.repo_id not in curated_ids:
                continue
            repo_files = {p.name for p in Path(repo.repo_path).iterdir() if p.is_file()}
            if "config.json" not in repo_files or not (
                "model.bin" in repo_files or "model.onnx" in repo_files
            ):
                continue
            installed.append(
                {
                    "id": repo.repo_id,
                    "backend": "transformers",
                    "path": str(repo.repo_path),
                }
            )
            seen.add(repo.repo_id)
    except Exception:  # pragma: no cover - best effort
        pass

    return installed


def get_model_download_size(model_id: str) -> tuple[int, int]:
    """Return the download size and file count for ``model_id``.

    Parameters
    ----------
    model_id: str
        Hugging Face model identifier.

    Returns
    -------
    tuple[int, int]
        Total size in bytes and number of files available for download.
    """

    api = HfApi()
    info = api.model_info(model_id)
    total = 0
    files = 0
    for sibling in getattr(info, "siblings", []):
        total += sibling.size or 0
        files += 1
    return total, files


def get_installed_size(model_path: str | Path) -> tuple[int, int]:
    """Return the size on disk and file count for an installed model."""

    path = Path(model_path)
    if not path.exists():
        return 0, 0

    total = 0
    files = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            total += p.stat().st_size
    return total, files


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
