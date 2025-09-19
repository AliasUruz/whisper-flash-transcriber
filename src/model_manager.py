"""Utilities for curated ASR model management."""

from __future__ import annotations

import copy
import logging
import shutil
import time
from pathlib import Path
from threading import Event, RLock
from typing import Dict, List

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download


MODEL_LOGGER = logging.getLogger("whisper_recorder.model")


class DownloadCancelledError(Exception):
    """Raised when a model download is cancelled by the user."""

    def __init__(
        self,
        message: str = "Model download cancelled.",
        *,
        by_user: bool = False,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.by_user = by_user
        self.timed_out = timed_out

# Curated catalog of officially supported ASR models.
# Each entry maps a Hugging Face model id to the backend that powers it.
CURATED: List[Dict[str, str]] = [
    {"id": "openai/whisper-large-v3", "backend": "transformers"},
    {"id": "openai/whisper-large-v3-turbo", "backend": "transformers"},
    {"id": "distil-whisper/distil-large-v3", "backend": "transformers"},
]

DISPLAY_NAMES: Dict[str, str] = {
    "openai/whisper-large-v3": "Whisper Large v3",
    "openai/whisper-large-v3-turbo": "Whisper Large v3 Turbo",
    "distil-whisper/distil-large-v3": "Distil Whisper Large v3",
    "Systran/faster-whisper-large-v3": "Faster Whisper Large v3",
    "h2oai/faster-whisper-large-v3-turbo": "Faster Whisper Large v3 Turbo",
    "Systran/faster-distil-whisper-large-v3": "Faster Distil Whisper Large v3",
}


_CACHE_TTL_SECONDS = 60.0

_download_size_cache: dict[str, tuple[float, tuple[int, int]]] = {}
_download_size_lock = RLock()

_list_installed_cache: dict[str, tuple[float, List[Dict[str, str]]]] = {}
_list_installed_lock = RLock()


def _normalize_cache_dir(cache_dir: str | Path) -> Path:
    """Return a normalized ``Path`` instance for cache directory comparisons."""

    if isinstance(cache_dir, Path):
        return cache_dir.expanduser()
    return Path(cache_dir).expanduser()


def _invalidate_list_installed_cache(cache_dir: str | Path | None = None) -> None:
    """Invalidate cached results for :func:`list_installed`."""

    with _list_installed_lock:
        if cache_dir is None:
            _list_installed_cache.clear()
            return
        cache_key = str(_normalize_cache_dir(cache_dir))
        _list_installed_cache.pop(cache_key, None)


def list_catalog() -> List[Dict[str, str]]:
    """Return curated catalog entries with display names."""
    return [
        {**entry, "display_name": DISPLAY_NAMES.get(entry["id"], entry["id"])}
        for entry in CURATED
    ]


def list_installed(cache_dir: str | Path) -> List[Dict[str, str]]:
    """Discover curated models available on disk and in the shared HF cache.

    Only models listed in :data:`CURATED` and containing essential files
    (``config.json`` together with ``model.bin`` or ``model.onnx``) are
    returned. Any other directories or isolated files found in ``cache_dir``
    are ignored. The shared Hugging Face cache is queried as a fallback.
    """

    normalized_dir = _normalize_cache_dir(cache_dir)
    cache_key = str(normalized_dir)
    now = time.monotonic()

    with _list_installed_lock:
        cached_entry = _list_installed_cache.get(cache_key)
        if cached_entry:
            cached_at, cached_value = cached_entry
            if now - cached_at < _CACHE_TTL_SECONDS:
                return copy.deepcopy(cached_value)
            _list_installed_cache.pop(cache_key, None)

    curated_ids = {c["id"] for c in CURATED}
    installed: List[Dict[str, str]] = []
    seen = set()

    cache_dir = Path(cache_dir)
    MODEL_LOGGER.debug("Listing curated models installed under %s", cache_dir)
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

    with _list_installed_lock:
        _list_installed_cache[cache_key] = (time.monotonic(), copy.deepcopy(installed))

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

    now = time.monotonic()
    with _download_size_lock:
        cached_entry = _download_size_cache.get(model_id)
        if cached_entry:
            cached_at, cached_value = cached_entry
            if now - cached_at < _CACHE_TTL_SECONDS:
                return cached_value
            _download_size_cache.pop(model_id, None)

    api = HfApi()
    info = api.model_info(model_id)
    total = 0
    files = 0
    for sibling in getattr(info, "siblings", []):
        total += sibling.size or 0
        files += 1
    MODEL_LOGGER.debug(
        "Computed download size for model %s: %.2f GB across %s files",
        model_id,
        total / (1024 ** 3) if total else 0.0,
        files,
    )
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
    *,
    timeout: float | int | None = None,
    cancel_event: Event | None = None,
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
    timeout: float | int | None, optional
        Maximum number of seconds to wait before aborting the download. ``None`` disables the timeout.
    cancel_event: Event | None, optional
        When provided, the download is aborted if the event is set.
    """

    cache_dir = Path(cache_dir)
    local_dir = cache_dir / backend / model_id
    if local_dir.is_dir() and any(local_dir.iterdir()):
        MODEL_LOGGER.info(
            "[METRIC] stage=model_download status=skip model=%s backend=%s path=%s",
            model_id,
            backend,
            local_dir,
        )
        return str(local_dir)

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    MODEL_LOGGER.info(
        "Starting model download: model=%s backend=%s quant=%s target=%s",
        model_id,
        backend,
        quant or "default",
        local_dir,
    )

    timeout_value: float | None = None
    deadline: float | None = None
    if timeout is not None:
        try:
            candidate = float(timeout)
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate > 0:
            timeout_value = candidate
            deadline = time.monotonic() + candidate

    def _check_abort() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelledError("Model download cancelled by caller.", by_user=True)
        if deadline is not None and time.monotonic() >= deadline:
            assert timeout_value is not None
            raise DownloadCancelledError(
                f"Model download timed out after {timeout_value:.0f} seconds.", timed_out=True
            )

    def _cleanup_partial() -> None:
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir)
        except Exception:  # pragma: no cover - best effort cleanup
            logging.debug("Failed to clean up partial download at %s", local_dir, exc_info=True)

    _check_abort()

    progress_class = None
    if cancel_event is not None or deadline is not None:
        progress_class = _make_cancellable_progress(_check_abort)

    try:
        if backend in {"transformers", "ct2"}:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                allow_patterns=None,
                tqdm_class=progress_class,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
        _check_abort()
    except DownloadCancelledError:
        _cleanup_partial()
        raise
    except KeyboardInterrupt as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        MODEL_LOGGER.info(
            "[METRIC] stage=model_download status=cancelled model=%s backend=%s duration_ms=%.2f",
            model_id,
            backend,
            duration_ms,
        )
        raise DownloadCancelledError("Model download cancelled by user.") from exc
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        MODEL_LOGGER.exception(
            "Model download failed: model=%s backend=%s target=%s",
            model_id,
            backend,
            local_dir,
        )
        MODEL_LOGGER.info(
            "[METRIC] stage=model_download status=error model=%s backend=%s duration_ms=%.2f",
            model_id,
            backend,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    MODEL_LOGGER.info(
        "[METRIC] stage=model_download status=success model=%s backend=%s duration_ms=%.2f path=%s",
        model_id,
        backend,
        duration_ms,
        local_dir,
    )
    return str(local_dir)


def _make_cancellable_progress(check_abort):
    from tqdm.auto import tqdm

    class _Progress(tqdm):
        def update(self, n=1):
            check_abort()
            result = super().update(n)
            check_abort()
            return result

        def refresh(self, *args, **kwargs):
            check_abort()
            return super().refresh(*args, **kwargs)

    return _Progress
