"""Utilities for curated ASR model management."""

from __future__ import annotations

import copy
import inspect
import json
import logging
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from threading import Event, RLock
from typing import Callable, Dict, List, NamedTuple, Optional

from huggingface_hub import HfApi, scan_cache_dir, snapshot_download

from .logging_utils import get_logger, log_context


MODEL_LOGGER = get_logger("whisper_recorder.model", component="ModelManager")


_CT2_KNOWN_QUANTIZATIONS: set[str] = {
    "default",
    "int8",
    "int8_float16",
    "int8_float32",
    "int8_bfloat16",
    "int16",
    "float16",
    "float32",
}

_CT2_QUANTIZATION_ALIASES: dict[str, str] = {
    "auto": "default",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "full": "float32",
    "int8float16": "int8_float16",
    "int8_float_16": "int8_float16",
    "int8float32": "int8_float32",
    "int8_float_32": "int8_float32",
    "int8bf16": "int8_bfloat16",
    "int8_bf16": "int8_bfloat16",
}


@dataclass(frozen=True)
class ModelDownloadResult:
    """Structured result for :func:`ensure_download`."""

    path: str
    downloaded: bool
    bytes_downloaded: Optional[int] = None
    duration_seconds: Optional[float] = None
    target_dir: Optional[str] = None


class DownloadCancelledError(Exception):
    """Raised when a model download is cancelled or aborted."""

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


class InsufficientSpaceError(RuntimeError):
    """Raised when there is not enough free space to finish a download."""

    def __init__(self, message: str, *, required_bytes: int, free_bytes: int) -> None:
        super().__init__(message)
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)


# Curated catalog of officially supported ASR models.
# Each entry maps a Hugging Face model id to the backend that powers it.
CURATED: List[Dict[str, str]] = [
    {"id": "openai/whisper-large-v3-turbo", "backend": "ctranslate2"},
]

DISPLAY_NAMES: Dict[str, str] = {
    "openai/whisper-large-v3-turbo": "Whisper Large v3 Turbo",
}

# Para reintroduzir outros modelos futuramente, basta estender as estruturas
# CURATED e DISPLAY_NAMES abaixo.


_INSTALL_METADATA_FILENAME = "install.json"


def _write_install_metadata(
    target_dir: Path,
    *,
    model_id: str,
    backend_label: str,
    quant_label: str,
) -> None:
    """Persist lightweight metadata alongside the installed model.

    The metadata is used as a quick sanity check for the installation and to
    enrich telemetry/log statements. Failures while writing metadata are
    intentionally ignored by callers.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / _INSTALL_METADATA_FILENAME
    payload = {
        "model_id": model_id,
        "backend": backend_label,
        "quant": quant_label,
        "timestamp": time.time(),
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_catalog_entry(model_id: str) -> dict[str, str] | None:
    """Return the curated entry for ``model_id`` when available."""

    for entry in CURATED:
        if entry.get("id") == model_id:
            return dict(entry)
    return None


def _model_relative_path(model_id: str) -> Path:
    """Return the canonical relative path used to store ``model_id`` locally."""

    candidate = PurePosixPath(str(model_id or "").strip())
    if candidate.is_absolute():
        raise ValueError(f"Model identifier '{model_id}' cannot be absolute.")

    parts: list[str] = []
    for part in candidate.parts:
        normalized = part.strip()
        if not normalized or normalized in {".", ".."}:
            raise ValueError(
                f"Model identifier '{model_id}' contains unsafe path segment '{part}'."
            )
        if any(sep in normalized for sep in ("\\", ":")):
            raise ValueError(
                f"Model identifier '{model_id}' contains invalid character in segment '{part}'."
            )
        parts.append(normalized)

    if not parts:
        raise ValueError("Model identifier cannot be empty.")

    return Path(*parts)


def get_installation_dir(
    cache_dir: str | Path,
    backend: str | None,
    model_id: str,
) -> Path:
    """Return the canonical directory where ``model_id`` should live locally."""

    normalized_cache = _normalize_cache_dir(cache_dir)
    relative_model_path = _model_relative_path(model_id)
    storage_backend = backend_storage_name(backend)
    return normalized_cache / storage_backend / relative_model_path


def find_existing_installation(
    cache_dir: str | Path | None,
    backend: str | None,
    model_id: str,
) -> Path | None:
    """Return the first complete installation of ``model_id`` found on disk."""

    if not cache_dir:
        return None

    normalized_cache = _normalize_cache_dir(cache_dir)
    relative_model_path = _model_relative_path(model_id)
    candidate_names = backend_storage_candidates(backend)
    seen: set[Path] = set()

    for name in candidate_names:
        candidate_dir = normalized_cache / name / relative_model_path
        if candidate_dir in seen:
            continue
        seen.add(candidate_dir)
        try:
            if _is_installation_complete(candidate_dir):
                return candidate_dir
        except Exception:  # pragma: no cover - best effort probing
            MODEL_LOGGER.debug(
                "Unable to probe installation directory %s", candidate_dir, exc_info=True
            )
            continue

    return None


def normalize_backend_label(backend: str | None) -> str:
    """Return a normalized backend label for UI/configuration."""
    if not backend:
        return ""
    normalized = backend.strip().lower()
    if normalized in {"ct2", "ctranslate2"}:
        return "ctranslate2"
    if normalized in {"faster whisper", "faster_whisper"}:
        return "faster-whisper"
    return normalized


def backend_storage_name(backend: str | None) -> str:
    """Map backend label to the canonical directory name used on disk."""

    normalized = normalize_backend_label(backend)
    if not normalized:
        return "ct2"
    if normalized == "ctranslate2":
        return "ct2"
    if normalized == "faster-whisper":
        return "faster-whisper"
    return normalized


def backend_storage_candidates(backend: str | None) -> list[str]:
    """Return ordered storage directory candidates for ``backend``."""

    normalized = normalize_backend_label(backend)
    primary = backend_storage_name(normalized)

    candidates: list[str] = []
    if primary:
        candidates.append(primary)

    legacy_map = {
        "ctranslate2": ["ctranslate2"],
        "faster-whisper": ["ct2", "ctranslate2"],
        "transformers": ["transformers"],
    }

    for legacy_name in legacy_map.get(normalized, []):
        if legacy_name and legacy_name not in candidates:
            candidates.append(legacy_name)

    if normalized and normalized not in candidates:
        candidates.append(normalized)

    return candidates


def _normalize_quant_label(label: str | None) -> str:
    """Normalize the quantization branch label used for CT2 downloads."""

    if label is None:
        return "default"

    normalized = str(label).strip()
    if not normalized or normalized.lower() in {"default", "none"}:
        return "default"

    return normalized


def get_curated_entry(model_id: str | None) -> Dict[str, str] | None:
    """Return the curated catalog entry for ``model_id`` if available."""

    if not model_id:
        return None

    for entry in CURATED:
        if entry.get("id") == model_id:
            normalized = dict(entry)
            normalized["backend"] = normalize_backend_label(entry.get("backend"))
            return normalized
    return None


@lru_cache(maxsize=128)
def _ct2_quant_revision_exists(model_id: str, revision: str) -> bool:
    """Return ``True`` if the given CTranslate2 revision exists for ``model_id``."""

    api = HfApi()
    try:
        api.model_info(model_id, revision=revision)
    except Exception as exc:  # pragma: no cover - defensive network handling
        MODEL_LOGGER.debug(
            "Failed to resolve revision '%s' for model %s: %s",
            revision,
            model_id,
            exc,
            exc_info=True,
        )
        return False
    return True


def _normalize_ct2_quant_label(raw_quant: str | None) -> tuple[str, str | None]:
    """Normalize user-provided CT2 quantization label."""

    if raw_quant is None:
        return "default", None

    raw_value = str(raw_quant).strip()
    if not raw_value:
        return "default", None

    normalized_key = raw_value.lower().replace("-", "_")
    normalized = _CT2_QUANTIZATION_ALIASES.get(normalized_key, normalized_key)
    if normalized not in _CT2_KNOWN_QUANTIZATIONS:
        return "default", raw_value
    return normalized, None


def _resolve_ct2_quantization(
    model_id: str, raw_quant: str | None
) -> tuple[str, str | None]:
    """Return sanitized CT2 quantization label and revision to request."""

    quant_label, rejected = _normalize_ct2_quant_label(raw_quant)
    if rejected is not None:
        MODEL_LOGGER.warning(
            "Unsupported CTranslate2 quantization '%s'; falling back to default weights.",
            rejected,
        )

    if quant_label != "default":
        if _ct2_quant_revision_exists(model_id, quant_label):
            return quant_label, quant_label

        MODEL_LOGGER.warning(
            "Requested CTranslate2 quantization '%s' is not available for model %s; using default branch instead.",
            quant_label,
            model_id,
        )
        quant_label = "default"

    return quant_label, None


_CACHE_TTL_SECONDS = 60.0

_download_size_cache: dict[str, tuple[float, tuple[int, int]]] = {}
_download_size_lock = RLock()

_list_installed_cache: dict[str, tuple[float, List[Dict[str, str]]]] = {}
_list_installed_lock = RLock()



class _PreparedInstallation(NamedTuple):
    local_dir: Path
    ready_path: Path | None
    stale_local_dir: bool


_MODEL_WEIGHT_FILE_HINTS = {
    "model.bin",
    "model.onnx",
    "model.safetensors",
}

_SNAPSHOT_SUPPORT_CACHE: dict[str, bool] = {}
_SNAPSHOT_SUPPORT_LOCK = RLock()


def _set_snapshot_kwarg(target: dict, name: str, value) -> None:
    """Set ``name`` in ``target`` only when ``snapshot_download`` supports it."""

    if _snapshot_download_supports(name):
        target[name] = value
    else:
        MODEL_LOGGER.debug(
            "snapshot_download does not support parameter '%s'; skipping.",
            name,
        )


def _format_bytes(value: int) -> str:
    """Return a human-friendly string representation for ``value`` bytes."""

    units = ["B", "KB", "MB", "GB", "TB"]
    amount = float(max(0, int(value)))
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} PB"


@lru_cache(maxsize=1)
def _snapshot_download_signature() -> inspect.Signature | None:
    """Return the resolved signature for :func:`snapshot_download`."""

    func = snapshot_download
    seen = set()
    while hasattr(func, "__wrapped__"):
        wrapped = getattr(func, "__wrapped__", None)
        if wrapped is None or wrapped in seen:
            break
        seen.add(func)
        func = wrapped

    try:
        return inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None

@lru_cache(maxsize=None)
def _snapshot_download_supports(parameter: str) -> bool:
    """Return ``True`` when ``snapshot_download`` accepts ``parameter``."""

    if not parameter:
        return False

    normalized = str(parameter)

    with _SNAPSHOT_SUPPORT_LOCK:
        cached = _SNAPSHOT_SUPPORT_CACHE.get(normalized)
    if cached is not None:
        return cached

    signature = _snapshot_download_signature()
    if signature is None:
        supported = False
    elif normalized in signature.parameters:
        supported = True
    else:
        supported = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    with _SNAPSHOT_SUPPORT_LOCK:
        _SNAPSHOT_SUPPORT_CACHE[normalized] = supported

    return supported


def _model_dir_is_complete(path: Path) -> bool:
    """Return ``True`` when ``path`` contains the expected model assets."""

    if not path.exists() or not path.is_dir():
        return False

    has_config = False
    has_weights = False

    try:
        iterator = path.rglob("*")
    except Exception:  # pragma: no cover - defensive best effort
        return False

    for candidate in iterator:
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        if name == "config.json":
            has_config = True
            continue
        if name in _MODEL_WEIGHT_FILE_HINTS or (
            name.endswith((".bin", ".onnx", ".safetensors")) and "model" in name
        ):
            has_weights = True

        if has_config and has_weights:
            return True

    return has_config and has_weights


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


def _is_installation_complete(model_dir: Path) -> bool:
    """Return ``True`` when the on-disk model contains essential assets."""

    if not model_dir.exists():
        return False

    has_config = False
    has_weights = False

    try:
        iterator = model_dir.rglob("*")
    except Exception:
        return False

    for path in iterator:
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "config.json":
            has_config = True
        if name.endswith((".bin", ".onnx", ".safetensors")):
            has_weights = True
        if has_config and has_weights:
            return True
    return False


def _prepare_local_installation(
    cache_dir: Path,
    backend_label: str,
    model_id: str,
) -> _PreparedInstallation:
    storage_backend = backend_storage_name(backend_label)
    candidate_names = backend_storage_candidates(backend_label)
    if storage_backend not in candidate_names:
        candidate_names.insert(0, storage_backend)

    local_dir = cache_dir / storage_backend / model_id
    complete_dir: Path | None = None
    complete_source: str | None = None
    stale_dirs: list[Path] = []

    for candidate_name in candidate_names:
        candidate_dir = cache_dir / candidate_name / model_id
        if not candidate_dir.exists() or not candidate_dir.is_dir():
            continue
        try:
            has_entries = any(candidate_dir.iterdir())
        except Exception:  # pragma: no cover - best effort probing
            has_entries = False
        if not has_entries:
            stale_dirs.append(candidate_dir)
            continue
        if _is_installation_complete(candidate_dir):
            complete_dir = candidate_dir
            complete_source = candidate_name
            break
        stale_dirs.append(candidate_dir)

    ready_path: Path | None = None
    if complete_dir is not None:
        destination = local_dir
        if complete_source != storage_backend and destination != complete_dir:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
            except Exception:  # pragma: no cover - directory creation best effort
                pass
            if destination.exists() and destination != complete_dir:
                try:
                    shutil.rmtree(destination)
                except Exception:  # pragma: no cover - defensive cleanup
                    MODEL_LOGGER.debug(
                        "Failed to remove conflicting destination directory %s before relocation.",
                        destination,
                        exc_info=True,
                    )
            try:
                shutil.move(str(complete_dir), str(destination))
            except Exception:
                MODEL_LOGGER.exception(
                    "Failed to relocate legacy model directory from %s to %s.",
                    complete_dir,
                    destination,
                )
                ready_path = complete_dir
            else:
                MODEL_LOGGER.info(
                    "Relocated ASR model directory for backend %s: %s -> %s.",
                    backend_label,
                    complete_dir,
                    destination,
                )
                ready_path = destination
                _invalidate_list_installed_cache(cache_dir)
        else:
            ready_path = complete_dir

    stale_local_dir = False
    removed_any = False
    for stale_dir in stale_dirs:
        if ready_path is not None and stale_dir == ready_path:
            continue
        if stale_dir == local_dir:
            stale_local_dir = True
        if not stale_dir.exists():
            continue
        try:
            shutil.rmtree(stale_dir)
        except Exception:  # pragma: no cover - best effort cleanup
            MODEL_LOGGER.debug(
                "Failed to remove incomplete model directory %s before re-download.",
                stale_dir,
                exc_info=True,
            )
        else:
            removed_any = True
            MODEL_LOGGER.info(
                "Removed incomplete model directory at %s before retrying download.",
                stale_dir,
            )

    if removed_any:
        _invalidate_list_installed_cache(cache_dir)

    if stale_local_dir:
        MODEL_LOGGER.warning(
            "Detected incomplete model directory at %s; removing before re-downloading.",
            local_dir,
        )

    return _PreparedInstallation(
        local_dir=local_dir,
        ready_path=ready_path,
        stale_local_dir=stale_local_dir,
    )


def get_installation_dir(
    cache_dir: str | Path,
    backend: str | None,
    model_id: str,
) -> Path:
    """Return the canonical on-disk directory for ``model_id``.

    This helper mirrors the logic used when preparing local installations and
    is relied upon by the faster-whisper/ctranslate2 backend loader to locate
    previously downloaded models.  Prior to this implementation the
    application attempted to call ``model_manager.get_installation_dir`` but
    the function was never defined, resulting in an ``AttributeError`` during
    backend initialization and preventing the faster-whisper and CTranslate2
    integrations from bootstrapping correctly.

    Parameters
    ----------
    cache_dir:
        Base directory where models are stored. It is typically configured via
        the application settings.
    backend:
        Backend identifier (``"ctranslate2"``, ``"faster-whisper"``, etc.).
        When ``None`` the curated/default backend is used.
    model_id:
        Hugging Face model identifier.

    Returns
    -------
    Path
        Path pointing to the canonical installation directory for the given
        model/backend pair.
    """

    normalized_dir = _normalize_cache_dir(cache_dir)
    storage_backend = backend_storage_name(backend)
    relative_path = _model_relative_path(model_id)
    return normalized_dir / storage_backend / relative_path


def find_existing_installation(
    cache_dir: str | Path | None,
    backend: str | None,
    model_id: str,
) -> str | None:
    """Return the path to an existing local installation when available."""

    if not cache_dir:
        return None

    try:
        relative_path = _model_relative_path(model_id)
    except ValueError:
        return None

    cache_path = _normalize_cache_dir(cache_dir)
    storage_backend = backend_storage_name(backend)
    candidate_names = backend_storage_candidates(backend)
    if storage_backend not in candidate_names:
        candidate_names.insert(0, storage_backend)

    for candidate_name in candidate_names:
        candidate_dir = cache_path / candidate_name / relative_path
        if _is_installation_complete(candidate_dir):
            return str(candidate_dir)
    return None


def ensure_local_installation(
    cache_dir: str | Path,
    backend: str | None,
    model_id: str,
) -> Path | None:
    cache_path = Path(cache_dir)

    curated_entry = get_curated_entry(model_id)
    curated_backend = (
        normalize_backend_label(curated_entry.get("backend")) if curated_entry else ""
    )

    requested_backend = normalize_backend_label(backend)
    backend_label = requested_backend or curated_backend or "ctranslate2"

    if curated_backend and backend_label != curated_backend:
        MODEL_LOGGER.warning(
            "Overriding backend '%s' with curated backend '%s' for model '%s'.",
            backend_label,
            curated_backend,
            model_id,
        )
        backend_label = curated_backend

    prepared = _prepare_local_installation(cache_path, backend_label, model_id)
    return prepared.ready_path


def list_catalog() -> List[Dict[str, str]]:
    """Return curated catalog entries with display names."""
    catalog = []
    for entry in CURATED:
        normalized = {**entry}
        normalized["backend"] = normalize_backend_label(entry.get("backend"))
        normalized["display_name"] = DISPLAY_NAMES.get(entry["id"], entry["id"])
        catalog.append(normalized)
    return catalog


def list_installed(cache_dir: str | Path) -> List[Dict[str, str]]:
    """Discover curated models available on disk and in the shared HF cache.

    Only models listed in :data:`CURATED` and containing essential files
    (``config.json`` together with at least one weight artifact such as
    ``model.bin``, ``model.onnx`` or ``model.safetensors``) are returned.
    Any other directories or isolated files found in ``cache_dir`` are
    ignored. The shared Hugging Face cache is queried as a fallback.
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

    curated_entries = {
        entry["id"]: normalize_backend_label(entry.get("backend"))
        for entry in CURATED
    }
    installed: List[Dict[str, str]] = []
    seen: set[str] = set()

    MODEL_LOGGER.debug("Listing curated models installed under %s", normalized_dir)
    for model_id, backend_label in curated_entries.items():
        try:
            relative_path = _model_relative_path(model_id)
        except ValueError as exc:
            MODEL_LOGGER.warning(
                "Skipping curated model %s due to invalid identifier: %s",
                model_id,
                exc,
            )
            continue

        storage_backend = backend_storage_name(backend_label)
        candidate_dir = normalized_dir / storage_backend / relative_path
        if candidate_dir.is_dir():
            if _model_dir_is_complete(candidate_dir):
                installed.append(
                    {
                        "id": model_id,
                        "backend": backend_label,
                        "path": str(candidate_dir),
                    }
                )
                seen.add(model_id)
            else:
                MODEL_LOGGER.warning(
                    "Model %s found at %s but installation is incomplete; ignoring.",
                    model_id,
                    candidate_dir,
                )

    # Detect curated models placed in an unexpected backend directory to surface
    # configuration issues while avoiding duplicate/invalid entries.
    try:
        for backend_dir in normalized_dir.iterdir():
            if not backend_dir.is_dir():
                continue
            backend_label = normalize_backend_label(backend_dir.name)
            for model_id, curated_backend in curated_entries.items():
                if model_id in seen:
                    continue
                try:
                    relative_path = _model_relative_path(model_id)
                except ValueError:
                    continue
                stray_dir = backend_dir / relative_path
                if stray_dir.is_dir() and _model_dir_is_complete(stray_dir):
                    MODEL_LOGGER.warning(
                        "Model %s is installed under backend directory '%s', "
                        "but curated backend is '%s'. The installation will be "
                        "ignored to avoid inconsistent state.",
                        model_id,
                        backend_label or backend_dir.name,
                        curated_backend,
                    )
    except FileNotFoundError:
        pass

    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id in seen or repo.repo_id not in curated_entries:
                continue
            repo_path = Path(repo.repo_path)
            if not _model_dir_is_complete(repo_path):
                continue
            installed.append(
                {
                    "id": repo.repo_id,
                    "backend": "transformers",
                    "path": str(repo_path),
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
    siblings = list(getattr(info, "siblings", []) or [])

    def _normalize_size(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    total = 0
    sized_files = 0
    for sibling in siblings:
        size_value = getattr(sibling, "size", None)
        if size_value is None:
            lfs_meta = getattr(sibling, "lfs", None)
            if hasattr(lfs_meta, "size"):
                size_value = getattr(lfs_meta, "size")
            elif isinstance(lfs_meta, dict):
                size_value = lfs_meta.get("size")
        normalized = _normalize_size(size_value)
        if normalized is None and size_value is not None:
            MODEL_LOGGER.debug(
                "Ignoring non-numeric size for %s/%s: %r",
                model_id,
                getattr(sibling, "rfilename", ""),
                size_value,
            )
        if normalized is None:
            continue
        total += normalized
        sized_files += 1

    total_files = len(siblings) if siblings else sized_files

    if total <= 0 or sized_files == 0:
        try:
            tree_items = api.list_repo_tree(
                repo_id=model_id,
                repo_type="model",
                recursive=True,
            )
        except Exception:  # pragma: no cover - best effort fallback
            MODEL_LOGGER.debug(
                "Failed to retrieve repo tree for %s when computing download size.",
                model_id,
                exc_info=True,
            )
        else:
            fallback_total = 0
            fallback_files = 0
            for item in tree_items:
                item_type = getattr(item, "type", None)
                if item_type not in (None, "file", "blob"):
                    continue
                normalized = _normalize_size(getattr(item, "size", None))
                if normalized is None:
                    raw_size = getattr(item, "size", None)
                    if raw_size is not None:
                        MODEL_LOGGER.debug(
                            "Ignoring non-numeric tree size for %s/%s: %r",
                            model_id,
                            getattr(item, "path", getattr(item, "rfilename", "")),
                            raw_size,
                        )
                    continue
                fallback_total += normalized
                fallback_files += 1
            if fallback_total > 0:
                total = fallback_total
            if fallback_files:
                total_files = fallback_files
                sized_files = max(sized_files, fallback_files)

    if total_files == 0:
        total_files = sized_files

    MODEL_LOGGER.debug(
        "Computed download size for model %s: %.2f GB across %s files",
        model_id,
        total / (1024 ** 3) if total else 0.0,
        total_files,
    )

    with _download_size_lock:
        _download_size_cache[model_id] = (time.monotonic(), (total, total_files))

    return total, total_files


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
    hf_cache_dir: str | Path | None = None,
    timeout: float | int | None = None,
    cancel_event: Event | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_stage_change: Callable[[str, dict], None] | None = None,
) -> ModelDownloadResult:
    """Ensure that the given model is present locally.

    Parameters
    ----------
    model_id: str
        Full model identifier as in the curated catalog.
    backend: str
        Backend selected by the user (e.g., ``"ctranslate2"``, ``"faster-whisper"``, ``"transformers"``).
    cache_dir: str | Path
        Root directory where models are cached.
    quant: str | None
        Quantization branch for CT2 models. Ignored for Transformers.
    hf_cache_dir: str | Path | None, optional
        Custom Hugging Face cache directory used while resolving artifacts.
    timeout: float | int | None, optional
        Maximum number of seconds to wait before aborting the download. ``None`` disables the timeout.
    cancel_event: Event | None, optional
        When provided, the download is aborted if the event is set.
    on_progress: Callable[[int, int], None] | None, optional
        Observer invoked whenever progress metrics are updated. Receives the
        number of downloaded bytes and the total expected bytes.
    on_stage_change: Callable[[str, dict], None] | None, optional
        Observer invoked whenever the internal stage changes. Receives the
        stage identifier and a metadata dictionary derived from existing log
        metrics.

    Returns
    -------
    ModelDownloadResult
        Structured metadata about the local installation, including whether
        a fresh download was performed.
    """

    cache_dir = Path(cache_dir)

    def _emit_stage(stage_id: str, **metadata) -> None:
        if on_stage_change is None:
            return
        payload = {
            "model_id": model_id,
            "backend": backend,
            "quant": quant,
            "target_dir": str(cache_dir),
        }
        payload.update(metadata)
        try:
            on_stage_change(stage_id, payload)
        except Exception:  # pragma: no cover - observer callbacks
            MODEL_LOGGER.debug("Stage callback failed for %s", stage_id, exc_info=True)

    def _emit_progress(done: int, total: int) -> None:
        if on_progress is None:
            return
        try:
            on_progress(int(done), int(total))
        except Exception:  # pragma: no cover - observer callbacks
            MODEL_LOGGER.debug(
                "Progress callback failed (done=%s, total=%s)", done, total, exc_info=True
            )

    backend_label = normalize_backend_label(backend)
    curated_entry = _resolve_catalog_entry(model_id)
    curated_backend = normalize_backend_label(curated_entry.get("backend")) if curated_entry else ""
    if curated_backend and backend_label and backend_label != curated_backend:
        MODEL_LOGGER.warning(
            "Requested backend '%s' for model %s does not match curated backend '%s'; enforcing curated backend.",
            backend_label,
            model_id,
            curated_backend,
        )
    if curated_backend:
        backend_label = curated_backend
    storage_backend = backend_storage_name(backend_label or backend)
    backend_label = backend_label or normalize_backend_label(storage_backend) or storage_backend

    quant_label = _normalize_quant_label(quant if backend_label == "ctranslate2" else None)

    prepared = _prepare_local_installation(cache_dir, backend_label, model_id)
    local_dir = prepared.local_dir
    if prepared.ready_path is not None:
        ready_path = prepared.ready_path
        try:
            _write_install_metadata(
                ready_path,
                model_id=model_id,
                backend_label=backend_label,
                quant_label=quant_label,
            )
        except Exception:  # pragma: no cover - metadata persistence best effort
            MODEL_LOGGER.debug(
                "Unable to persist metadata for model %s at %s",
                model_id,
                ready_path,
                exc_info=True,
            )
        _emit_stage("already_present", path=str(ready_path))
        MODEL_LOGGER.info(
            log_context(
                "Model download skipped because artifacts already exist.",
                event="model.download_skipped",
                model=model_id,
                backend=backend_label,
                path=str(prepared.ready_path),
            )
        )
        _emit_progress(0, 0)
        return ModelDownloadResult(str(ready_path), downloaded=False, target_dir=str(ready_path))

    stale_local_dir = prepared.stale_local_dir

    local_dir.parent.mkdir(parents=True, exist_ok=True)

    estimated_bytes = 0
    estimated_files = 0
    _emit_stage("estimating_size")
    try:
        estimated_bytes, estimated_files = get_model_download_size(model_id)
    except Exception:  # pragma: no cover - metadata retrieval best effort
        MODEL_LOGGER.debug(
            "Unable to compute download size metadata for model %s.",
            model_id,
            exc_info=True,
        )
        _emit_stage("size_estimate_failed")
    else:
        if estimated_bytes > 0:
            try:
                usage = shutil.disk_usage(local_dir.parent)
            except FileNotFoundError:
                local_dir.parent.mkdir(parents=True, exist_ok=True)
                usage = shutil.disk_usage(local_dir.parent)
            free_bytes = usage.free
            safety_margin = max(int(estimated_bytes * 0.1), 256 * 1024 * 1024)
            required_bytes = estimated_bytes + safety_margin
            MODEL_LOGGER.info(
                log_context(
                    "Model download size estimated.",
                    event="model.download_size_estimate",
                    model=model_id,
                    estimated_bytes=estimated_bytes,
                    estimated_files=estimated_files,
                    free_bytes=free_bytes,
                )
            )
            _emit_stage(
                "size_estimated",
                estimated_bytes=estimated_bytes,
                estimated_files=estimated_files,
                free_bytes=free_bytes,
                required_bytes=required_bytes,
            )
            if free_bytes < required_bytes:
                MODEL_LOGGER.error(
                    "Insufficient free space for model %s: required %s (with safety margin) but only %s available.",
                    model_id,
                    _format_bytes(required_bytes),
                    _format_bytes(free_bytes),
                )
                raise InsufficientSpaceError(
                    (
                        "Insufficient free space to download model %s: "
                        "requires approximately %s (including safety margin) but only %s is available."
                    )
                    % (model_id, _format_bytes(required_bytes), _format_bytes(free_bytes)),
                    required_bytes=required_bytes,
                    free_bytes=free_bytes,
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
        if deadline is not None and time.monotonic() >= deadline:
            seconds = timeout_value if timeout_value is not None else 0.0
            raise DownloadCancelledError(
                f"Model download timed out after {seconds:.0f} seconds.",
                timed_out=True,
            )
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelledError("Model download cancelled by caller.", by_user=True)

    def _cleanup_partial(context: str | None = None) -> None:
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir)
                if context:
                    MODEL_LOGGER.info(
                        "Removed incomplete model directory at %s (%s).",
                        local_dir,
                        context,
                    )
                else:
                    MODEL_LOGGER.info("Removed incomplete model directory at %s.", local_dir)
        except Exception:  # pragma: no cover - best effort cleanup
            logging.debug("Failed to clean up partial download at %s", local_dir, exc_info=True)

    progress_class = None
    if cancel_event is not None or deadline is not None or on_progress is not None:
        progress_class = _make_cancellable_progress(_check_abort, _emit_progress)

    # Seleciona branch de quantização quando aplicável (modelos CT2).
    revision = None
    if storage_backend == "ct2" and quant_label != "default":
        revision = quant_label

    download_kwargs = {
        "repo_id": model_id,
        "local_dir": str(local_dir),
    }
    if hf_cache_dir:
        try:
            cache_path = Path(hf_cache_dir).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
        except Exception:
            MODEL_LOGGER.debug(
                "Failed to prepare Hugging Face cache directory '%s'.",
                hf_cache_dir,
                exc_info=True,
            )
        else:
            _set_snapshot_kwarg(download_kwargs, "cache_dir", str(cache_path))
    if progress_class is not None:
        _set_snapshot_kwarg(download_kwargs, "tqdm_class", progress_class)
    _set_snapshot_kwarg(download_kwargs, "resume_download", True)
    _set_snapshot_kwarg(download_kwargs, "local_dir_use_symlinks", False)
    _set_snapshot_kwarg(download_kwargs, "local_dir_use_hardlinks", False)
    if revision is not None:
        _set_snapshot_kwarg(download_kwargs, "revision", revision)

    if stale_local_dir:
        _cleanup_partial("stale_before_download")

    start_time = time.perf_counter()
    _emit_stage(
        "download_start",
        path=str(local_dir),
        estimated_bytes=estimated_bytes,
        estimated_files=estimated_files,
    )
    if estimated_bytes:
        _emit_progress(0, estimated_bytes)
    else:
        _emit_progress(0, 0)
    MODEL_LOGGER.info(
        "Starting model download: model=%s backend=%s quant=%s target=%s",
        model_id,
        backend_label,
        quant_label,
        local_dir,
    )

    try:
        _check_abort()
        if storage_backend == "transformers":
            snapshot_download(**download_kwargs)
        elif storage_backend in {"ct2", "faster-whisper"}:
            snapshot_download(**download_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend_label}")
        _check_abort()
    except DownloadCancelledError as cancel_exc:
        _cleanup_partial("cancelled" if not getattr(cancel_exc, "timed_out", False) else "timeout")
        _emit_stage(
            "cancelled",
            timed_out=getattr(cancel_exc, "timed_out", False),
            by_user=getattr(cancel_exc, "by_user", False),
            message=str(cancel_exc) or "",
        )
        raise
    except KeyboardInterrupt as exc:
        _cleanup_partial("keyboard_interrupt")
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        MODEL_LOGGER.info(
            "[METRIC] stage=model_download status=cancelled model=%s backend=%s duration_ms=%.2f",
            model_id,
            backend_label,
            duration_ms,
        )
        _cleanup_partial()
        _invalidate_list_installed_cache(cache_dir)
        _emit_stage("cancelled", by_user=True, message="Keyboard interrupt")
        raise DownloadCancelledError(
            "Model download cancelled by user.",
            by_user=True,
        ) from exc
    except Exception:
        _cleanup_partial("error")
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        MODEL_LOGGER.exception(
            "Model download failed: model=%s backend=%s target=%s",
            model_id,
            backend_label,
            local_dir,
        )
        MODEL_LOGGER.info(
            "[METRIC] stage=model_download status=error model=%s backend=%s duration_ms=%.2f",
            model_id,
            backend_label,
            duration_ms,
        )
        _cleanup_partial()
        _invalidate_list_installed_cache(cache_dir)
        _emit_stage("error", duration_ms=duration_ms, message="Download failed")
        raise

    if not _is_installation_complete(local_dir):
        _cleanup_partial("incomplete")
        _invalidate_list_installed_cache(cache_dir)
        _emit_stage("error", message="Installation incomplete")
        raise RuntimeError(
            "Model download completed but installation is missing essential files."
        )

    duration_seconds = time.perf_counter() - start_time
    try:
        _write_install_metadata(
            local_dir,
            model_id=model_id,
            backend_label=backend_label,
            quant_label=quant_label,
        )
    except Exception:  # pragma: no cover - metadata persistence best effort
        MODEL_LOGGER.debug(
            "Unable to persist metadata for model %s at %s", model_id, local_dir, exc_info=True
        )
    MODEL_LOGGER.info(
        "[METRIC] stage=model_download status=success model=%s backend=%s duration_ms=%.2f path=%s",
        model_id,
        backend_label,
        duration_seconds * 1000.0,
        local_dir,
    )
    effective_bytes, _ = get_installed_size(local_dir)
    if effective_bytes:
        _emit_progress(effective_bytes, effective_bytes)
    else:
        _emit_progress(0, 0)
    throughput = None
    if duration_seconds > 0 and effective_bytes:
        throughput = effective_bytes / duration_seconds
    _emit_stage(
        "success",
        duration_seconds=duration_seconds,
        bytes_downloaded=effective_bytes,
        throughput_bps=throughput,
        path=str(local_dir),
    )
    _invalidate_list_installed_cache(cache_dir)
    return ModelDownloadResult(
        str(local_dir),
        downloaded=True,
        bytes_downloaded=effective_bytes or None,
        duration_seconds=duration_seconds,
        target_dir=str(local_dir),
    )

def _make_cancellable_progress(check_abort, progress_callback: Callable[[int, int], None] | None = None):
    from tqdm.auto import tqdm

    class _Progress(tqdm):
        def update(self, n=1):
            check_abort()
            result = super().update(n)
            check_abort()
            if progress_callback is not None:
                try:
                    progress_callback(int(self.n), int(self.total or 0))
                except Exception:  # pragma: no cover - observer callbacks
                    MODEL_LOGGER.debug("Progress callback failed.", exc_info=True)
            return result

        def refresh(self, *args, **kwargs):
            check_abort()
            return super().refresh(*args, **kwargs)

    return _Progress
