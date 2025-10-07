from __future__ import annotations

from dataclasses import dataclass
import logging


__all__ = ["BatchSizeHint", "select_batch_size"]


_CT2_COMPUTE_TYPE_LIMITS: dict[str, int] = {
    # Values derived from empirical runs with the faster-whisper bindings.
    # They intentionally err on the side of caution so that "base" never
    # exceeds what a typical GPU with the corresponding precision can handle.
    "int8": 48,
    "int8_float16": 32,
    "int16": 24,
    "float16": 16,
    "bfloat16": 16,
    "float32": 8,
}


@dataclass(slots=True)
class BatchSizeHint:
    """Normalized batch-size inputs used by :func:`select_batch_size`.

    ``base`` mirrors the configuration knob exposed in the UI. The optional
    ``ct2_compute_type`` and ``chunk_length_sec`` parameters are collected from
    configuration as well and avoid the need for runtime device probing.
    """

    base: int
    ct2_compute_type: str | None = None
    chunk_length_sec: float | None = None


def _normalize_compute_type(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    return normalized or None


def _limit_for_compute_type(compute_type: str | None) -> int | None:
    if compute_type is None:
        return None
    return _CT2_COMPUTE_TYPE_LIMITS.get(compute_type)


def _sanitize_base(value: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        numeric = 1
    return max(1, numeric)


def select_batch_size(
    *,
    base: int,
    ct2_compute_type: str | None = None,
    chunk_length_sec: float | None = None,
) -> int:
    """Return a deterministic batch size for CTranslate2 execution.

    The function never inspects the CUDA runtime directly. Instead, it
    reconciles the user-provided ``base`` with conservative ceilings tied to
    the configured ``ct2_compute_type`` and optionally scales the result down
    for very long audio chunks. Short chunks simply stick to the configured
    value so that operators retain full control over throughput.
    """

    batch = _sanitize_base(base)
    compute_type = _normalize_compute_type(ct2_compute_type)
    limit = _limit_for_compute_type(compute_type)
    if limit is not None:
        batch = min(batch, limit)

    if chunk_length_sec is not None:
        try:
            chunk_value = float(chunk_length_sec)
        except (TypeError, ValueError):
            chunk_value = 0.0

        if chunk_value > 0:
            if chunk_value > 180:
                batch = max(1, batch // 4)
            elif chunk_value > 90:
                batch = max(1, batch // 3)
            elif chunk_value > 60:
                batch = max(1, batch // 2)

    logging.debug(
        "Resolved CTranslate2 batch size.",
        extra={
            "event": "batch_size",
            "base": base,
            "ct2_compute_type": compute_type,
            "chunk_length_sec": chunk_length_sec,
            "value": batch,
        },
    )
    return batch
