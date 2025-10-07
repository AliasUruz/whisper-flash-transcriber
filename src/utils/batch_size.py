from __future__ import annotations

import logging


_COMPUTE_TYPE_BASELINES: dict[str, int] = {
    "default": 16,
    "int8": 24,
    "int8_float16": 24,
    "int8_float32": 16,
    "float16": 16,
    "bfloat16": 12,
    "float32": 8,
}


def _normalize_compute_type(value: str | None) -> str:
    if not value:
        return "default"
    lowered = value.strip().lower()
    if lowered in {"fp16", "half"}:
        return "float16"
    if lowered in {"fp32"}:
        return "float32"
    return lowered


def select_batch_size(
    ct2_compute_type: str | None,
    fallback: int = 4,
    *,
    chunk_length_sec: float | None = None,
    max_batch_size: int | None = None,
) -> int:
    """Sugere um ``batch_size`` seguro sem depender de medições de VRAM.

    A heurística considera apenas parâmetros expostos pelo runtime CTranslate2,
    em especial o ``ct2_compute_type`` configurado e, opcionalmente, o tamanho
    alvo de ``chunk_length_sec``. O objetivo é fornecer um valor coerente mesmo
    em ambientes sem PyTorch instalado.
    """

    normalized_type = _normalize_compute_type(ct2_compute_type)
    baseline = _COMPUTE_TYPE_BASELINES.get(normalized_type, max(fallback, 1))

    value = baseline

    if chunk_length_sec is not None:
        try:
            chunk = float(chunk_length_sec)
        except (TypeError, ValueError):
            chunk = None
        if chunk is not None and chunk > 0:
            if chunk >= 75:
                value = max(1, int(value * 0.4))
            elif chunk >= 60:
                value = max(1, int(value * 0.5))
            elif chunk >= 45:
                value = max(1, int(value * 0.66))
            elif chunk >= 30:
                value = max(1, int(value * 0.8))

    if max_batch_size is not None:
        try:
            limit = int(max_batch_size)
        except (TypeError, ValueError):
            limit = None
        if limit is not None and limit > 0:
            value = min(value, limit)

    value = max(1, int(value))
    if isinstance(chunk_length_sec, (int, float)):
        chunk_repr = f"{float(chunk_length_sec):.1f}"
    else:
        chunk_repr = "unknown"

    logging.debug(
        "Batch size suggestion: compute_type=%s chunk=%s fallback=%s -> %s",
        normalized_type,
        chunk_repr,
        fallback,
        value,
    )
    return value
