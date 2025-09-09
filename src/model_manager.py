"""Gerenciador de modelos ASR.

Fornece catálogo curado, listagem de modelos instalados e
função utilitária para garantir o download de modelos nos
backends suportados (Transformers e CTranslate2).
"""

from __future__ import annotations

import os
from typing import Dict, List

from huggingface_hub import snapshot_download

# Catálogo curado de modelos suportados
# Cada entrada possui ID base e metadados dos backends
CURATED: List[Dict] = [
    {
        "id": "tiny",
        "description": "Modelo mais leve",
        "backends": {
            "transformers": {"repo_id": "openai/whisper-tiny"},
            "ct2": {"size": "tiny"},
        },
    },
    {
        "id": "base",
        "description": "Modelo base",
        "backends": {
            "transformers": {"repo_id": "openai/whisper-base"},
            "ct2": {"size": "base"},
        },
    },
    {
        "id": "small",
        "description": "Modelo pequeno",
        "backends": {
            "transformers": {"repo_id": "openai/whisper-small"},
            "ct2": {"size": "small"},
        },
    },
    {
        "id": "medium",
        "description": "Modelo médio",
        "backends": {
            "transformers": {"repo_id": "openai/whisper-medium"},
            "ct2": {"size": "medium"},
        },
    },
    {
        "id": "large-v2",
        "description": "Modelo grande v2",
        "backends": {
            "transformers": {"repo_id": "openai/whisper-large-v2"},
            "ct2": {"size": "large-v2"},
        },
    },
    {
        "id": "large-v3",
        "description": "Modelo grande v3",
        "backends": {
            "transformers": {"repo_id": "openai/whisper-large-v3"},
            "ct2": {"size": "large-v3"},
        },
    },
]


def list_catalog() -> List[Dict]:
    """Retorna o catálogo curado de modelos."""

    return CURATED


def list_installed(cache_dir: str) -> List[Dict]:
    """Lista modelos já baixados no ``cache_dir``.

    O diretório esperado possui subpastas por backend (``transformers`` e
    ``ct2``), cada qual contendo subpastas com o ``model_id``.
    """

    installed: List[Dict] = []
    for backend in ("transformers", "ct2"):
        backend_path = os.path.join(cache_dir, backend)
        if not os.path.isdir(backend_path):
            continue
        for model_id in os.listdir(backend_path):
            model_path = os.path.join(backend_path, model_id)
            if os.path.isdir(model_path):
                installed.append({
                    "id": model_id,
                    "backend": backend,
                    "path": model_path,
                })
    return installed


def ensure_download(
    model_id: str,
    backend: str,
    cache_dir: str,
    quant: str | None = None,
) -> str:
    """Garante que o modelo solicitado esteja disponível localmente.

    Parameters
    ----------
    model_id:
        Identificador do modelo no catálogo ``CURATED``.
    backend:
        ``"transformers"`` ou ``"ct2"``.
    cache_dir:
        Diretório raiz onde os modelos são armazenados.
    quant:
        Para o backend CT2, branch de quantização (``int8``, ``int8_float16``,
        ``float16`` etc.). Ignorado para Transformers.

    Returns
    -------
    str
        Caminho local onde o modelo está armazenado.
    """

    catalog_map = {entry["id"]: entry for entry in CURATED}
    if model_id not in catalog_map:
        raise ValueError(f"Modelo '{model_id}' não encontrado no catálogo curado.")

    entry = catalog_map[model_id]
    backend_info = entry["backends"].get(backend)
    if backend_info is None:
        raise ValueError(f"Backend '{backend}' não disponível para o modelo '{model_id}'.")

    local_dir = os.path.join(cache_dir, backend, model_id)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    if backend == "transformers":
        repo_id = backend_info["repo_id"]
        snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    elif backend == "ct2":
        try:
            from faster_whisper.utils import download_model
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("faster-whisper não está instalado") from exc
        download_model(
            backend_info["size"],
            output_dir=local_dir,
            cache_dir=cache_dir,
            revision=quant,
        )
    else:  # pragma: no cover - sanidade
        raise ValueError(f"Backend desconhecido: {backend}")

    return local_dir
