import os
import json
import logging
from typing import List, Optional

from huggingface_hub import snapshot_download

# Diretório padrão de cache para modelos ASR
DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "whisper_flash_transcriber"
)

# Nome do arquivo que persiste a lista de modelos instalados
INSTALLED_FILE = "installed.json"

# Marcadores usados para identificar modelos HuggingFace/CT2
HF_MARKERS = {"config.json", "model_index.json", "pytorch_model.bin", "model.safetensors"}
CT2_EXTENSIONS = (".bin", ".ct2", ".onnx")


def _is_model_dir(path: str) -> bool:
    """Heurística simples para identificar pastas de modelos HF ou CT2."""
    if not os.path.isdir(path):
        return False
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    if entries & HF_MARKERS:
        return True
    for entry in entries:
        if entry.endswith(CT2_EXTENSIONS):
            return True
    return False


def _persist_list(cache_dir: str, models: List[str]) -> None:
    installed_path = os.path.join(cache_dir, INSTALLED_FILE)
    try:
        with open(installed_path, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)
    except OSError as e:
        logging.warning("Falha ao salvar %s: %s", installed_path, e)


def scan_installed(cache_dir: str) -> List[str]:
    """Retorna a lista de modelos já baixados no ``cache_dir``.

    Se ``installed.json`` existir, ele será carregado. Caso contrário, o diretório
    será examinado procurando por artefatos típicos de modelos HuggingFace ou
    CTranslate2. A lista resultante é persistida em ``installed.json``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    installed_path = os.path.join(cache_dir, INSTALLED_FILE)

    # 1. Tentar carregar lista persistida
    try:
        with open(installed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass

    # 2. Escanear diretório
    models: List[str] = []
    for entry in os.listdir(cache_dir):
        path = os.path.join(cache_dir, entry)
        if _is_model_dir(path):
            models.append(entry)

    _persist_list(cache_dir, models)
    return models


def ensure_download(
    model_id: str,
    cache_dir: Optional[str] = None,
    config_manager: Optional["ConfigManager"] = None,
    **kwargs,
) -> str:
    """Garante que ``model_id`` esteja disponível localmente.

    Após cada download bem-sucedido, atualiza ``asr_installed_models`` no
    ``ConfigManager`` (se fornecido) e persiste ``installed.json``.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    local_dir = os.path.join(cache_dir, model_id)

    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            **kwargs,
        )

    installed = scan_installed(cache_dir)
    if config_manager is not None:
        try:
            config_manager.set_asr_installed_models(installed)
            config_manager.save_config()
        except Exception as e:
            logging.warning("Falha ao atualizar config após download: %s", e)

    return local_dir


# Evitar import circular para hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from .config_manager import ConfigManager
