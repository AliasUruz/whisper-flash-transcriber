import os
import json
import logging
from typing import List, Optional, Dict

from huggingface_hub import snapshot_download, HfApi

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


def _load_installed(cache_dir: str) -> Dict[str, Optional[str]]:
    installed_path = os.path.join(cache_dir, INSTALLED_FILE)
    try:
        with open(installed_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: (v if isinstance(v, str) or v is None else None) for k, v in data.items()}
        if isinstance(data, list):
            return {k: None for k in data}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _persist_list(cache_dir: str, models: Dict[str, Optional[str]]) -> None:
    installed_path = os.path.join(cache_dir, INSTALLED_FILE)
    try:
        with open(installed_path, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)
    except OSError as e:
        logging.warning("Falha ao salvar %s: %s", installed_path, e)


def _validate_download(local_dir: str) -> bool:
    try:
        entries = os.listdir(local_dir)
    except OSError as e:
        logging.error("Falha ao acessar %s: %s", local_dir, e)
        return False
    if not entries:
        logging.error("Download de modelo falhou: diretório vazio %s", local_dir)
        return False
    entries_set = set(entries)
    if entries_set & HF_MARKERS:
        return True
    for entry in entries_set:
        if entry.endswith(CT2_EXTENSIONS):
            return True
    logging.error("Artefatos de modelo ausentes em %s", local_dir)
    return False


def scan_installed(cache_dir: str) -> List[str]:
    """Retorna a lista de modelos já baixados no ``cache_dir``.

    Se ``installed.json`` existir, ele será carregado. Caso contrário, o diretório
    será examinado procurando por artefatos típicos de modelos HuggingFace ou
    CTranslate2. A lista resultante é persistida em ``installed.json``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    models_info = _load_installed(cache_dir)

    try:
        entries = os.listdir(cache_dir)
    except OSError:
        entries = []
    for entry in entries:
        path = os.path.join(cache_dir, entry)
        if _is_model_dir(path):
            models_info.setdefault(entry, models_info.get(entry))

    _persist_list(cache_dir, models_info)
    return list(models_info.keys())


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
    api = HfApi()
    force = kwargs.pop("force_download", False)
    if force or not os.path.isdir(local_dir) or not os.listdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            force_download=force,
            **kwargs,
        )

    if not _validate_download(local_dir):
        logging.error("Validação de download falhou para %s", model_id)

    revision = None
    try:
        revision = api.model_info(model_id).sha
    except Exception as e:
        logging.warning("Falha ao obter revisão de %s: %s", model_id, e)

    models_info = _load_installed(cache_dir)
    models_info[model_id] = revision
    _persist_list(cache_dir, models_info)

    installed = list(models_info.keys())
    if config_manager is not None:
        try:
            config_manager.set_asr_installed_models(installed)
            config_manager.save_config()
        except Exception as e:
            logging.warning("Falha ao atualizar config após download: %s", e)

    return local_dir


def check_updates(cache_dir: Optional[str] = None) -> Dict[str, str]:
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    api = HfApi()
    models_info = _load_installed(cache_dir)
    updates: Dict[str, str] = {}
    for model_id, local_rev in models_info.items():
        try:
            remote_rev = api.model_info(model_id).sha
            if local_rev and local_rev != remote_rev:
                updates[model_id] = remote_rev
        except Exception as e:
            logging.warning("Falha ao verificar atualização de %s: %s", model_id, e)
    return updates


def update_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    config_manager: Optional["ConfigManager"] = None,
    **kwargs,
) -> str:
    return ensure_download(
        model_id,
        cache_dir=cache_dir,
        config_manager=config_manager,
        force_download=True,
        **kwargs,
    )


# Evitar import circular para hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # pragma: no cover
    from .config_manager import ConfigManager
