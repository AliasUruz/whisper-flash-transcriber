import json
import logging
from pathlib import Path
from typing import Dict, Any

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - huggingface_hub is a transformers dependency
    snapshot_download = None  # type: ignore

# Base directory for storing ASR models
MODEL_DIR = Path.home() / ".cache" / "whisper_flash_transcriber" / "asr"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Curated models with metadata; repo indicates HF repo id
CURATED: Dict[str, Dict[str, Any]] = {
    "distil-whisper/large-v2": {
        "backend": "Transformers",
        "dtype": "fp16",
        "chunk": 30,
        "batch": 16,
        "device": "auto",
        "repo": "distil-whisper/distil-large-v2",
    },
    "Systran/faster-whisper-large-v3": {
        "backend": "Faster-Whisper",
        "dtype": "fp16",
        "chunk": 30,
        "batch": 16,
        "device": "auto",
        "repo": "Systran/faster-whisper-large-v3",
    },
}


def asr_installed_models() -> Dict[str, Dict[str, Any]]:
    """Return installed ASR models with metadata."""
    models: Dict[str, Dict[str, Any]] = {}
    for path in MODEL_DIR.glob("*"):
        if not path.is_dir():
            continue
        meta_file = path / "metadata.json"
        try:
            info = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        except Exception:
            info = {}
        models[path.name.replace("__", "/")] = info
    return models


def ensure_download(backend: str, model: str) -> Path:
    """Ensure model is downloaded; returns local path."""
    info = CURATED.get(model, {})
    repo = info.get("repo", model)
    local_dir = MODEL_DIR / model.replace("/", "__")
    if local_dir.exists():
        logging.info("Model already available: %s", local_dir)
        return local_dir
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub not available for downloading models")
    try:
        snapshot_download(repo_id=repo, local_dir=local_dir)
        meta = {"backend": backend, **{k: v for k, v in info.items() if k != "repo"}}
        (local_dir / "metadata.json").write_text(json.dumps(meta))
        logging.info("Model '%s' downloaded to %s", repo, local_dir)
        return local_dir
    except Exception as e:  # pragma: no cover - network/IO errors
        logging.error("Failed to download model '%s': %s", repo, e)
        raise
