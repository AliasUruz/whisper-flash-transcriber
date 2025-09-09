import os
import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import model_manager
from src.config_manager import ConfigManager


def test_scan_installed_recovers_from_corrupted_json(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / model_manager.INSTALLED_FILE).write_text("{corrupted")
    model_dir = cache / "modelA"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    models = model_manager.scan_installed(str(cache))
    assert models == ["modelA"]
    data = json.loads((cache / model_manager.INSTALLED_FILE).read_text())
    assert "modelA" in data


def test_scan_installed_handles_permission_error(monkeypatch, tmp_path):
    def fake_listdir(path):
        raise PermissionError
    monkeypatch.setattr(os, "listdir", fake_listdir)
    models = model_manager.scan_installed(str(tmp_path))
    assert models == []


def test_ensure_download_updates_config(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cfg = ConfigManager(config_file=str(tmp_path / "cfg.json"))

    def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, force_download=False, **kwargs):
        os.makedirs(local_dir, exist_ok=True)
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            f.write("{}")

    monkeypatch.setattr(model_manager, "snapshot_download", fake_snapshot_download)

    class DummyInfo:
        sha = "123"

    monkeypatch.setattr(model_manager.HfApi, "model_info", lambda self, mid: DummyInfo())

    path = model_manager.ensure_download("dummy/model", cache_dir=str(cache), config_manager=cfg)
    assert os.path.isdir(path)
    assert "dummy/model" in cfg.get_asr_installed_models()
