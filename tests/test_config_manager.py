import os
import time
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src import config_manager


def test_skip_save_when_unchanged(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))
    cm = config_manager.ConfigManager(config_file=str(cfg_path),
                                      default_config=config_manager.DEFAULT_CONFIG)

    mtime_cfg = os.path.getmtime(cfg_path)
    mtime_sec = os.path.getmtime(secrets_path) if secrets_path.exists() else None

    time.sleep(1)
    cm.save_config()

    assert os.path.getmtime(cfg_path) == mtime_cfg
    if mtime_sec is not None:
        assert os.path.getmtime(secrets_path) == mtime_sec
    else:
        assert not secrets_path.exists()
