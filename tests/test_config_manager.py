import os
import time
from src import config_manager


def test_skip_save_when_unchanged(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))
    cm = config_manager.ConfigManager(
        config_file=str(cfg_path),
        default_config=config_manager.DEFAULT_CONFIG,
    )

    mtime_cfg = os.path.getmtime(cfg_path)
    if secrets_path.exists():
        mtime_sec = os.path.getmtime(secrets_path)
    else:
        mtime_sec = None

    time.sleep(1)
    cm.save_config()

    assert os.path.getmtime(cfg_path) == mtime_cfg
    if mtime_sec is not None:
        assert os.path.getmtime(secrets_path) == mtime_sec
    else:
        assert not secrets_path.exists()
