import os
import sys
import time
import os, sys
import json
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
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


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("false", False), (1, True), (0, False)],
)
def test_parse_bool_values(tmp_path, monkeypatch, value, expected):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    config = {
        "auto_paste": value,
        "display_transcripts_in_terminal": value,
        "save_temp_recordings": value,
        "use_vad": value,
        "record_to_memory": value,
        "max_memory_seconds": 5,
    }

    cfg_path.write_text(json.dumps(config))
    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))
    cm = config_manager.ConfigManager(
        config_file=str(cfg_path),
        default_config=config_manager.DEFAULT_CONFIG,
    )

    assert cm.get("auto_paste") is expected
    assert cm.get(config_manager.DISPLAY_TRANSCRIPTS_KEY) is expected
    assert cm.get(config_manager.SAVE_TEMP_RECORDINGS_CONFIG_KEY) is expected
    assert cm.get(config_manager.USE_VAD_CONFIG_KEY) is expected
    assert cm.get_record_to_memory() is expected
    assert cm.get_max_memory_seconds() == 5


def test_hybrid_storage_mode_maps_to_auto(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    cfg_path.write_text(json.dumps({"record_storage_mode": "hybrid"}))
    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))

    cm = config_manager.ConfigManager(
        config_file=str(cfg_path),
        default_config=config_manager.DEFAULT_CONFIG,
    )

    assert cm.get_record_storage_mode() == "auto"


def test_max_memory_seconds_mode(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    cfg_path.write_text(json.dumps({"max_memory_seconds_mode": "auto"}))
    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))

    cm = config_manager.ConfigManager(
        config_file=str(cfg_path),
        default_config=config_manager.DEFAULT_CONFIG,
    )

    assert cm.get_max_memory_seconds_mode() == "auto"
