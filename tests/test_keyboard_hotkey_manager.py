import json
from pathlib import Path
from keyboard_hotkey_manager import KeyboardHotkeyManager


def test_update_config(tmp_path, monkeypatch):
    config_file = tmp_path / "cfg.json"
    manager = KeyboardHotkeyManager(config_file=str(config_file))

    monkeypatch.setattr(manager, "_unregister_hotkeys", lambda: None)
    monkeypatch.setattr(manager, "_register_hotkeys", lambda: True)

    result = manager.update_config(record_key="f2", agent_key="f6", record_mode="press")
    assert result
    assert manager.record_key == "f2"
    assert manager.agent_key == "f6"
    assert manager.record_mode == "press"

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["record_key"] == "f2"
    assert data["agent_key"] == "f6"
    assert data["record_mode"] == "press"
