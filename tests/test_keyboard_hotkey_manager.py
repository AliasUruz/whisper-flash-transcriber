import os
import sys
import types
import unittest
from unittest.mock import patch

# Garantir que o diretório src esteja no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Módulo keyboard falso para evitar dependência real
fake_keyboard = types.ModuleType("keyboard")
fake_keyboard.unhook_all = lambda *a, **k: None
sys.modules.setdefault("keyboard", fake_keyboard)

from src.keyboard_hotkey_manager import KeyboardHotkeyManager

class KeyboardHotkeyManagerFailureTests(unittest.TestCase):
    def setUp(self):
        patcher_load = patch.object(KeyboardHotkeyManager, "_load_config", lambda self: None)
        patcher_save = patch.object(KeyboardHotkeyManager, "_save_config", lambda self: None)
        patcher_unreg = patch.object(KeyboardHotkeyManager, "_unregister_hotkeys", lambda self: None)
        self.addCleanup(patcher_load.stop)
        self.addCleanup(patcher_save.stop)
        self.addCleanup(patcher_unreg.stop)
        patcher_load.start()
        patcher_save.start()
        patcher_unreg.start()
        self.manager = KeyboardHotkeyManager(config_file="dummy.json")

    def test_start_failure(self):
        with patch.object(self.manager, "_register_hotkeys", return_value=False):
            result = self.manager.start()
        self.assertFalse(result)
        self.assertFalse(self.manager.is_running)

    def test_update_config_failure_when_running(self):
        self.manager.is_running = True
        with patch.object(self.manager, "_register_hotkeys", return_value=False):
            result = self.manager.update_config(record_key="a")
        self.assertFalse(result)
        self.assertFalse(self.manager.is_running)

    def test_restart_propagates_failure(self):
        with patch.object(self.manager, "_register_hotkeys", return_value=False):
            result = self.manager.restart()
        self.assertFalse(result)
        self.assertFalse(self.manager.is_running)

if __name__ == "__main__":
    unittest.main()
