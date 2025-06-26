import os
import sys
import types
import unittest
from unittest.mock import patch, MagicMock

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
        patcher_save = patch.object(KeyboardHotkeyManager, "_save_config")
        patcher_unreg = patch.object(KeyboardHotkeyManager, "_unregister_hotkeys", lambda self: None)
        self.addCleanup(patcher_load.stop)
        self.addCleanup(patcher_save.stop)
        self.addCleanup(patcher_unreg.stop)
        patcher_load.start()
        self.mock_save = patcher_save.start()
        patcher_unreg.start()
        self.manager = KeyboardHotkeyManager(config_file="dummy.json")

    def test_start_failure(self):
        patcher = patch.object(self.manager, "_register_hotkeys", return_value=False)
        self.addCleanup(patcher.stop)
        patcher.start()
        result = self.manager.start()
        self.assertFalse(result)
        self.assertFalse(self.manager.is_running)

    def test_update_config_failure_when_running(self):
        self.manager.is_running = True
        patcher = patch.object(self.manager, "_register_hotkeys", return_value=False)
        self.addCleanup(patcher.stop)
        patcher.start()
        result = self.manager.update_config(record_key="a")
        self.assertFalse(result)
        self.assertFalse(self.manager.is_running)

    def test_restart_propagates_failure(self):
        patcher = patch.object(self.manager, "_register_hotkeys", return_value=False)
        self.addCleanup(patcher.stop)
        patcher.start()
        result = self.manager.restart()
        self.assertFalse(result)
        self.assertFalse(self.manager.is_running)

    def test_restart_calls_unhook_and_sleep(self):
        with patch('src.keyboard_hotkey_manager.keyboard.unhook_all') as mock_unhook_all,
             patch('src.keyboard_hotkey_manager.time.sleep') as mock_sleep,
             patch.object(self.manager, "_register_hotkeys", return_value=True):
            
            self.manager.restart()
            
            mock_unhook_all.assert_called()
            self.assertEqual(mock_sleep.call_count, 2) # Two sleep calls in restart

    def test_start_success(self):
        patcher = patch.object(self.manager, "_register_hotkeys", return_value=True)
        self.addCleanup(patcher.stop)
        patcher.start()
        result = self.manager.start()
        self.assertTrue(result)
        self.assertTrue(self.manager.is_running)

    def test_update_config_when_stopped(self):
        patcher = patch.object(self.manager, "_register_hotkeys", return_value=True)
        self.addCleanup(patcher.stop)
        mock_register = patcher.start()
        result = self.manager.update_config(record_key="b", agent_key="c")
        self.assertTrue(result)
        self.assertTrue(self.mock_save.called)
        self.assertFalse(mock_register.called)

if __name__ == "__main__":
    unittest.main()
