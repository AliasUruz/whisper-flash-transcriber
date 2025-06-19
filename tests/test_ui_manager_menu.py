import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import types
fake_ctk = types.ModuleType("customtkinter")
sys.modules["customtkinter"] = fake_ctk
fake_pystray = types.ModuleType("pystray")
fake_pystray.MenuItem = lambda *a, **k: types.SimpleNamespace(text=a[0])
fake_pystray.Menu = lambda *a, **k: list(a)
fake_pystray.Menu.SEPARATOR = object()
sys.modules["pystray"] = fake_pystray
fake_pil = types.ModuleType("PIL")
fake_pil.Image = types.SimpleNamespace(new=lambda *a, **k: None)
fake_pil.ImageDraw = types.SimpleNamespace(Draw=lambda *a, **k: None)
sys.modules["PIL"] = fake_pil
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = fake_torch

import src.ui_manager as ui_manager
from src.ui_manager import UIManager
from src.config_manager import GEMINI_MODEL_OPTIONS_CONFIG_KEY

class DummyConfig:
    def __init__(self):
        self.data = {GEMINI_MODEL_OPTIONS_CONFIG_KEY: []}
    def get(self, key, default=None):
        return self.data.get(key, default)

class DummyCore:
    def __init__(self, state):
        self.current_state = state
    def toggle_recording(self):
        pass
    def cancel_recording_and_corrections(self):
        pass
    def is_transcription_running(self):
        return False
    def is_correction_running(self):
        return False

class DummyPystray:
    class MenuItem:
        def __init__(self, text, *a, **k):
            self.text = text
    class Menu(list):
        def __init__(self, *items):
            super().__init__(items)
    Menu.SEPARATOR = object()


def test_menu_contains_cancel_option_only_when_recording(monkeypatch):
    dummy_tk = types.SimpleNamespace()
    core = DummyCore('RECORDING')
    ui = UIManager(dummy_tk, DummyConfig(), core)
    monkeypatch.setattr(ui_manager, 'pystray', DummyPystray)

    menu = ui.create_dynamic_menu()
    texts = [getattr(item, 'text', None) for item in menu if hasattr(item, 'text')]
    assert '🛑 Cancelar Transcrição e Correções' in texts

    core.current_state = 'IDLE'
    menu = ui.create_dynamic_menu()
    texts = [getattr(item, 'text', None) for item in menu if hasattr(item, 'text')]
    assert '🛑 Cancelar Transcrição e Correções' not in texts
