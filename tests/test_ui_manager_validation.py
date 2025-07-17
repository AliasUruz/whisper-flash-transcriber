import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Stub do customtkinter para evitar dependências gráficas
fake_ctk = types.ModuleType("customtkinter")
sys.modules.setdefault("customtkinter", fake_ctk)
sys.modules.setdefault("pystray", types.ModuleType("pystray"))
sys.modules.setdefault("PIL", types.ModuleType("PIL"))
sys.modules.setdefault("PIL.Image", types.ModuleType("PIL.Image"))
sys.modules.setdefault("PIL.ImageDraw", types.ModuleType("PIL.ImageDraw"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

from src.ui_manager import UIManager

class DummyVar:
    def __init__(self, value):
        self._value = value
    def get(self):
        return self._value

def _make_manager():
    return UIManager.__new__(UIManager)

def test_safe_get_int_invalid(monkeypatch):
    manager = _make_manager()
    var = DummyVar("abc")
    mock_error = MagicMock()
    monkeypatch.setattr('src.ui_manager.messagebox.showerror', mock_error)
    result = manager._safe_get_int(var, "Teste", None)
    assert result is None
    mock_error.assert_called_once()

def test_safe_get_float_invalid(monkeypatch):
    manager = _make_manager()
    var = DummyVar("xyz")
    mock_error = MagicMock()
    monkeypatch.setattr('src.ui_manager.messagebox.showerror', mock_error)
    result = manager._safe_get_float(var, "Teste", None)
    assert result is None
    mock_error.assert_called_once()

def test_safe_get_int_valid(monkeypatch):
    manager = _make_manager()
    var = DummyVar("42")
    mock_error = MagicMock()
    monkeypatch.setattr('src.ui_manager.messagebox.showerror', mock_error)
    assert manager._safe_get_int(var, "Teste", None) == 42
    mock_error.assert_not_called()

def test_safe_get_float_valid(monkeypatch):
    manager = _make_manager()
    var = DummyVar("3.14")
    mock_error = MagicMock()
    monkeypatch.setattr('src.ui_manager.messagebox.showerror', mock_error)
    assert manager._safe_get_float(var, "Teste", None) == 3.14
    mock_error.assert_not_called()


def test_format_elapsed():
    manager = _make_manager()
    assert manager._format_elapsed(11) == "00:11"
    assert manager._format_elapsed(73) == "01:13"
