import os
import sys
import importlib
from types import ModuleType

# Garantir que o diretório src esteja no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def test_optimum_without_version(monkeypatch):
    """``ensure_dependencies`` deve lidar com ``optimum`` sem ``__version__``."""
    previous = sys.modules.pop("optimum", None)
    stub = ModuleType("optimum")
    sys.modules["optimum"] = stub

    dep_module = importlib.reload(importlib.import_module("utils.dependency_checker"))
    monkeypatch.setattr(dep_module, "_prompt_user", lambda msg: False)
    monkeypatch.setattr(dep_module.subprocess, "check_call", lambda *a, **k: None)

    dep_module.ensure_dependencies()

    sys.modules.pop("optimum", None)
    if previous is not None:
        sys.modules["optimum"] = previous
