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


def test_missing_soundfile_prompts_install(monkeypatch):
    """A ausência de ``soundfile`` deve acionar o prompt."""
    dep_module = importlib.reload(importlib.import_module("utils.dependency_checker"))

    orig_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "soundfile":
            raise ImportError
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    prompts = []
    monkeypatch.setattr(dep_module, "_prompt_user", lambda msg: prompts.append(msg) or False)
    monkeypatch.setattr(dep_module.subprocess, "check_call", lambda *a, **k: None)

    dep_module.ensure_dependencies()

    assert any("soundfile" in m for m in prompts)


def test_missing_onnxruntime_prompts_install(monkeypatch):
    """A ausência de ``onnxruntime`` deve acionar o prompt."""
    dep_module = importlib.reload(importlib.import_module("utils.dependency_checker"))

    orig_import = importlib.import_module

    def fake_import(name, package=None):
        if name == "onnxruntime":
            raise ImportError
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    prompts = []
    monkeypatch.setattr(dep_module, "_prompt_user", lambda msg: prompts.append(msg) or False)
    monkeypatch.setattr(dep_module.subprocess, "check_call", lambda *a, **k: None)

    dep_module.ensure_dependencies()

    assert any("onnxruntime" in m for m in prompts)
