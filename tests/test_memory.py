import importlib
import os
import sys
from types import SimpleNamespace
from unittest import mock

# Garantir que o diretório src esteja no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def _run_with_available_bytes(num_bytes):
    fake_mem = SimpleNamespace(available=num_bytes)
    with mock.patch("psutil.virtual_memory", return_value=fake_mem):
        module = importlib.reload(importlib.import_module("utils.memory"))
        return module.get_available_memory_mb()


def test_get_available_memory_mb():
    assert _run_with_available_bytes(1_048_576) == 1
    assert _run_with_available_bytes(5_000_000_000) == 4768

