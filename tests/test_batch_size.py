import importlib
import os
import sys
from types import SimpleNamespace
from unittest import mock

# Garantir que o diretório src esteja no path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    ),
)


def _run_with_vram(free_gb):
    free_bytes = int(free_gb * (1024 ** 3))
    total_bytes = free_bytes * 2

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            mem_get_info=lambda device: (free_bytes, total_bytes),
            is_available=lambda: True,
        ),
        device=lambda x: x,
    )

    with mock.patch.dict(sys.modules, {'torch': fake_torch}):
        module = importlib.reload(importlib.import_module('utils.batch_size'))
        return module.select_batch_size(0)


def test_select_batch_size_thresholds():
    cases = [
        (12, 32),
        (8, 16),
        (5, 8),
        (3, 4),
        (1, 2),
    ]
    for free_gb, expected in cases:
        assert _run_with_vram(free_gb) == expected
