import sys
from pathlib import Path
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model_manager import ensure_download  # noqa: E402


def test_ensure_download_invalid_backend(tmp_path):
    with pytest.raises(ValueError):
        ensure_download('tiny', 'invalid', tmp_path.as_posix(), None)


def test_ensure_download_transformers_failure(tmp_path, monkeypatch):
    def fake_download(*args, **kwargs):
        raise RuntimeError('fail')
    monkeypatch.setattr('model_manager.snapshot_download', fake_download)
    with pytest.raises(RuntimeError):
        ensure_download('tiny', 'transformers', tmp_path.as_posix(), None)


def test_ensure_download_ct2_failure(tmp_path, monkeypatch):
    fw_utils = types.SimpleNamespace()

    def fake_ct2(*args, **kwargs):
        raise RuntimeError('fail')
    fw_utils.download_model = fake_ct2
    fake_fw = types.SimpleNamespace(utils=fw_utils)
    sys.modules['faster_whisper'] = fake_fw
    sys.modules['faster_whisper.utils'] = fw_utils
    try:
        with pytest.raises(RuntimeError):
            ensure_download('tiny', 'ct2', tmp_path.as_posix(), 'float16')
    finally:
        sys.modules.pop('faster_whisper', None)
        sys.modules.pop('faster_whisper.utils', None)
