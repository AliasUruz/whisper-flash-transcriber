import importlib.machinery
import importlib
import types
import sys
import os
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Stubs básicos para torch e transformers
fake_torch = types.ModuleType("torch")
fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
fake_torch.__version__ = "0.0"
fake_torch.cuda = types.SimpleNamespace(
    is_available=lambda: True,
    get_device_capability=lambda _=0: (8, 0),
)
fake_torch.float16 = 1
fake_torch.float32 = 2
sys.modules["torch"] = fake_torch

fake_transformers = types.ModuleType("transformers")
fake_transformers.pipeline = MagicMock()
fake_transformers.AutoProcessor = MagicMock()
fake_transformers.AutoModelForSpeechSeq2Seq = MagicMock()
sys.modules["transformers"] = fake_transformers

# Stub para transformers.integrations.BetterTransformer
fake_integrations = types.ModuleType("transformers.integrations")
fake_integrations.BetterTransformer = object
sys.modules["transformers.integrations"] = fake_integrations

if "src.transcription_handler" in sys.modules:
    importlib.reload(sys.modules["src.transcription_handler"])

from src.transcription_handler import TranscriptionHandler  # noqa: E402
import src.transcription_handler as th_module  # noqa: E402
th_module.BETTERTRANSFORMER_AVAILABLE = True
from src.config_manager import (  # noqa: E402
    BATCH_SIZE_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY,
    MANUAL_BATCH_SIZE_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE,
    OPENROUTER_API_KEY_CONFIG_KEY,
    OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    USE_FLASH_ATTENTION_2_CONFIG_KEY,
    USE_TURBO_CONFIG_KEY,
    TEXT_CORRECTION_TIMEOUT_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
)


class DummyConfig:
    def __init__(self):
        self.data = {
            BATCH_SIZE_CONFIG_KEY: 16,
            BATCH_SIZE_MODE_CONFIG_KEY: "auto",
            MANUAL_BATCH_SIZE_CONFIG_KEY: 8,
            GPU_INDEX_CONFIG_KEY: 0,
            "batch_size_specified": False,
            "gpu_index_specified": False,
            TEXT_CORRECTION_ENABLED_CONFIG_KEY: False,
            TEXT_CORRECTION_SERVICE_CONFIG_KEY: SERVICE_NONE,
            USE_TURBO_CONFIG_KEY: False,
            OPENROUTER_API_KEY_CONFIG_KEY: "",
            OPENROUTER_MODEL_CONFIG_KEY: "",
            GEMINI_API_KEY_CONFIG_KEY: "",
            "gemini_agent_model": "",
            "gemini_prompt": "",
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY: 1.0,
            DISPLAY_TRANSCRIPTS_KEY: False,
            SAVE_TEMP_RECORDINGS_CONFIG_KEY: False,
            USE_FLASH_ATTENTION_2_CONFIG_KEY: True,
            TEXT_CORRECTION_TIMEOUT_CONFIG_KEY: 30,
        }

    def get(self, key, default=None):
        return self.data.get(key, default)


def noop(*_a, **_k):
    return None


def _create_handler(cfg, on_opt_fallback=None):
    return TranscriptionHandler(
        cfg,
        gemini_api_client=None,
        on_model_ready_callback=noop,
        on_model_error_callback=noop,
        on_optimization_fallback_callback=on_opt_fallback or (lambda *_: None),
        on_transcription_result_callback=noop,
        on_agent_result_callback=noop,
        on_segment_transcribed_callback=None,
        is_state_transcribing_fn=lambda: False,
    )


def test_bettertransformer_aplicado_quando_turbo(monkeypatch):
    cfg = DummyConfig()
    cfg.data[USE_TURBO_CONFIG_KEY] = True

    import src.transcription_handler as th_module
    monkeypatch.setattr(th_module, "BETTERTRANSFORMER_AVAILABLE", True)

    called = {"count": 0}

    class DummyModel:
        def to_bettertransformer(self):
            called["count"] += 1
            return self

    class DummyPipeline:
        def __init__(self):
            self.model = DummyModel()

    monkeypatch.setattr(th_module, "pipeline", lambda *a, **k: DummyPipeline())

    handler = _create_handler(cfg)
    handler._load_model_task()

    assert called["count"] == 1


def test_bettertransformer_ignorado_sem_turbo(monkeypatch):
    cfg = DummyConfig()
    cfg.data[USE_TURBO_CONFIG_KEY] = False

    import src.transcription_handler as th_module

    called = {"flag": 0}

    class DummyModel:
        def to_bettertransformer(self):
            called["flag"] += 1
            return self

    class DummyPipeline:
        def __init__(self):
            self.model = DummyModel()

    monkeypatch.setattr(th_module, "pipeline", lambda *a, **k: DummyPipeline())

    handler = _create_handler(cfg)
    handler._load_model_task()

    assert called["flag"] == 0


def test_bettertransformer_indisponivel(monkeypatch):
    cfg = DummyConfig()
    cfg.data[USE_TURBO_CONFIG_KEY] = True

    if "transformers.integrations" in sys.modules:
        del sys.modules["transformers.integrations"]

    import importlib
    import src.transcription_handler as th_module
    importlib.reload(th_module)

    called = {"flag": 0}

    class DummyModel:
        def to_bettertransformer(self):
            called["flag"] += 1
            return self

    class DummyPipeline:
        def __init__(self):
            self.model = DummyModel()

    monkeypatch.setattr(th_module, "pipeline", lambda *a, **k: DummyPipeline())

    handler = _create_handler(cfg)
    handler._load_model_task()

    assert called["flag"] == 0


def test_mensagem_quando_bettertransformer_indisponivel(monkeypatch):
    cfg = DummyConfig()
    cfg.data[USE_TURBO_CONFIG_KEY] = True

    if "transformers.integrations" in sys.modules:
        del sys.modules["transformers.integrations"]

    import importlib
    import src.transcription_handler as th_module
    importlib.reload(th_module)

    messages = []

    class DummyModel:
        def to_bettertransformer(self):
            raise AssertionError("Should not be called")

    class DummyPipeline:
        def __init__(self):
            self.model = DummyModel()

    monkeypatch.setattr(th_module, "pipeline", lambda *a, **k: DummyPipeline())

    handler = _create_handler(cfg, on_opt_fallback=lambda msg: messages.append(msg))
    handler._load_model_task()

    assert messages
    assert messages[0].startswith(th_module.OPTIMIZATION_TURBO_FALLBACK_MSG)
    assert (
        "BetterTransformer indisponível. Verifique se as versões de Transformers e Optimum são compatíveis"
        in messages[0]
    )


def test_model_transformado_com_bettertransformer(monkeypatch):
    cfg = DummyConfig()
    cfg.data[USE_TURBO_CONFIG_KEY] = True

    import src.transcription_handler as th_module
    monkeypatch.setattr(th_module, "BETTERTRANSFORMER_AVAILABLE", True)

    better_model = object()

    class DummyModel:
        def to_bettertransformer(self):
            return better_model

    class DummyPipeline:
        def __init__(self):
            self.model = DummyModel()

    monkeypatch.setattr(th_module, "pipeline", lambda *a, **k: DummyPipeline())

    handler = _create_handler(cfg)
    handler._load_model_task()

    assert handler.transcription_pipeline.model is better_model
