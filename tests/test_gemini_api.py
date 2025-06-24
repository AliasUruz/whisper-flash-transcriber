import os
import sys
import types
from unittest.mock import MagicMock

# Garantir que o diretório src esteja no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


class DummyConfig:
    def __init__(self):
        self.values = {
            'gemini_api_key': '',
            'gemini_model': '',
            'gemini_agent_model': 'agent-model',
            'gemini_prompt': '{text}',
            'prompt_agentico': 'agent prompt',
        }

    def get(self, key):
        return self.values.get(key)


created_models = []


class FakeGenerativeModel:
    def __init__(self, model_id):
        self.model_id = model_id
        created_models.append(model_id)

    def generate_content(self, text):
        return MagicMock(text='texto corrigido via mock')


def setup_fake_genai(monkeypatch):
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("generativeai")
    fake_genai.configure = lambda api_key=None: None
    fake_genai.GenerativeModel = FakeGenerativeModel
    fake_google.generativeai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    if "src.gemini_api" in sys.modules:
        import importlib
        importlib.reload(sys.modules["src.gemini_api"])



def test_reinitialization(monkeypatch):
    setup_fake_genai(monkeypatch)
    from src.gemini_api import GeminiAPI

    cfg = DummyConfig()
    api = GeminiAPI(cfg)
    assert not api.is_valid

    cfg.values['gemini_api_key'] = 'valid-key'
    cfg.values['gemini_model'] = 'new-model'

    api.reinitialize_client()

    assert api.is_valid
    assert created_models == ['new-model']


def test_get_correction_success(monkeypatch):
    created_models.clear()
    setup_fake_genai(monkeypatch)
    from src.gemini_api import GeminiAPI

    cfg = DummyConfig()
    cfg.values['gemini_api_key'] = 'valid'
    cfg.values['gemini_model'] = 'base-model'

    api = GeminiAPI(cfg)
    corrected = api.get_correction('texto')

    assert corrected == 'texto corrigido via mock'
    assert created_models == ['base-model']


def test_get_agent_response(monkeypatch):
    created_models.clear()
    setup_fake_genai(monkeypatch)
    from src import gemini_api as ga

    def fake_load(self):
        if self.current_model_id is None:
            self.current_model_id = self.config_manager.get("gemini_model")
        self.current_api_key = (
            self.last_api_key or self.config_manager.get("gemini_api_key")
        )
        if not self.current_api_key:
            self.model = None
            self.is_valid = False
            self.last_api_key = self.current_api_key
            return
        self.model = FakeGenerativeModel(self.current_model_id)
        self.last_api_key = self.current_api_key
        self.last_model_id = self.current_model_id
        self.is_valid = True

    monkeypatch.setattr(ga.GeminiAPI, "_load_model_from_config", fake_load)
    GeminiAPI = ga.GeminiAPI

    cfg = DummyConfig()
    cfg.values['gemini_api_key'] = 'valid'
    cfg.values['gemini_model'] = 'base-model'
    cfg.values['gemini_agent_model'] = 'agent-model'

    api = GeminiAPI(cfg)
    response = api.get_agent_response('oi')

    assert response == 'texto corrigido via mock'
    assert created_models == ['base-model', 'agent-model']
    assert api.current_model_id == 'base-model'
