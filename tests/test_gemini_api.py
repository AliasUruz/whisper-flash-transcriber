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

    def get(self, key, default=None):
        return self.values.get(key, default)


created_models = []


class FakeGenerativeModel:
    def __init__(self, model_id):
        self.model_id = model_id
        created_models.append(model_id)

    def generate_content(self, text, *, request_options=None):
        return MagicMock(text='texto corrigido via mock')


def setup_fake_genai(monkeypatch):
    fake_google = types.ModuleType("google")
    fake_genai = types.ModuleType("generativeai")
    fake_types = types.ModuleType("types")
    class RequestOptions:
        def __init__(self, *, retry=None, timeout=None):
            self.retry = retry
            self.timeout = timeout

    class FakeBrokenResponseError(Exception):
        pass

    class FakeIncompleteIterationError(Exception):
        pass
    fake_helper = types.ModuleType("helper_types")
    fake_helper.RequestOptions = RequestOptions
    fake_types.helper_types = fake_helper
    fake_types.BrokenResponseError = FakeBrokenResponseError
    fake_types.IncompleteIterationError = FakeIncompleteIterationError
    fake_genai.configure = lambda api_key=None: None
    fake_genai.GenerativeModel = FakeGenerativeModel
    fake_genai.types = fake_types
    fake_google.generativeai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.generativeai.types", fake_types)
    monkeypatch.setitem(
        sys.modules,
        "google.generativeai.types.helper_types",
        fake_helper,
    )
    import importlib
    if "src.gemini_api" in sys.modules:
        importlib.reload(sys.modules["src.gemini_api"])



def test_reinitialization(monkeypatch):
    setup_fake_genai(monkeypatch)
    from src.gemini_api import GeminiAPI

    cfg = DummyConfig()
    api = GeminiAPI(cfg)
    assert not api.is_valid

    cfg.values['gemini_api_key'] = 'valid-key'
    cfg.values['gemini_model'] = 'new-model'

    api.reinitialize_client(api_key='valid-key', model_id='new-model')

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

    cfg = DummyConfig()
    cfg.values['gemini_api_key'] = 'valid'
    cfg.values['gemini_model'] = 'base-model'
    cfg.values['gemini_agent_model'] = 'agent-model'

    api = ga.GeminiAPI(cfg)
    
    mock_execute_request = MagicMock(return_value='mocked agent response')
    monkeypatch.setattr(api, '_execute_request', mock_execute_request)

    response = api.get_agent_response('oi')

    assert response == 'mocked agent response'
    mock_execute_request.assert_called_once_with('agent prompt\n\noi', model_id='agent-model')


def test_execute_request_passes_timeout(monkeypatch):
    created_models.clear()
    setup_fake_genai(monkeypatch)
    from src.gemini_api import GeminiAPI
    from google.generativeai.types.helper_types import RequestOptions

    cfg = DummyConfig()
    cfg.values['gemini_api_key'] = 'valid'
    cfg.values['gemini_model'] = 'model'

    captured = {}

    def fake_generate(self, prompt, *, request_options=None):
        captured['options'] = request_options
        return MagicMock(text='ok')

    monkeypatch.setattr(FakeGenerativeModel, 'generate_content', fake_generate)

    api = GeminiAPI(cfg)
    api._execute_request('oi', timeout=5)

    assert isinstance(captured['options'], RequestOptions)
    assert captured['options'].timeout == 5


def test_execute_request_returns_empty_on_timeout(monkeypatch):
    created_models.clear()
    setup_fake_genai(monkeypatch)
    from src.gemini_api import GeminiAPI
    from google.generativeai.types import IncompleteIterationError

    cfg = DummyConfig()
    cfg.values['gemini_api_key'] = 'valid'
    cfg.values['gemini_model'] = 'model'

    def fake_generate(self, prompt, *, request_options=None):
        raise IncompleteIterationError('timeout')

    monkeypatch.setattr(FakeGenerativeModel, 'generate_content', fake_generate)
    monkeypatch.setattr('src.gemini_api.time.sleep', lambda t: None)

    api = GeminiAPI(cfg)
    result = api._execute_request('oi', max_retries=2, retry_delay=0.01, timeout=0.1)

    assert result == ''

def test_correct_text_async(monkeypatch):
    created_models.clear()
    setup_fake_genai(monkeypatch)
    from src.gemini_api import GeminiAPI

    cfg = DummyConfig()
    cfg.values['gemini_api_key'] = 'valid'
    cfg.values['gemini_model'] = 'base-model'

    api = GeminiAPI(cfg)
    
    mock_execute_request = MagicMock(return_value='mocked async correction')
    monkeypatch.setattr(api, '_execute_request', mock_execute_request)

    result = api.correct_text_async('original text', 'prompt {text}', 'valid', 'async-model')

    assert result == 'mocked async correction'
    mock_execute_request.assert_called_once_with('prompt original text', model_id='async-model')
