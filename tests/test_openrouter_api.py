import os
import sys
import requests
import json
from unittest.mock import patch, MagicMock

# Garantir que o diretório src esteja no path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

from src.openrouter_api import OpenRouterAPI  # noqa: E402


def test_correct_text_success():
    api = OpenRouterAPI(api_key='dummy')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'texto corrigido'}}]
    }
    mock_response.raise_for_status.return_value = None

    with patch(
        'src.openrouter_api.requests.post',
        return_value=mock_response,
    ) as mock_post:
        result = api.correct_text('texto original')
        assert result == 'texto corrigido'
        mock_post.assert_called_once()


def test_correct_text_fails_after_retries():
    api = OpenRouterAPI(api_key='dummy')
    with patch(
        'src.openrouter_api.requests.post',
        side_effect=requests.exceptions.RequestException('boom'),
    ) as mock_post:
        with patch('src.openrouter_api.time.sleep', return_value=None):
            result = api.correct_text(
                'texto original', max_retries=3, retry_delay=0.01
            )

    assert mock_post.call_count == 3
    assert result == 'texto original'


def test_correct_text_with_empty_string():
    api = OpenRouterAPI(api_key='dummy')
    with patch('src.openrouter_api.requests.post') as mock_post:
        result = api.correct_text('')
    assert result == ''
    mock_post.assert_not_called()


def test_correct_text_async_custom_prompt():
    api = OpenRouterAPI(api_key='old')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'corrigido'}}]
    }
    mock_response.raise_for_status.return_value = None

    prompt = 'Corrija o texto: {text}'
    with patch(
        'src.openrouter_api.requests.post',
        return_value=mock_response,
    ) as mock_post:
        result = api.correct_text_async(
            'texto original', prompt, 'newkey', 'modelo1'
        )

        assert result == 'corrigido'
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        sent_payload = json.loads(kwargs['data'])
        assert sent_payload['model'] == 'modelo1'
        assert sent_payload['messages'][0]['content'] == (
            "You are a text correction assistant. "
            "Your task is to correct the following transcribed text:\n"
            "1. Fix punctuation (commas, periods, question marks, etc.)\n"
            "2. Maintain the original meaning and all content\n"
            "3. Do not add, edit, or remove information/words.\n"
            "4. Return only the corrected text without any explanations or "
            "additional comments"
        )
        assert sent_payload['messages'][1]['content'] == 'Corrija o texto: texto original'
        assert kwargs['headers']['Authorization'] == 'Bearer newkey'
