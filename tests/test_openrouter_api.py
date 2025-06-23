import os
import sys
import requests
from unittest.mock import patch, MagicMock

# Garantir que o diretório src esteja no path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.openrouter_api import OpenRouterAPI


def test_correct_text_success():
    api = OpenRouterAPI(api_key='dummy')
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'texto corrigido'}}]
    }
    mock_response.raise_for_status.return_value = None

    with patch('src.openrouter_api.requests.post', return_value=mock_response) as mock_post:
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
            result = api.correct_text('texto original', max_retries=3, retry_delay=0.01)

    assert mock_post.call_count == 3
    assert result == 'texto original'


def test_correct_text_with_empty_string():
    api = OpenRouterAPI(api_key='dummy')
    with patch('src.openrouter_api.requests.post') as mock_post:
        result = api.correct_text('')
    assert result == ''
    mock_post.assert_not_called()
