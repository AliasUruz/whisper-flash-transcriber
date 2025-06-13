import pytest
from openrouter_api import OpenRouterAPI


def test_headers_optional():
    api = OpenRouterAPI(api_key="dummy")
    assert "HTTP-Referer" not in api.headers
    assert "X-Title" not in api.headers


def test_headers_custom():
    api = OpenRouterAPI(
        api_key="dummy",
        referer="https://example.com",
        app_title="MinhaApp",
    )
    assert api.headers["HTTP-Referer"] == "https://example.com"
    assert api.headers["X-Title"] == "MinhaApp"
