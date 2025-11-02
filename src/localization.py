"""Minimal helpers for handling localized strings."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_LANGUAGE = "en-US"
_DEFAULT_LANGUAGE_CODE = "en-us"


TRANSLATIONS: dict[str, dict[str, str]] = {
    # Text Correction section (AI Services)
    "settings.text_correction.title": {
        "pt-BR": "Correção de Texto (Serviços de IA)",
        "en-US": "Text Correction (AI Services)",
    },
    "settings.text_correction.toggle": {
        "pt-BR": "Ativar Correção de Texto",
        "en-US": "Enable Text Correction",
    },
    "settings.text_correction.toggle.tooltip": {
        "pt-BR": "Usa um serviço de IA para refinar o texto.",
        "en-US": "Uses an AI service to refine the transcript.",
    },
    "settings.text_correction.service.label": {
        "pt-BR": "Serviço:",
        "en-US": "Service:",
    },
    "settings.text_correction.service.tooltip": {
        "pt-BR": "Selecione o serviço de correção de texto.",
        "en-US": "Select the text correction service.",
    },
    "settings.text_correction.timeout.label": {
        "pt-BR": "Tempo limite da correção (s):",
        "en-US": "Correction timeout (s):",
    },
    "settings.text_correction.timeout.tooltip": {
        "pt-BR": "Tempo máximo para aguardar a correção antes de usar o texto bruto.",
        "en-US": "Maximum wait time before falling back to the raw text.",
    },
    "settings.text_correction.openrouter.api_key.label": {
        "pt-BR": "Chave OpenRouter:",
        "en-US": "OpenRouter Key:",
    },
    "settings.text_correction.openrouter.api_key.tooltip": {
        "pt-BR": "Chave da API OpenRouter.",
        "en-US": "OpenRouter API key.",
    },
    "settings.text_correction.openrouter.model.label": {
        "pt-BR": "Modelo OpenRouter:",
        "en-US": "OpenRouter Model:",
    },
    "settings.text_correction.openrouter.model.tooltip": {
        "pt-BR": "Modelo utilizado no OpenRouter.",
        "en-US": "Model used with OpenRouter.",
    },
    "settings.text_correction.gemini.api_key.label": {
        "pt-BR": "Chave Gemini:",
        "en-US": "Gemini Key:",
    },
    "settings.text_correction.gemini.api_key.tooltip": {
        "pt-BR": "Chave da API Gemini.",
        "en-US": "Gemini API key.",
    },
    "settings.text_correction.gemini.model.label": {
        "pt-BR": "Modelo Gemini:",
        "en-US": "Gemini Model:",
    },
    "settings.text_correction.gemini.model.tooltip": {
        "pt-BR": "Modelo utilizado nas requisições Gemini.",
        "en-US": "Model used for Gemini requests.",
    },
    "settings.text_correction.gemini.agent_model.label": {
        "pt-BR": "Modelo do Agente:",
        "en-US": "Agent Model:",
    },
    "settings.text_correction.gemini.agent_model.tooltip": {
        "pt-BR": "Modelo dedicado às ações do modo agente.",
        "en-US": "Model dedicated to agent mode actions.",
    },
    "settings.text_correction.gemini.prompt.label": {
        "pt-BR": "Prompt de Correção (Gemini):",
        "en-US": "Correction Prompt (Gemini):",
    },
    "settings.text_correction.gemini.prompt.tooltip": {
        "pt-BR": "Prompt usado para refinar o texto.",
        "en-US": "Prompt used to refine the text.",
    },
    "settings.text_correction.agent_prompt.label": {
        "pt-BR": "Prompt do Modo Agente:",
        "en-US": "Agent Mode Prompt:",
    },
    "settings.text_correction.agent_prompt.tooltip": {
        "pt-BR": "Prompt executado no modo agente.",
        "en-US": "Prompt executed in agent mode.",
    },
    "settings.text_correction.model_list.label": {
        "pt-BR": "Modelos Gemini (um por linha):",
        "en-US": "Gemini Models (one per line):",
    },
    "settings.text_correction.model_list.tooltip": {
        "pt-BR": "Lista de modelos para tentativa, um por linha.",
        "en-US": "List of models to try, one per line.",
    },
}


def _normalize_language_code(language: str | None) -> str:
    if not language:
        return _DEFAULT_LANGUAGE_CODE
    normalized = language.strip().replace("_", "-")
    lowered = normalized.lower()
    if lowered.startswith("pt"):
        return "pt-br"
    if lowered.startswith("en"):
        return "en-us"
    return lowered or _DEFAULT_LANGUAGE_CODE


def _resolve_from_mapping(translations: Mapping[str, str], language_code: str) -> str | None:
    if not translations:
        return None
    normalized_map = {key.lower(): value for key, value in translations.items()}
    if language_code in normalized_map:
        return normalized_map[language_code]
    primary = language_code.split("-", 1)[0]
    if primary in normalized_map:
        return normalized_map[primary]
    if _DEFAULT_LANGUAGE_CODE in normalized_map:
        return normalized_map[_DEFAULT_LANGUAGE_CODE]
    return next(iter(translations.values()), None)


def choose_translation(
    language: str | None,
    key: str | None = None,
    default: str | None = None,
    *,
    pt_br: str | None = None,
    en_us: str | None = None,
) -> str:
    """Return the text that best matches the requested language.

    Parameters
    ----------
    language:
        BCP 47 style language tag (for example ``en-US`` or ``pt-BR``).
    key:
        Identifier in :data:`TRANSLATIONS` to resolve. When omitted the function
        falls back to ``pt_br``/``en_us`` values.
    default:
        Text returned when the key is unknown or does not have a translation for
        the requested language.
    pt_br / en_us:
        Legacy parameters kept for backwards compatibility.
    """

    language_code = _normalize_language_code(language)

    if key is not None:
        mapped = _resolve_from_mapping(TRANSLATIONS.get(key, {}), language_code)
        if mapped is not None:
            return mapped
        if default is not None:
            return default

    if pt_br is not None or en_us is not None:
        pt_value = pt_br if pt_br is not None else en_us
        en_value = en_us if en_us is not None else pt_br
        if language_code.startswith("pt"):
            return pt_value if pt_value is not None else (default or "")
        return en_value if en_value is not None else (default or pt_value or "")

    if key is not None:
        fallback = _resolve_from_mapping(TRANSLATIONS.get(key, {}), _DEFAULT_LANGUAGE_CODE)
        if fallback is not None:
            return fallback
        if default is not None:
            return default
        return key

    if default is not None:
        return default

    raise ValueError("choose_translation requires translation data or a default value")
