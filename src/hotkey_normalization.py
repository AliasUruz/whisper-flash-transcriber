"""Utilities to normalize hotkey names across keyboard layouts."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["_normalize_key_name"]

# Pattern that splits modifiers using '+' while allowing escaped plus signs.
_HOTKEY_SEPARATOR = re.compile(r"(?<!\\)\+")

# Prefixes commonly typed by users when referring to a key.
_PREFIX_PATTERN = re.compile(r"^(?:tecla|key)\s+", re.IGNORECASE)

# Map of layout-specific tokens to their `keyboard` equivalents.
_ABNT_EQUIVALENTS: dict[str, str] = {
    "\\'": "apostrophe",
    "'": "apostrophe",
    "´": "apostrophe",
    "`": "grave",
    "¨": "dead_diaeresis",
    "^": "caret",
    "~": "tilde",
    "ç": "semicolon",
    "Ç": "semicolon",
    "¸": ",",
}

# Generic aliases and translations that should converge to canonical `keyboard` names.
_KEY_ALIASES: dict[str, str] = {
    "control": "ctrl",
    "ctl": "ctrl",
    "left control": "left ctrl",
    "right control": "right ctrl",
    "left ctrl": "left ctrl",
    "right ctrl": "right ctrl",
    "shift esquerdo": "left shift",
    "shift direito": "right shift",
    "left shift": "left shift",
    "right shift": "right shift",
    "altgr": "alt gr",
    "alt-gr": "alt gr",
    "option": "alt",
    "windows": "win",
    "super": "win",
    "command": "cmd",
    "seta para cima": "up",
    "seta para baixo": "down",
    "seta para esquerda": "left",
    "seta para direita": "right",
    "seta esquerda": "left",
    "seta direita": "right",
    "seta cima": "up",
    "seta baixo": "down",
    "espaço": "space",
    "espaco": "space",
    "barra de espaco": "space",
    "barra de espaço": "space",
    "retorno": "enter",
    "return": "enter",
    "capslock": "caps lock",
    "numlock": "num lock",
    "scrolllock": "scroll lock",
    "tabulação": "tab",
    "tabulacao": "tab",
    "delete": "del",
    "seta": "arrow",
}

def _strip_accents(value: str) -> str:
    """Remove diacritics to ease comparisons between layouts."""

    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if unicodedata.category(char) != "Mn"
    )


def _sanitize_token(token: str) -> str:
    """Clean up a single key token before alias resolution."""

    token = token.strip().strip("\"'[](){}")
    token = _PREFIX_PATTERN.sub("", token).strip()
    token = token.replace("\\+", "+")
    token = re.sub(r"\s+", " ", token)
    return token


def _resolve_alias(token: str) -> str:
    """Resolve aliases, including ABNT-specific mappings."""

    if not token:
        return token

    lowered = token.lower()
    ascii_lower = _strip_accents(lowered)

    for candidate in (lowered, ascii_lower):
        if candidate in _ABNT_EQUIVALENTS:
            return _ABNT_EQUIVALENTS[candidate]

    for candidate in (lowered, ascii_lower):
        if candidate in _KEY_ALIASES:
            return _KEY_ALIASES[candidate]

    return lowered


def _normalize_key_name(value: str | None) -> str:
    """Normalize user-provided key names for consistent registration."""

    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    parts = _HOTKEY_SEPARATOR.split(text)
    normalized_parts: list[str] = []
    for raw_part in parts:
        candidate = _sanitize_token(raw_part)
        if not candidate:
            continue
        normalized = _resolve_alias(candidate)
        if normalized:
            normalized_parts.append(normalized)

    return "+".join(normalized_parts)
