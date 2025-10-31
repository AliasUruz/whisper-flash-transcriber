"""Minimal helpers for handling localized strings."""

from __future__ import annotations

DEFAULT_LANGUAGE = "en-US"


def choose_translation(
    language: str | None,
    *,
    pt_br: str,
    en_us: str,
) -> str:
    """Return the text that best matches the requested language.

    Parameters
    ----------
    language:
        BCP 47 style language tag (for example ``en-US`` or ``pt-BR``).
    pt_br:
        Text to use when the locale is Portuguese (Brazil).
    en_us:
        Text to use for English (US) or as the global fallback.
    """

    code = (language or DEFAULT_LANGUAGE).lower()
    if code.startswith("pt"):
        return pt_br
    return en_us
