"""Helper functions for validating form inputs in the UI layer."""

from __future__ import annotations

import tkinter.messagebox as messagebox
from typing import Any, Optional, Protocol


class _SupportsGet(Protocol):
    def get(self) -> Any:
        ...


def _extract_value(value: Any | _SupportsGet) -> Any:
    """Return the underlying value from ``value`` or from its ``get`` method."""
    if hasattr(value, "get"):
        return value.get()  # type: ignore[call-arg]
    return value


def safe_get_int(var: Any | _SupportsGet, field_name: str, parent) -> Optional[int]:
    """Convert a variable to ``int`` showing a message box on failure."""
    try:
        raw_value = _extract_value(var)
        return int(raw_value)
    except (TypeError, ValueError):
        messagebox.showerror("Valor inválido", f"Valor inválido para {field_name}.", parent=parent)
        return None


def safe_get_float(var: Any | _SupportsGet, field_name: str, parent) -> Optional[float]:
    """Convert a variable to ``float`` showing a message box on failure."""
    try:
        raw_value = _extract_value(var)
        return float(raw_value)
    except (TypeError, ValueError):
        messagebox.showerror("Valor inválido", f"Valor inválido para {field_name}.", parent=parent)
        return None


__all__ = ["safe_get_int", "safe_get_float"]
