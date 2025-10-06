from __future__ import annotations

import ctypes
import logging
import os
from importlib import import_module, util as importlib_util
from types import ModuleType

LOGGER = logging.getLogger(__name__)

_PSUTIL: ModuleType | None
_spec = importlib_util.find_spec("psutil")
if _spec is not None:
    _PSUTIL = import_module("psutil")  # type: ignore[assignment]
else:
    _PSUTIL = None

_WARNED_ABOUT_PSUTIL = False


def _bytes_to_mb(value: int) -> int:
    return int(value / (1024 ** 2))


def _warn_psutil_missing() -> None:
    global _WARNED_ABOUT_PSUTIL
    if not _WARNED_ABOUT_PSUTIL:
        LOGGER.warning(
            "psutil module is unavailable; falling back to limited memory metrics."
        )
        _WARNED_ABOUT_PSUTIL = True


def _fallback_memory_status() -> tuple[int, int]:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):  # pragma: no cover - Windows specific
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys), int(status.ullTotalPhys)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("GlobalMemoryStatusEx call failed.", exc_info=True)
        return 0, 0

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            if isinstance(avail_pages, int) and avail_pages >= 0:
                available = avail_pages * page_size
            else:
                available = phys_pages * page_size
            return available, phys_pages * page_size
        except (OSError, ValueError, AttributeError):
            LOGGER.debug("sysconf memory fallback failed.", exc_info=True)
    return 0, 0


def get_available_memory_mb() -> int:
    if _PSUTIL is not None:
        return _bytes_to_mb(int(_PSUTIL.virtual_memory().available))
    _warn_psutil_missing()
    available, _ = _fallback_memory_status()
    return _bytes_to_mb(available)


def get_total_memory_mb() -> int:
    if _PSUTIL is not None:
        return _bytes_to_mb(int(_PSUTIL.virtual_memory().total))
    _warn_psutil_missing()
    _, total = _fallback_memory_status()
    return _bytes_to_mb(total)


__all__ = ["get_available_memory_mb", "get_total_memory_mb"]
