# -*- coding: utf-8 -*-
"""Hotkey driver abstractions for Whisper Flash Transcriber."""
from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Mapping

import keyboard  # type: ignore[import]

LOGGER = logging.getLogger("whisper_flash_transcriber.hotkeys.drivers")


class BaseHotkeyDriver(ABC):
    """Interface for all hotkey backends."""

    name: str = "base"

    def __init__(self, *, log: Callable[[int, str], None] | None = None) -> None:
        self._log = log

    @abstractmethod
    def register(
        self,
        *,
        record_key: str,
        agent_key: str | None,
        record_mode: str,
        callbacks: Mapping[str, Callable[[], None] | None],
        suppress: bool = False,
    ) -> None:
        """Register the configured hotkeys."""

    @abstractmethod
    def unregister(self) -> None:
        """Remove all registered hotkeys for this driver."""

    @abstractmethod
    def detect(self, timeout: float) -> str | None:
        """Capture a single key event for configuration purposes."""

    # --- Helpers ---------------------------------------------------------

    def _maybe_log(self, level: int, message: str, **fields: Any) -> None:
        if self._log is not None:
            self._log(level, message, **fields)
        else:
            LOGGER.log(level, message, extra={"fields": fields} if fields else None)


class KeyboardLibHotkeyDriver(BaseHotkeyDriver):
    """Driver that relies on the ``keyboard`` package."""

    name = "keyboard"

    def __init__(self, *, log: Callable[[int, str], None] | None = None) -> None:
        super().__init__(log=log)
        self._handles: list[tuple[str, Any]] = []
        self._lock = threading.Lock()

    def register(
        self,
        *,
        record_key: str,
        agent_key: str | None,
        record_mode: str,
        callbacks: Mapping[str, Callable[[], None] | None],
        suppress: bool = False,
    ) -> None:
        with self._lock:
            self.unregister()
            if record_mode == "toggle":
                toggle_cb = callbacks.get("toggle")
                if toggle_cb is not None:
                    handle = keyboard.add_hotkey(
                        record_key,
                        toggle_cb,
                        suppress=suppress,
                        trigger_on_release=False,
                    )
                    self._handles.append(("add_hotkey", handle))
            else:
                start_cb = callbacks.get("start")
                stop_cb = callbacks.get("stop")
                if start_cb is not None:
                    press_handle = keyboard.on_press_key(record_key, lambda _: start_cb())
                    self._handles.append(("press", press_handle))
                if stop_cb is not None:
                    release_handle = keyboard.on_release_key(record_key, lambda _: stop_cb())
                    self._handles.append(("release", release_handle))

            agent_cb = callbacks.get("agent")
            if agent_key and agent_cb is not None:
                agent_handle = keyboard.add_hotkey(
                    agent_key,
                    agent_cb,
                    suppress=False,
                    trigger_on_release=False,
                )
                self._handles.append(("add_hotkey", agent_handle))

    def unregister(self) -> None:
        with self._lock:
            while self._handles:
                handle_type, handle = self._handles.pop()
                try:
                    if handle_type == "add_hotkey":
                        keyboard.remove_hotkey(handle)
                    else:
                        handle.remove()
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    self._maybe_log(
                        logging.DEBUG,
                        "Failed to remove keyboard handle.",
                        handle_type=handle_type,
                        error=str(exc),
                    )

    def detect(self, timeout: float) -> str | None:
        detected_key: list[str | None] = [None]
        event = threading.Event()

        def _hook(event_obj: Any) -> bool | None:
            name = getattr(event_obj, "name", None)
            device = getattr(event_obj, "device", "keyboard")
            if device != "keyboard":
                return None
            if name in {None, "", "shift", "ctrl", "alt", "left shift", "right shift", "left ctrl", "right ctrl", "left alt", "right alt"}:
                return None
            detected_key[0] = str(name)
            event.set()
            return False

        hook = keyboard.hook(_hook)
        try:
            event.wait(timeout)
            return detected_key[0]
        finally:
            keyboard.unhook(hook)


try:
    _PYNPUT_SPEC = importlib.util.find_spec("pynput.keyboard")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _PYNPUT_SPEC = None
if _PYNPUT_SPEC is not None:
    pynput_keyboard = importlib.import_module("pynput.keyboard")
else:
    pynput_keyboard = None


class PynputHotkeyDriver(BaseHotkeyDriver):
    """Fallback driver that uses ``pynput`` listeners."""

    name = "pynput"

    def __init__(self, *, log: Callable[[int, str], None] | None = None) -> None:
        if pynput_keyboard is None:
            raise RuntimeError("pynput.keyboard is not available")
        super().__init__(log=log)
        self._listener: Any | None = None
        self._lock = threading.Lock()
        self._record_key: str | None = None
        self._agent_key: str | None = None
        self._record_mode: str = "toggle"
        self._callbacks: dict[str, Callable[[], None] | None] = {}
        self._record_active = False
        self._agent_active = False

    def register(
        self,
        *,
        record_key: str,
        agent_key: str | None,
        record_mode: str,
        callbacks: Mapping[str, Callable[[], None] | None],
        suppress: bool = False,
    ) -> None:
        del suppress  # Not supported by pynput, but kept for API compatibility
        with self._lock:
            self.unregister()
            self._record_key = self._normalize_key_name(record_key)
            self._agent_key = self._normalize_key_name(agent_key) if agent_key else None
            self._record_mode = record_mode
            self._callbacks = dict(callbacks)
            self._record_active = False
            self._agent_active = False

            listener = pynput_keyboard.Listener(
                on_press=self._handle_press,
                on_release=self._handle_release,
            )
            listener.start()
            if not listener.running:
                raise RuntimeError("pynput listener failed to start")
            self._listener = listener

    def unregister(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                self._maybe_log(
                    logging.DEBUG,
                    "Failed to stop pynput listener.",
                    error=str(exc),
                )

    def detect(self, timeout: float) -> str | None:
        if pynput_keyboard is None:
            raise RuntimeError("pynput.keyboard is not available")
        end_time = time.monotonic() + max(timeout, 0.0)
        ignored = {
            "shift",
            "ctrl",
            "alt",
            "shift_l",
            "shift_r",
            "ctrl_l",
            "ctrl_r",
            "alt_l",
            "alt_r",
        }
        with pynput_keyboard.Events() as events:
            while True:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return None
                event = events.get(remaining)
                if event is None or not hasattr(event, "key"):
                    return None
                if not isinstance(event, pynput_keyboard.Events.Press):
                    continue
                name = self._normalize_event_key(event.key)
                if not name or name in ignored:
                    continue
                return name.replace("_", " ")

    # --- Internal callbacks ---------------------------------------------

    def _handle_press(self, key: Any) -> None:
        key_name = self._normalize_event_key(key)
        if not key_name:
            return
        if self._record_key and key_name == self._record_key:
            if self._record_mode == "toggle":
                if self._record_active:
                    return
                self._record_active = True
                callback = self._callbacks.get("toggle")
                if callback is not None:
                    callback()
            else:
                if not self._record_active:
                    self._record_active = True
                    callback = self._callbacks.get("start")
                    if callback is not None:
                        callback()
        if self._agent_key and key_name == self._agent_key:
            if self._agent_active:
                return
            self._agent_active = True
            callback = self._callbacks.get("agent")
            if callback is not None:
                callback()

    def _handle_release(self, key: Any) -> None:
        key_name = self._normalize_event_key(key)
        if not key_name:
            return
        if self._record_key and key_name == self._record_key:
            if self._record_mode == "toggle":
                self._record_active = False
            else:
                if self._record_active:
                    self._record_active = False
                    callback = self._callbacks.get("stop")
                    if callback is not None:
                        callback()
        if self._agent_key and key_name == self._agent_key:
            self._agent_active = False

    @staticmethod
    def _normalize_key_name(name: str | None) -> str | None:
        if not name:
            return None
        normalized = name.strip().lower()
        alias_map = {
            "left shift": "shift_l",
            "right shift": "shift_r",
            "left ctrl": "ctrl_l",
            "right ctrl": "ctrl_r",
            "left control": "ctrl_l",
            "right control": "ctrl_r",
            "left alt": "alt_l",
            "right alt": "alt_r",
            "caps lock": "caps_lock",
        }
        normalized = alias_map.get(normalized, normalized)
        return normalized.replace(" ", "_")

    @staticmethod
    def _normalize_event_key(key: Any) -> str | None:
        if key is None:
            return None
        if isinstance(key, pynput_keyboard.KeyCode):
            char = key.char
            if char is None:
                return None
            return char.lower()
        if isinstance(key, pynput_keyboard.Key):
            value = key.name
            if value is None:
                return None
            return value.lower()
        return str(key).lower()


def build_available_drivers(*, log: Callable[[int, str], None] | None = None) -> list[BaseHotkeyDriver]:
    """Return the available driver instances ordered by priority."""

    drivers: list[BaseHotkeyDriver] = []
    try:
        drivers.append(KeyboardLibHotkeyDriver(log=log))
    except Exception as exc:  # pragma: no cover - unlikely during import
        LOGGER.debug("Keyboard driver unavailable: %s", exc)
    if pynput_keyboard is not None:
        try:
            drivers.append(PynputHotkeyDriver(log=log))
        except Exception as exc:  # pragma: no cover - initialization guard
            LOGGER.debug("Pynput driver unavailable: %s", exc)
    return drivers
