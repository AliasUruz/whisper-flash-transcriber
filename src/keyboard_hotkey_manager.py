# -*- coding: utf-8 -*-
import json
import shutil
import time
import threading
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import keyboard  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    keyboard = None  # type: ignore

from .config_manager import HOTKEY_CONFIG_FILE, LEGACY_HOTKEY_LOCATIONS
from .hotkey_normalization import _normalize_key_name
from .logging_utils import get_logger, join_thread_with_timeout, log_context

LOGGER = get_logger(
    "whisper_flash_transcriber.hotkeys",
    component="KeyboardHotkeyManager",
)


if TYPE_CHECKING:  # pragma: no cover - hints only
    from .hotkeys import BaseHotkeyDriver
else:  # pragma: no cover - fallback for optional dependency
    BaseHotkeyDriver = Any  # type: ignore[misc,assignment]


AUXILIARY_JOIN_TIMEOUT = 2.0

class KeyboardHotkeyManager:
    """Gerencia hotkeys usando drivers intercambiáveis."""

    def __init__(self, config_file: str | Path = HOTKEY_CONFIG_FILE):
        """
        Inicializa o gerenciador de hotkeys.

        Args:
            config_file (str): Caminho para o arquivo de configuração
        """
        path = Path(config_file).expanduser()
        self.config_file = str(path)
        self._config_path = path
        self.is_running = False
        self.callback_toggle = None
        self.callback_start = None
        self.callback_stop = None
        self.callback_agent = None
        self.record_key = "f3"  # Tecla padrão
        self.agent_key = "f4"  # Tecla padrão para comando agêntico
        self.record_mode = "toggle"  # Modo padrão
        self._debounce_window_ms = 0
        self._last_trigger_ts: dict[str, float] = {}
        self.hotkey_handlers: dict[str, list[Any]] = {}
        self._drivers: list[BaseHotkeyDriver] = []
        self._active_driver: BaseHotkeyDriver | None = None
        self._active_driver_index: int | None = None
        self._driver_lock = threading.Lock()
        self._driver_failures: list[dict[str, Any]] = []

        self._auxiliary_threads: dict[str, dict[str, Any]] = {}
        self._aux_threads_lock = threading.Lock()

        # Carregar configuração se existir
        self._load_config()
        self._initialize_drivers()

    def _log(
        self,
        level: int,
        message: str,
        *,
        event: str | None = None,
        exc_info: bool = False,
        **details: Any,
    ) -> None:
        payload = {key: value for key, value in details.items() if value is not None}
        LOGGER.log(
            level,
            log_context(message, event=event, details=payload or None),
            exc_info=exc_info,
        )

    def _driver_log(self, level: int, message: str, **details: Any) -> None:
        self._log(level, message, event="hotkeys.driver", **details)

    def _initialize_drivers(self) -> None:
        with self._driver_lock:
            try:
                from .hotkeys import build_available_drivers  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency missing
                self._drivers = []
                self._active_driver = None
                self._active_driver_index = None
                self._driver_failures = []
                self._log(
                    logging.WARNING,
                    "Hotkey drivers unavailable due to missing optional dependency.",
                    event="hotkeys.drivers_import_failed",
                    missing=str(exc),
                )
                return

            self._drivers = build_available_drivers(log=self._driver_log)
            self._active_driver = None
            self._active_driver_index = None
            self._driver_failures = []
        names = ", ".join(driver.name for driver in self._drivers) or "<none>"
        self._log(
            logging.INFO,
            "Hotkey drivers initialized.",
            event="hotkeys.drivers_initialized",
            drivers=names,
        )
        if not self._drivers:
            self._log(
                logging.ERROR,
                "No hotkey drivers available; registration will fail.",
                event="hotkeys.drivers_missing",
            )

    def _driver_entries(self) -> list[tuple[int, BaseHotkeyDriver]]:
        with self._driver_lock:
            return list(enumerate(self._drivers))

    def _should_process_event(self, event_key: str) -> bool:
        window_ms = max(0, int(self._debounce_window_ms))
        now = time.perf_counter()
        if window_ms <= 0:
            self._last_trigger_ts[event_key] = now
            return True

        last = self._last_trigger_ts.get(event_key)
        if last is None or (now - last) * 1000.0 >= window_ms:
            self._last_trigger_ts[event_key] = now
            return True
        return False

    def get_active_driver_name(self) -> str | None:
        with self._driver_lock:
            return self._active_driver.name if self._active_driver else None

    def is_using_fallback(self) -> bool:
        with self._driver_lock:
            return bool(self._active_driver_index and self._active_driver_index > 0)

    def describe_driver_state(self) -> dict[str, Any]:
        with self._driver_lock:
            active = self._active_driver.name if self._active_driver else None
            index = self._active_driver_index
            failures = list(self._driver_failures)
            available = [driver.name for driver in self._drivers]
        return {
            "active": active,
            "fallback_active": bool(index and index > 0),
            "available": available,
            "failures": failures,
        }

    def _load_config(self):
        """Load configuration from disk, creating the file with defaults when it is missing."""
        try:
            path = self._config_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                migrated = False
                for legacy in LEGACY_HOTKEY_LOCATIONS:
                    legacy_path = Path(legacy).expanduser()
                    try:
                        if legacy_path.resolve() == path.resolve():
                            continue
                    except Exception:
                        if str(legacy_path) == str(path):
                            continue
                    if not legacy_path.exists():
                        continue
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(legacy_path), str(path))
                        self._log(
                            logging.INFO,
                            "Legacy hotkey configuration migrated.",
                            event="hotkeys.config_migrated",
                            legacy_path=str(legacy_path),
                            path=str(path),
                        )
                        migrated = True
                        break
                    except Exception as exc:
                        self._log(
                            logging.WARNING,
                            "Failed to migrate legacy hotkey configuration.",
                            event="hotkeys.config_migrate_failed",
                            legacy_path=str(legacy_path),
                            path=str(path),
                            error=str(exc),
                        )
                        break
                if not migrated:
                    self._log(
                        logging.INFO,
                        "Hotkey configuration file missing; creating defaults.",
                        event="hotkeys.config_create_default",
                        path=str(path),
                    )
                    self._save_config()

            with path.open('r', encoding='utf-8') as f:
                config = json.load(f)
                self.record_key = (
                    _normalize_key_name(config.get('record_key', self.record_key))
                    or self.record_key
                )
                self.agent_key = (
                    _normalize_key_name(config.get('agent_key', self.agent_key))
                    or self.agent_key
                )
                self.record_mode = str(
                    config.get('record_mode', self.record_mode)
                ).lower()
                if self.record_mode not in {"toggle", "press"}:
                    self.record_mode = "toggle"
                self._log(
                    logging.INFO,
                    "Hotkey configuration loaded.",
                    event="hotkeys.config_loaded",
                    record_key=self.record_key,
                    agent_key=self.agent_key,
                    record_mode=self.record_mode,
                    path=self.config_file,
                )
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self._log(
                logging.ERROR,
                "Hotkey configuration invalid; rebuilding with defaults.",
                event="hotkeys.config_corrupted",
                error=str(e),
                path=self.config_file,
                exc_info=True,
            )
            try:
                self.record_key = "f3"
                self.agent_key = "f4"
                self.record_mode = "toggle"
                self._save_config()
            except Exception as write_error:
                self._log(
                    logging.ERROR,
                    "Failed to rebuild hotkey configuration after corruption.",
                    event="hotkeys.config_rebuild_failed",
                    error=str(write_error),
                    path=self.config_file,
                    exc_info=True,
                )
                raise
            raise RuntimeError(
                f"Hotkey configuration '{self.config_file}' was corrupted and has been reset. Please restart the application."
            ) from e
        except Exception as e:
            self._log(
                logging.ERROR,
                "Unexpected error while loading hotkey configuration.",
                event="hotkeys.config_unexpected_error",
                error=str(e),
                path=self.config_file,
                exc_info=True,
            )
            raise

    def _save_config(self):
        """Persist the current hotkey configuration to disk."""
        try:
            path = self._config_path
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized_record = _normalize_key_name(self.record_key) or "f3"
            normalized_agent = _normalize_key_name(self.agent_key) or "f4"
            self.record_key = normalized_record
            self.agent_key = normalized_agent
            self.record_mode = str(self.record_mode or "toggle").lower()
            if self.record_mode not in {"toggle", "press"}:
                self.record_mode = "toggle"
            config = {
                'record_key': normalized_record,
                'agent_key': normalized_agent,
                'record_mode': self.record_mode
            }
            with path.open('w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            self._log(
                logging.INFO,
                "Hotkey configuration saved.",
                event="hotkeys.config_save_success",
                record_key=self.record_key,
                agent_key=self.agent_key,
                record_mode=self.record_mode,
                path=self.config_file,
            )
        except Exception as e:
            self._log(
                logging.ERROR,
                "Failed to persist hotkey configuration.",
                event="hotkeys.config_save_failed",
                error=str(e),
                path=self.config_file,
                exc_info=True,
            )
            raise RuntimeError(
                f"Unable to persist hotkey configuration '{self.config_file}': {e}"
            ) from e

    def dry_run_register(
        self,
        hotkey: str | None = None,
        *,
        suppress: bool = False,
    ) -> dict[str, Any]:
        """Attempt to register a hotkey to validate permissions without keeping it active."""

        candidate = (hotkey or self.record_key or "f3").strip()
        entries = self._driver_entries()
        if not entries:
            self._log(
                logging.ERROR,
                "Dry-run aborted because no drivers are available.",
                event="hotkeys.diagnostics.no_driver",
                hotkey=candidate,
            )
            return {
                "ok": False,
                "message": "No hotkey driver is available for diagnostics.",
                "details": {"hotkey": candidate},
                "suggestion": "Reinstall the application dependencies and try again.",
                "fatal": True,
            }

        driver = entries[0][1]
        callbacks: dict[str, Callable[[], None] | None] = {"toggle": lambda: None}
        start_time = time.perf_counter()
        try:
            driver.register(
                record_key=candidate,
                agent_key=None,
                record_mode="toggle",
                callbacks=callbacks,
                suppress=suppress,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log(
                logging.INFO,
                "Dry-run hotkey registration succeeded.",
                event="hotkeys.diagnostics.success",
                hotkey=candidate,
                driver=driver.name,
                duration_ms=round(duration_ms, 3),
            )
            return {
                "ok": True,
                "message": f"Hotkey '{candidate}' can be registered via {driver.name}.",
                "details": {
                    "hotkey": candidate,
                    "duration_ms": duration_ms,
                    "driver": driver.name,
                    "suppress": suppress,
                },
                "suggestion": None,
                "fatal": False,
            }
        except PermissionError as exc:
            self._log(
                logging.ERROR,
                "Dry-run hotkey registration failed due to missing privileges.",
                event="hotkeys.diagnostics.permission_denied",
                hotkey=candidate,
                driver=driver.name,
                error=str(exc),
                exc_info=True,
            )
            return {
                "ok": False,
                "message": "The operating system denied permission to register global hotkeys.",
                "details": {
                    "hotkey": candidate,
                    "driver": driver.name,
                    "error": str(exc),
                },
                "suggestion": "Run the application with administrator privileges or allow keyboard hooks.",
                "fatal": True,
            }
        except Exception as exc:
            self._log(
                logging.ERROR,
                "Dry-run hotkey registration failed with an unexpected error.",
                event="hotkeys.diagnostics.failure",
                hotkey=candidate,
                driver=driver.name,
                error=str(exc),
                exc_info=True,
            )
            return {
                "ok": False,
                "message": "Global hotkey registration failed due to an unexpected error.",
                "details": {
                    "hotkey": candidate,
                    "driver": driver.name,
                    "error": str(exc),
                },
                "suggestion": "Ensure no other software blocks keyboard hooks and retry.",
                "fatal": True,
            }
        finally:
            try:
                driver.unregister()
            except Exception as cleanup_exc:
                self._log(
                    logging.WARNING,
                    "Failed to clean up dry-run hotkey registration.",
                    event="hotkeys.diagnostics.cleanup_failed",
                    hotkey=candidate,
                    driver=driver.name,
                    error=str(cleanup_exc),
                    exc_info=True,
                )

    def start(self):
        """Inicia o gerenciador de hotkeys."""
        if self.is_running:
            self._log(
                logging.WARNING,
                "KeyboardHotkeyManager already running; ignoring start request.",
                event="hotkeys.start_skipped",
            )
            return True

        try:
            # Registrar as hotkeys e verificar o resultado
            success = self._register_hotkeys()
            if not success:
                self._log(
                    logging.ERROR,
                    "Failed to register hotkeys during startup.",
                    event="hotkeys.start_register_failed",
                )
                self.stop()
                return False

            self.is_running = True
            self._log(
                logging.INFO,
                "KeyboardHotkeyManager started.",
                event="hotkeys.start_success",
                record_key=self.record_key,
                agent_key=self.agent_key,
                record_mode=self.record_mode,
                driver=self.get_active_driver_name(),
                fallback=self.is_using_fallback(),
            )
            return True
        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while starting KeyboardHotkeyManager.",
                event="hotkeys.start_error",
                error=str(e),
                exc_info=True,
            )
            self.stop()
            return False

    def stop(self):
        """Para o gerenciador de hotkeys."""
        self._stop_auxiliary_threads()
        # Sempre tente remover as hotkeys, mesmo que o estado esteja incorreto
        driver_name = self.get_active_driver_name()
        self._unregister_hotkeys()
        self.is_running = False
        self._log(
            logging.INFO,
            "KeyboardHotkeyManager stopped.",
            event="hotkeys.stop",
            driver=driver_name,
        )

    def update_config(self, record_key=None, agent_key=None, record_mode=None):
        """
        Atualiza a configuração do gerenciador de hotkeys.

        Args:
            record_key (str): Tecla de gravação
            agent_key (str): Tecla para comando agêntico
            record_mode (str): Modo de gravação ('toggle' ou 'press')
        """
        try:
            # Desregistrar hotkeys atuais
            was_running = self.is_running
            if was_running:
                self._unregister_hotkeys()

            # Atualizar valores
            if record_key is not None:
                normalized_record = _normalize_key_name(record_key)
                self.record_key = normalized_record or "f3"

            if agent_key is not None:
                normalized_agent = _normalize_key_name(agent_key)
                self.agent_key = normalized_agent or "f4"

            if record_mode is not None:
                normalized_mode = str(record_mode or "toggle").lower()
                if normalized_mode not in {"toggle", "press"}:
                    normalized_mode = "toggle"
                self.record_mode = normalized_mode

            # Salvar configuração
            self._save_config()

            # Registrar novas hotkeys se estava em execução
            if was_running:
                result = self._register_hotkeys()
                if not result:
                    self._log(
                        logging.ERROR,
                        "Failed to register hotkeys after applying update.",
                        event="hotkeys.update_register_failed",
                        record_key=self.record_key,
                        agent_key=self.agent_key,
                        record_mode=self.record_mode,
                    )
                    self.is_running = False
                    return False
                # Retomar o estado de execução se o registro foi bem-sucedido
                self.is_running = True

            self._log(
                logging.INFO,
                "Hotkey configuration updated.",
                event="hotkeys.update_success",
                record_key=self.record_key,
                agent_key=self.agent_key,
                record_mode=self.record_mode,
                driver=self.get_active_driver_name(),
                fallback=self.is_using_fallback(),
            )
            return True

        except Exception as e:
            self._log(
                logging.ERROR,
                "Failed to update hotkey configuration.",
                event="hotkeys.update_error",
                error=str(e),
                exc_info=True,
            )
            return False

    def set_callbacks(self, toggle=None, start=None, stop=None, agent=None):
        """
        Define os callbacks para os eventos de hotkey.

        Args:
            toggle (callable): Callback para o evento de toggle
            start (callable): Callback para o evento de início de gravação
            stop (callable): Callback para o evento de fim de gravação
            agent (callable): Callback para o comando agêntico
        """
        if toggle is not None:
            self.callback_toggle = toggle

        if start is not None:
            self.callback_start = start

        if stop is not None:
            self.callback_stop = stop

        if agent is not None:
            self.callback_agent = agent

    def set_debounce_window(self, milliseconds: int) -> None:
        """Adjust the debounce window for hotkey callbacks in milliseconds."""

        value = max(0, int(milliseconds))
        self._debounce_window_ms = value
        self._log(
            logging.INFO,
            "Updated hotkey debounce window.",
            event="hotkeys.debounce_window_updated",
            window_ms=value,
        )

    def describe_persistence_state(self) -> dict[str, object]:
        """Retorna informações de diagnóstico do arquivo de hotkeys."""

        path = Path(self.config_file).resolve()
        exists = path.is_file()
        try:
            size = path.stat().st_size if exists else 0
        except OSError:
            size = 0

        driver_state = self.describe_driver_state()

        return {
            "path": str(path),
            "exists": exists,
            "size": size,
            "record_key": self.record_key,
            "agent_key": self.agent_key,
            "record_mode": self.record_mode,
            "driver_state": driver_state,
        }

    def set_auxiliary_thread(
        self,
        name: str,
        *,
        thread: threading.Thread | None = None,
        stop_event: threading.Event | None = None,
        timeout: float | None = None,
        thread_label: str | None = None,
    ) -> None:
        """Register or update metadata for auxiliary hotkey service threads."""

        if not name:
            raise ValueError("Auxiliary thread name must be a non-empty string")

        with self._aux_threads_lock:
            entry = self._auxiliary_threads.setdefault(
                name,
                {
                    "thread": None,
                    "stop_event": None,
                    "timeout": AUXILIARY_JOIN_TIMEOUT,
                    "label": thread_label or name,
                },
            )

            if thread is not None:
                entry["thread"] = thread
                entry["label"] = thread_label or getattr(thread, "name", entry.get("label") or name)
            elif thread_label is not None:
                entry["label"] = thread_label

            if stop_event is not None:
                entry["stop_event"] = stop_event

            if timeout is not None:
                entry["timeout"] = timeout

            thread_alive = thread.is_alive() if isinstance(thread, threading.Thread) else None

        self._log(
            logging.DEBUG,
            "Auxiliary hotkey thread metadata updated.",
            event="hotkeys.aux_thread_updated",
            name=name,
            thread_alive=thread_alive,
            timeout=timeout,
        )

    def _stop_auxiliary_threads(self) -> None:
        """Signal and wait for auxiliary background threads to stop."""

        with self._aux_threads_lock:
            snapshot = [
                (
                    name,
                    payload.get("thread"),
                    payload.get("stop_event"),
                    float(payload.get("timeout", AUXILIARY_JOIN_TIMEOUT)),
                    payload.get("label") or name,
                )
                for name, payload in self._auxiliary_threads.items()
            ]

            for payload in self._auxiliary_threads.values():
                payload["thread"] = None

        for name, thread, stop_event, timeout, label in snapshot:
            if isinstance(stop_event, threading.Event):
                stop_event.set()

            if isinstance(thread, threading.Thread):
                join_thread_with_timeout(
                    thread,
                    timeout=timeout,
                    logger=LOGGER,
                    thread_name=label,
                    event_prefix=f"hotkeys.{name}",
                    details={"auxiliary": name},
                )

    def _store_hotkey_handle(self, handle_id: str, handle: Any) -> None:
        """Persist the handle returned by the ``keyboard`` library."""

        if handle is None:
            self._log(
                logging.WARNING,
                "Keyboard library returned a null handle; hook may not be active.",
                event="hotkeys.handle_missing",
                handle_id=handle_id,
            )
            return

        handles = self.hotkey_handlers.setdefault(handle_id, [])
        handles.append(handle)
        self._log(
            logging.DEBUG,
            "Stored hotkey handle.",
            event="hotkeys.handle_stored",
            handle_id=handle_id,
            total_handles=len(handles),
        )

    def _unregister_hotkeys(self) -> None:
        """Detach any hotkey hooks and clear driver state."""

        if self.hotkey_handlers:
            for handle_id, handles in list(self.hotkey_handlers.items()):
                for handle in handles:
                    if keyboard is None or handle is None:
                        continue
                    try:
                        keyboard.unhook(handle)
                    except Exception as exc:  # pragma: no cover - optional dependency errors
                        self._log(
                            logging.DEBUG,
                            "Failed to unhook keyboard handle.",
                            event="hotkeys.unhook_failed",
                            handle_id=handle_id,
                            error=str(exc),
                        )
                self.hotkey_handlers[handle_id] = []
            self.hotkey_handlers.clear()

        entries = self._driver_entries()
        for _, driver in entries:
            try:
                driver.unregister()
            except Exception as exc:
                self._log(
                    logging.DEBUG,
                    "Driver failed to unregister hotkeys.",
                    event="hotkeys.unregister_error",
                    driver=driver.name,
                    error=str(exc),
                )
        with self._driver_lock:
            self._active_driver = None
            self._active_driver_index = None

    def _register_hotkeys(self):
        """Registra as hotkeys no sistema."""
        self._log(
            logging.INFO,
            "Starting hotkey registration.",
            event="hotkeys.register_start",
            record_key=self.record_key,
            agent_key=self.agent_key,
            record_mode=self.record_mode,
        )

        self._unregister_hotkeys()
        callbacks: dict[str, Callable[[], None] | None] = {
            "toggle": self._on_toggle_key if self.record_mode == "toggle" else None,
            "start": self._on_press_key if self.record_mode != "toggle" else None,
            "stop": self._on_release_key if self.record_mode != "toggle" else None,
            "agent": self._on_agent_key if self.callback_agent else None,
        }

        entries = self._driver_entries()
        if not entries:
            self._log(
                logging.ERROR,
                "No drivers available to register hotkeys.",
                event="hotkeys.register_no_driver",
            )
            return False

        last_error: Exception | None = None
        driver_failures: list[dict[str, Any]] = []

        for index, driver in entries:
            try:
                driver.register(
                    record_key=self.record_key,
                    agent_key=self.agent_key,
                    record_mode=self.record_mode,
                    callbacks=callbacks,
                    suppress=False,
                )
            except PermissionError as exc:
                last_error = exc
                driver_failures.append(
                    {
                        "driver": driver.name,
                        "error": str(exc),
                        "reason": "permission_denied",
                    }
                )
                self._log(
                    logging.ERROR,
                    "Driver failed to register hotkeys due to permission error.",
                    event="hotkeys.driver_permission_error",
                    driver=driver.name,
                    error=str(exc),
                )
                try:
                    driver.unregister()
                except Exception:
                    self._log(
                        logging.DEBUG,
                        "Driver cleanup after failure raised an exception.",
                        event="hotkeys.driver_cleanup_error",
                        driver=driver.name,
                    )
                continue
            except Exception as exc:
                last_error = exc
                driver_failures.append(
                    {
                        "driver": driver.name,
                        "error": str(exc),
                        "reason": "exception",
                    }
                )
                self._log(
                    logging.ERROR,
                    "Driver raised an unexpected error during registration.",
                    event="hotkeys.driver_register_error",
                    driver=driver.name,
                    error=str(exc),
                    exc_info=True,
                )
                try:
                    driver.unregister()
                except Exception:
                    self._log(
                        logging.DEBUG,
                        "Driver cleanup after failure raised an exception.",
                        event="hotkeys.driver_cleanup_error",
                        driver=driver.name,
                    )
                continue

            with self._driver_lock:
                self._active_driver = driver
                self._active_driver_index = index
                self._driver_failures = driver_failures

            self._log(
                logging.INFO,
                "Hotkeys registered using driver.",
                event="hotkeys.register_success",
                driver=driver.name,
                fallback=index > 0,
            )

            if index > 0:
                self._log(
                    logging.WARNING,
                    "Fallback hotkey driver activated.",
                    event="hotkeys.driver_fallback",
                    driver=driver.name,
                )
            return True

        with self._driver_lock:
            self._active_driver = None
            self._active_driver_index = None
            self._driver_failures = driver_failures

        self._log(
            logging.ERROR,
            "All hotkey drivers failed to register.",
            event="hotkeys.register_failure",
            last_error=str(last_error) if last_error else None,
        )
        return False

    def _on_toggle_key(self):
        """Handler para a tecla de toggle."""
        try:
            if self.callback_toggle and self._should_process_event("toggle"):
                threading.Thread(target=self.callback_toggle, daemon=True, name="ToggleCallback").start()
                self._log(
                    logging.INFO,
                    "Toggle callback invoked.",
                    event="hotkeys.callback_toggle_invoked",
                )
        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while invoking toggle callback.",
                event="hotkeys.callback_toggle_error",
                error=str(e),
                exc_info=True,
            )

    def _on_press_key(self):
        """Handler para a tecla de press."""
        try:
            if self.callback_start and self._should_process_event("press"):
                threading.Thread(target=self.callback_start, daemon=True, name="StartCallback").start()
                self._log(
                    logging.INFO,
                    "Start callback invoked.",
                    event="hotkeys.callback_start_invoked",
                )
        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while invoking start callback.",
                event="hotkeys.callback_start_error",
                error=str(e),
                exc_info=True,
            )

    def _on_agent_key(self):
        """Handler for the agent command hotkey."""
        try:
            if self.callback_agent and self._should_process_event("agent"):
                threading.Thread(target=self.callback_agent, daemon=True, name="AgentCallback").start()
                self._log(
                    logging.INFO,
                    "Agent callback invoked.",
                    event="hotkeys.callback_agent_invoked",
                )
        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while invoking agent callback.",
                event="hotkeys.callback_agent_error",
                error=str(e),
                exc_info=True,
            )

    def _on_release_key(self):
        """Handler para o evento de soltar a tecla no modo press."""
        try:
            if self.callback_stop:
                threading.Thread(target=self.callback_stop, daemon=True, name="StopCallback").start()
                self._log(
                    logging.INFO,
                    "Stop callback invoked (release key).",
                    event="hotkeys.callback_stop_invoked",
                )
        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while invoking stop callback.",
                event="hotkeys.callback_stop_error",
                error=str(e),
                exc_info=True,
            )

    def restart(self):
        """Reinicia o gerenciador de hotkeys."""
        self._log(
            logging.INFO,
            "Restarting KeyboardHotkeyManager...",
            event="hotkeys.restart_begin",
        )
        self.stop()
        time.sleep(0.5)  # Pequeno delay para garantir que tudo foi encerrado

        time.sleep(0.5)  # Delay adicional para garantir limpeza completa

        # Iniciar novamente
        result = self.start()
        if result:
            self._log(
                logging.INFO,
                "KeyboardHotkeyManager restarted successfully.",
                event="hotkeys.restart_success",
            )
        else:
            self._log(
                logging.ERROR,
                "Failed to restart KeyboardHotkeyManager.",
                event="hotkeys.restart_failed",
            )

        return result

    def detect_key(self, timeout=5.0):
        """Detecta uma tecla pressionada."""

        self._log(
            logging.INFO,
            "Starting key detection.",
            event="hotkeys.detect_start",
            timeout_seconds=timeout,
        )

        driver: BaseHotkeyDriver | None = None
        with self._driver_lock:
            if self._active_driver is not None:
                driver = self._active_driver
        if driver is None:
            entries = self._driver_entries()
            if entries:
                driver = entries[0][1]

        if driver is None:
            self._log(
                logging.ERROR,
                "No hotkey driver available for detection.",
                event="hotkeys.detect_no_driver",
                timeout_seconds=timeout,
            )
            return None

        try:
            detected_key = driver.detect(timeout=float(timeout))
        except Exception as exc:
            self._log(
                logging.ERROR,
                "Error while detecting key.",
                event="hotkeys.detect_error",
                error=str(exc),
                timeout_seconds=timeout,
                driver=driver.name,
                exc_info=True,
            )
            return None

        if detected_key:
            self._log(
                logging.INFO,
                "Key detection completed with value.",
                event="hotkeys.detect_success",
                key=detected_key,
                timeout_seconds=timeout,
                driver=driver.name,
            )
            return detected_key

        self._log(
            logging.INFO,
            "Key detection completed without input.",
            event="hotkeys.detect_timeout",
            timeout_seconds=timeout,
            driver=driver.name,
        )
        return None

    def detect_single_key(self, timeout=5.0):
        """Mantida para compatibilidade: delega para ``detect_key``."""
        self._log(
            logging.DEBUG,
            "detect_single_key delegated to detect_key.",
            event="hotkeys.detect_single_delegated",
            timeout_seconds=timeout,
        )
        return self.detect_key(timeout=timeout)
