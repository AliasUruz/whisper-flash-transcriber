# -*- coding: utf-8 -*-
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

try:  # pragma: no cover - optional dependency
    import keyboard  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    keyboard = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import keyboard  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    keyboard = None  # type: ignore[assignment]

try:
    import keyboard  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    class _KeyboardPlaceholder:
        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(
                "keyboard library is required for hotkey management but is not installed"
            )

    keyboard = _KeyboardPlaceholder()  # type: ignore[assignment]

try:  # Optional dependency used only for direct keyboard hook cleanup in tests
    import keyboard  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - optional in headless environments
    keyboard = None  # type: ignore[assignment]

from .config_manager import HOTKEY_CONFIG_FILE, LEGACY_HOTKEY_LOCATIONS
from .hotkey_normalization import _normalize_key_name
from .hotkeys import BaseHotkeyDriver, build_available_drivers
from .logging_utils import get_logger, join_thread_with_timeout, log_context

LOGGER = get_logger(
    "whisper_flash_transcriber.hotkeys",
    component="KeyboardHotkeyManager",
)

class KeyboardHotkeyManager:
    """
    Gerencia hotkeys usando a biblioteca keyboard.
    Esta classe oferece uma solução mais simples para o gerenciamento de hotkeys.
    """

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
        self._drivers: list[BaseHotkeyDriver] = []
        self._active_driver: BaseHotkeyDriver | None = None
        self._active_driver_index: int | None = None
        self._driver_lock = threading.Lock()
        self._driver_failures: list[dict[str, Any]] = []
        self._available_driver_names: list[str] = self._probe_available_driver_names()
        self._active_driver_name: str | None = None
        self._fallback_active: bool = False
        self.hotkey_handlers: dict[str, list[Any]] = {}
        self._debounce_window_seconds: float = 0.0
        self._last_event_timestamps: dict[str, float] = {}
        self._last_trigger_ts = self._last_event_timestamps

        self._auxiliary_threads: dict[str, dict[str, Any]] = {}
        self._aux_threads_lock = threading.Lock()

        # Carregar configuração se existir
        self._load_config()

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

    def _probe_available_driver_names(self) -> list[str]:
        try:
            drivers = build_available_drivers(
                log=lambda lvl, msg, **fields: self._log(
                    lvl,
                    msg,
                    event="hotkeys.driver_probe",
                    **fields,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive probe
            self._log(
                logging.DEBUG,
                "Failed to probe available hotkey drivers.",
                event="hotkeys.driver_probe_failed",
                error=str(exc),
            )
            return []

        names: list[str] = []
        for driver in drivers:
            try:
                names.append(driver.name)
            except Exception as driver_exc:  # pragma: no cover - defensive guard
                self._log(
                    logging.DEBUG,
                    "Failed to resolve hotkey driver name during probe.",
                    event="hotkeys.driver_probe_name_failed",
                    error=str(driver_exc),
                )

        try:
            has_keyboard_driver = bool(getattr(keyboard, "add_hotkey", None))
        except Exception:  # pragma: no cover - placeholder raised
            has_keyboard_driver = False
        if has_keyboard_driver and "keyboard" not in names:
            names.insert(0, "keyboard")
        return names

    def _set_driver_state(self, *, active_name: str | None, fallback: bool) -> None:
        with self._driver_lock:
            self._active_driver_name = active_name
            self._fallback_active = fallback
            self._active_driver_index = 0 if active_name is not None else None
            self._active_driver = None

    def _determine_fallback(self, active_name: str | None) -> bool:
        if active_name is None:
            return False
        if not self._available_driver_names:
            return False
        return self._available_driver_names[0] != active_name

    def _remember_driver_failure(self, *, reason: str, error: str | None = None) -> None:
        entry: dict[str, Any] = {"reason": reason}
        if error:
            entry["error"] = error
        with self._driver_lock:
            self._driver_failures.append(entry)
            if len(self._driver_failures) > 10:
                self._driver_failures = self._driver_failures[-10:]

    def _resolve_primary_driver_name(self) -> str | None:
        if self._available_driver_names:
            return self._available_driver_names[0]
        return None

    def get_active_driver_name(self) -> str | None:
        with self._driver_lock:
            return self._active_driver_name

    def is_using_fallback(self) -> bool:
        with self._driver_lock:
            return self._fallback_active

    def describe_driver_state(self) -> dict[str, Any]:
        with self._driver_lock:
            return {
                "active": self._active_driver_name,
                "fallback_active": self._fallback_active,
                "available": list(self._available_driver_names),
                "failures": [dict(item) for item in self._driver_failures],
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
                self.record_key = config.get('record_key', self.record_key)
                self.agent_key = config.get('agent_key', self.agent_key)
                self.record_mode = config.get('record_mode', self.record_mode)
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
            config = {
                'record_key': self.record_key,
                'agent_key': self.agent_key,
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
        registration_id = None
        start_time = time.perf_counter()
        try:
            registration_id = keyboard.add_hotkey(
                candidate,
                lambda: None,
                suppress=suppress,
                trigger_on_release=False,
            )
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._log(
                logging.INFO,
                "Dry-run hotkey registration succeeded.",
                event="hotkeys.diagnostics.success",
                hotkey=candidate,
                duration_ms=round(duration_ms, 3),
            )
            return {
                "ok": True,
                "message": f"Hotkey '{candidate}' can be registered.",
                "details": {
                    "hotkey": candidate,
                    "duration_ms": duration_ms,
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
                error=str(exc),
                exc_info=True,
            )
            return {
                "ok": False,
                "message": "The operating system denied permission to register global hotkeys.",
                "details": {
                    "hotkey": candidate,
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
                error=str(exc),
                exc_info=True,
            )
            return {
                "ok": False,
                "message": "Global hotkey registration failed due to an unexpected error.",
                "details": {
                    "hotkey": candidate,
                    "error": str(exc),
                },
                "suggestion": "Ensure no other software blocks keyboard hooks and retry.",
                "fatal": True,
            }
        finally:
            if registration_id is not None:
                try:
                    keyboard.remove_hotkey(registration_id)
                except Exception as cleanup_exc:
                    self._log(
                        logging.WARNING,
                        "Failed to clean up dry-run hotkey registration.",
                        event="hotkeys.diagnostics.cleanup_failed",
                        hotkey=candidate,
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
            self._available_driver_names = self._probe_available_driver_names()
            # Registrar as hotkeys e verificar o resultado
            success = self._register_hotkeys()
            if not success:
                self._remember_driver_failure(
                    reason="registration_failed",
                    error="Unable to register hotkeys with available drivers.",
                )
                self._set_driver_state(active_name=None, fallback=False)
                self._log(
                    logging.ERROR,
                    "Failed to register hotkeys during startup.",
                    event="hotkeys.start_register_failed",
                )
                self.stop()
                return False

            driver_name = self._resolve_primary_driver_name() or "keyboard"
            fallback_active = self._determine_fallback(driver_name)
            self._set_driver_state(active_name=driver_name, fallback=fallback_active)
            with self._driver_lock:
                self._driver_failures.clear()
            self.is_running = True
            self._log(
                logging.INFO,
                "KeyboardHotkeyManager started.",
                event="hotkeys.start_success",
                record_key=self.record_key,
                agent_key=self.agent_key,
                record_mode=self.record_mode,
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
        if keyboard is not None and self.hotkey_handlers:
            for handles in list(self.hotkey_handlers.values()):
                for handle in handles:
                    try:
                        keyboard.unhook(handle)
                    except Exception:  # pragma: no cover - defensive cleanup
                        self._log(
                            logging.DEBUG,
                            "Failed to unhook keyboard handle during stop.",
                            event="hotkeys.stop_unhook_failed",
                            handle=str(handle),
                        )
            self.hotkey_handlers.clear()
        # Sempre tente remover as hotkeys, mesmo que o estado esteja incorreto
        self._unregister_hotkeys()
        self._set_driver_state(active_name=None, fallback=False)
        self.is_running = False
        self._log(
            logging.INFO,
            "KeyboardHotkeyManager stopped.",
            event="hotkeys.stop",
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
                self.record_key = record_key.lower()

            if agent_key is not None:
                self.agent_key = agent_key.lower()

            if record_mode is not None:
                self.record_mode = record_mode

            # Salvar configuração
            self._save_config()

            # Registrar novas hotkeys se estava em execução
            if was_running:
                self._available_driver_names = self._probe_available_driver_names()
                result = self._register_hotkeys()
                if not result:
                    self._remember_driver_failure(
                        reason="registration_failed_after_update",
                        error="Unable to register hotkeys after configuration update.",
                    )
                    self._set_driver_state(active_name=None, fallback=False)
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
                driver_name = self._resolve_primary_driver_name() or "keyboard"
                fallback_active = self._determine_fallback(driver_name)
                self._set_driver_state(active_name=driver_name, fallback=fallback_active)
                with self._driver_lock:
                    self._driver_failures.clear()
                self.is_running = True

            self._log(
                logging.INFO,
                "Hotkey configuration updated.",
                event="hotkeys.update_success",
                record_key=self.record_key,
                agent_key=self.agent_key,
                record_mode=self.record_mode,
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

    def set_debounce_window(self, debounce_ms: float | int) -> None:
        """Configura o período de debounce entre eventos de hotkey."""

        try:
            window_seconds = max(0.0, float(debounce_ms) / 1000.0)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            self._log(
                logging.WARNING,
                "Invalid debounce window provided; ignoring update.",
                event="hotkeys.debounce_invalid",
                error=str(exc),
            )
            return

        self._debounce_window_seconds = window_seconds
        self._log(
            logging.INFO,
            "Hotkey debounce window updated.",
            event="hotkeys.debounce_updated",
            debounce_ms=int(window_seconds * 1000),
        )

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

        return {
            "path": str(path),
            "exists": exists,
            "size": size,
            "record_key": self.record_key,
            "agent_key": self.agent_key,
            "record_mode": self.record_mode,
        }

    def _store_hotkey_handle(self, handle_id, handle):
        """Guarda o handle retornado pela biblioteca ``keyboard``."""
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

    def _unregister_hotkeys(self) -> None:
        """Remove todas as hotkeys registradas pelos drivers ativos."""

        entries = self._driver_entries()
        for _, driver in entries:
            try:
                driver.unregister()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                self._log(
                    logging.DEBUG,
                    "Hotkey callback ignored due to debounce window.",
                    event="hotkeys.debounce_skipped",
                    handle_id=handle_id,
                    elapsed_ms=round(elapsed_ms, 3),
                    debounce_ms=window_ms,
                )
                return False

        self._last_trigger_ts[handle_id] = now
        return True

    def _register_hotkeys(self):
        """Registra as hotkeys no sistema."""
        try:
            self._log(
                logging.INFO,
                "Starting hotkey registration.",
                event="hotkeys.register_start",
                record_key=self.record_key,
                agent_key=self.agent_key,
                record_mode=self.record_mode,
            )

            # Desregistrar hotkeys existentes para evitar duplicação
            self._unregister_hotkeys()
            self._last_trigger_ts.clear()

            # Registrar a tecla de gravação
            self._log(
                logging.INFO,
                "Registering recording hotkey.",
                event="hotkeys.register_record",
                record_key=self.record_key,
                mode=self.record_mode,
            )

            # Definir o handler para a tecla de gravação
            if self.record_mode == "toggle":
                handler = self._on_toggle_key
            else:
                # Para o modo press, registramos dois handlers: um para pressionar e outro para soltar
                handler = self._on_press_key
                # Registrar handler para soltar a tecla no modo press
                if self.record_mode == "press":
                    try:
                        release_handle = keyboard.on_release_key(
                            self.record_key,
                            lambda _: self._on_release_key(),
                            suppress=False,
                        )
                        self._store_hotkey_handle(
                            f"{self.record_key}:release",
                            release_handle,
                        )
                    except OSError as e:
                        self._log(
                            logging.ERROR,
                            "OS error while registering release hotkey.",
                            event="hotkeys.register_release_os_error",
                            record_key=self.record_key,
                            error=str(e),
                        )
                        return False
                    except Exception as e:
                        self._log(
                            logging.ERROR,
                            "Unexpected error while registering release hotkey.",
                            event="hotkeys.register_release_error",
                            record_key=self.record_key,
                            error=str(e),
                            exc_info=True,
                        )
                        return False
                    self._log(
                        logging.INFO,
                        "Release handler registered for record hotkey.",
                        event="hotkeys.register_release_success",
                        record_key=self.record_key,
                    )

            # Usar on_press_key em vez de add_hotkey para maior confiabilidade
            try:
                press_handle = keyboard.on_press_key(
                    self.record_key,
                    lambda _: handler(),
                    suppress=True,
                )
                self._store_hotkey_handle(
                    f"{self.record_key}:press",
                    press_handle,
                )
            except OSError as e:
                self._log(
                    logging.ERROR,
                    "OS error while registering record hotkey.",
                    event="hotkeys.register_press_os_error",
                    record_key=self.record_key,
                    error=str(e),
                )
                return False
            except Exception as e:
                self._log(
                    logging.ERROR,
                    "Error while registering record hotkey.",
                    event="hotkeys.register_press_error",
                    record_key=self.record_key,
                    error=str(e),
                    exc_info=True,
                )
                return False
            self._log(
                logging.INFO,
                "Recording hotkey registered.",
                event="hotkeys.register_press_success",
                record_key=self.record_key,
            )

            # Registrar a tecla de recarga
            self._log(
                logging.INFO,
                "Registering agent hotkey.",
                event="hotkeys.register_agent",
                agent_key=self.agent_key,
            )
            try:
                agent_handle = keyboard.on_press_key(
                    self.agent_key,
                    lambda _: self._on_agent_key(),
                    suppress=False,
                )
                self._store_hotkey_handle(
                    f"{self.agent_key}:press",
                    agent_handle,
                )
            except OSError as e:
                self._log(
                    logging.ERROR,
                    "OS error while registering agent hotkey.",
                    event="hotkeys.register_agent_os_error",
                    agent_key=self.agent_key,
                    error=str(e),
                )
                return False
            except Exception as e:
                self._log(
                    logging.ERROR,
                    "Unexpected error while registering agent hotkey.",
                    event="hotkeys.register_agent_error",
                    agent_key=self.agent_key,
                    error=str(e),
                    exc_info=True,
                )
                return False
            self._log(
                logging.INFO,
                "Agent hotkey registered.",
                event="hotkeys.register_agent_success",
                agent_key=self.agent_key,
            )

            self._log(
                logging.INFO,
                "Hotkeys registered successfully.",
                event="hotkeys.register_complete",
                record_key=self.record_key,
                agent_key=self.agent_key,
            )

            if index > 0:
                self._log(
                    logging.WARNING,
                    "Fallback hotkey driver activated.",
                    event="hotkeys.driver_fallback",
                    driver=driver.name,
            )
            return True

        except Exception as e:
            self._log(
                logging.ERROR,
                "Unexpected error while registering hotkeys.",
                event="hotkeys.register_unexpected_error",
                error=str(e),
                exc_info=True,
            )
            return False

    def _unregister_hotkeys(self):
        """Desregistra as hotkeys do sistema."""

        drivers: list[BaseHotkeyDriver] = []
        active_driver: BaseHotkeyDriver | None = None
        active_index: int | None = None

        driver_lock = getattr(self, "_driver_lock", None)
        if driver_lock is not None:
            try:
                lock_cm = driver_lock
            except Exception:
                lock_cm = None
        else:
            lock_cm = None

        if lock_cm is not None:
            with lock_cm:
                active_driver = getattr(self, "_active_driver", None)
                active_index = getattr(self, "_active_driver_index", None)
                drivers = list(getattr(self, "_drivers", []))
                try:
                    self._active_driver = None
                except Exception:
                    pass
                try:
                    self._active_driver_index = None
                except Exception:
                    pass
        else:
            active_driver = getattr(self, "_active_driver", None)
            active_index = getattr(self, "_active_driver_index", None)
            if active_driver is not None:
                drivers = [active_driver]
            try:
                self._active_driver = None
            except Exception:
                pass
            try:
                self._active_driver_index = None
            except Exception:
                pass

        def _log_driver_failure(driver_obj: BaseHotkeyDriver, error: Exception) -> None:
            self._log(
                logging.DEBUG,
                "Failed to unregister hotkey driver.",
                event="hotkeys.driver_unregister_failed",
                driver=getattr(driver_obj, "name", type(driver_obj).__name__),
                error=str(error),
            )

        if active_driver is not None:
            try:
                active_driver.unregister()
            except Exception as exc:
                _log_driver_failure(active_driver, exc)
            else:
                self._log(
                    logging.DEBUG,
                    "Active hotkey driver unregistered.",
                    event="hotkeys.driver_unregister_success",
                    driver=getattr(active_driver, "name", type(active_driver).__name__),
                    driver_index=active_index,
                )

        for driver in drivers:
            if driver is None or driver is active_driver:
                continue
            try:
                driver.unregister()
            except Exception as exc:
                _log_driver_failure(driver, exc)

        try:
            drivers_to_cleanup: list[BaseHotkeyDriver] = []
            with self._driver_lock:
                if self._active_driver is not None:
                    drivers_to_cleanup.append(self._active_driver)
                for driver in self._drivers:
                    if driver is not None and driver not in drivers_to_cleanup:
                        drivers_to_cleanup.append(driver)
                self._active_driver = None
                self._active_driver_index = None

            for driver in drivers_to_cleanup:
                driver_name = getattr(driver, "name", driver.__class__.__name__)
                try:
                    driver.unregister()
                    self._log(
                        logging.DEBUG,
                        "Hotkey driver unregistered.",
                        event="hotkeys.driver_unregister_success",
                        driver=driver_name,
                    )
                except Exception as exc:  # pragma: no cover - defensive cleanup
                    self._log(
                        logging.ERROR,
                        "Error while unregistering hotkey driver.",
                        event="hotkeys.driver_unregister_error",
                        driver=driver_name,
                        error=str(exc),
                        exc_info=True,
                    )

            for handle_id, handles in list(self.hotkey_handlers.items()):
                for handle in handles:
                    try:
                        keyboard.unhook(handle)
                        self._log(
                            logging.DEBUG,
                            "Hotkey handle removed.",
                            event="hotkeys.unregister_handle_removed",
                            handle_id=handle_id,
                        )
                    except (KeyError, ValueError):
                        self._log(
                            logging.WARNING,
                            "Hotkey handle already removed or invalid; skipping.",
                            event="hotkeys.unregister_handle_missing",
                            handle_id=handle_id,
                        )
                    except Exception as e:
                        self._log(
                            logging.ERROR,
                            "Error while removing hotkey hook.",
                            event="hotkeys.unregister_handle_error",
                            handle_id=handle_id,
                            error=str(e),
                            exc_info=True,
                        )
                self.hotkey_handlers[handle_id] = []

            self.hotkey_handlers.clear()

            self._log(
                logging.INFO,
                "Hotkeys unregistered successfully.",
                event="hotkeys.unregister_success",
            )
            self.is_running = False
            self._last_trigger_ts.clear()

        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while unregistering hotkeys.",
                event="hotkeys.unregister_error",
                error=str(e),
                exc_info=True,
            )

    def _should_process_event(self, event: str) -> bool:
        """Return True when the event should be processed respecting debounce."""

        window_ms = max(0, int(self._debounce_window_ms))
        now = time.perf_counter()
        if window_ms <= 0:
            self._last_trigger_ts[event] = now
            return True
        last = self._last_trigger_ts.get(event)
        if last is None or (now - last) * 1000 >= window_ms:
            self._last_trigger_ts[event] = now
            return True
        return False

    def _should_process_event(self, event_type: str) -> bool:
        """Determine whether a hotkey event should trigger callbacks."""

        window = self._debounce_window_seconds
        now = time.perf_counter()
        last = self._last_event_timestamps.get(event_type)
        if window <= 0 or last is None or (now - last) >= window:
            self._last_event_timestamps[event_type] = now
            return True
        self._log(
            logging.DEBUG,
            "Hotkey event suppressed by debounce window.",
            event="hotkeys.debounce_suppressed",
            event_type=event_type,
            debounce_ms=int(window * 1000),
            elapsed_ms=int((now - last) * 1000),
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
        """
        Detecta uma tecla pressionada.

        Args:
            timeout (float): Tempo máximo de espera em segundos

        Returns:
            str: A tecla detectada ou None se nenhuma tecla for detectada
        """
        try:
            self._log(
                logging.INFO,
                "Starting key detection.",
                event="hotkeys.detect_start",
                timeout_seconds=timeout,
            )

            # Variáveis para armazenar a tecla detectada
            detected_key = [None]
            key_detected = threading.Event()

            # Função para capturar a tecla
            def on_key_event(e):
                # Ignorar eventos que não sejam do teclado (ex.: cliques do mouse)
                if getattr(e, 'device', 'keyboard') != 'keyboard':
                    return
                # Ignorar teclas especiais como shift, ctrl, alt
                if e.name in ['shift', 'ctrl', 'alt', 'left shift', 'right shift', 'left ctrl', 'right ctrl', 'left alt', 'right alt']:
                    return

                self._log(
                    logging.INFO,
                    "Key detected during capture.",
                    event="hotkeys.detect_key_seen",
                    key=e.name,
                )
                detected_key[0] = e.name
                key_detected.set()
                return False  # Parar de escutar

            # Registrar o hook
            hook = keyboard.hook(on_key_event)

            try:
                # Aguardar até que uma tecla seja detectada ou o timeout expire
                key_detected.wait(timeout)

                # Retornar a tecla detectada
                if detected_key[0]:
                    self._log(
                        logging.INFO,
                        "Key detection completed with value.",
                        event="hotkeys.detect_success",
                        key=detected_key[0],
                        timeout_seconds=timeout,
                    )
                    return detected_key[0]
                else:
                    self._log(
                        logging.INFO,
                        "Key detection completed without input.",
                        event="hotkeys.detect_timeout",
                        timeout_seconds=timeout,
                    )
                    return None
            finally:
                # Remover o hook
                keyboard.unhook(hook)

        except Exception as e:
            self._log(
                logging.ERROR,
                "Error while detecting key.",
                event="hotkeys.detect_error",
                error=str(e),
                timeout_seconds=timeout,
                exc_info=True,
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
