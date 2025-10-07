# -*- coding: utf-8 -*-
import json
import shutil
import time
import threading
import logging
from pathlib import Path
from typing import Any

import keyboard

from .config_manager import HOTKEY_CONFIG_FILE, LEGACY_HOTKEY_LOCATIONS
from .logging_utils import get_logger, log_context
from .hotkey_normalization import _normalize_key_name

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
        self.hotkey_handlers = {}
        self.debounce_window_ms: float = 200.0
        self._last_trigger_ts: dict[str, float] = {}

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

        candidate = _normalize_key_name(hotkey or self.record_key or "f3") or "f3"
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
        # Sempre tente remover as hotkeys, mesmo que o estado esteja incorreto
        self._unregister_hotkeys()
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
        normalized_id = handle_id
        if isinstance(handle_id, str):
            key_part, separator, suffix = handle_id.partition(":")
            normalized_key = _normalize_key_name(key_part)
            if normalized_key:
                normalized_id = normalized_key + (separator + suffix if separator else "")
        self.hotkey_handlers.setdefault(normalized_id, []).append(handle)

    def set_debounce_window(self, debounce_ms: float | int | None) -> None:
        """Configure a janela de debounce em milissegundos."""

        previous = self.debounce_window_ms
        try:
            if debounce_ms is None:
                candidate = 0.0
            else:
                candidate = float(debounce_ms)
        except (TypeError, ValueError):
            self._log(
                logging.WARNING,
                "Invalid debounce value provided; keeping previous window.",
                event="hotkeys.debounce_invalid",
                supplied=str(debounce_ms),
                previous_ms=previous,
            )
            return

        candidate = max(candidate, 0.0)
        if candidate == previous:
            return

        self.debounce_window_ms = candidate
        self._last_trigger_ts.clear()
        self._log(
            logging.INFO,
            "Hotkey debounce window updated.",
            event="hotkeys.debounce_updated",
            debounce_ms=candidate,
        )

    def _should_process_event(self, handle_id: str) -> bool:
        """Return ``True`` when the callback tied to ``handle_id`` must run."""

        window_ms = max(self.debounce_window_ms, 0.0)
        now = time.perf_counter()
        if window_ms <= 0.0:
            self._last_trigger_ts[handle_id] = now
            return True

        last_seen = self._last_trigger_ts.get(handle_id)
        if last_seen is not None:
            elapsed_ms = (now - last_seen) * 1000.0
            if elapsed_ms < window_ms:
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
            record_mode = str(self.record_mode or "toggle").lower()
            if record_mode not in {"toggle", "press"}:
                record_mode = "toggle"
            record_key = _normalize_key_name(self.record_key) or "f3"
            agent_key = _normalize_key_name(self.agent_key) or "f4"
            self.record_mode = record_mode
            self.record_key = record_key
            self.agent_key = agent_key

            self._log(
                logging.INFO,
                "Starting hotkey registration.",
                event="hotkeys.register_start",
                record_key=record_key,
                agent_key=agent_key,
                record_mode=record_mode,
            )

            # Desregistrar hotkeys existentes para evitar duplicação
            self._unregister_hotkeys()
            self._last_trigger_ts.clear()

            # Registrar a tecla de gravação
            self._log(
                logging.INFO,
                "Registering recording hotkey.",
                event="hotkeys.register_record",
                record_key=record_key,
                mode=record_mode,
            )

            # Definir o handler para a tecla de gravação
            if self.record_mode == "toggle":
                handler = self._on_toggle_key
            else:
                # Para o modo press, registramos dois handlers: um para pressionar e outro para soltar
                handler = self._on_press_key
                # Registrar handler para soltar a tecla no modo press
                if record_mode == "press":
                    try:
                        release_handle = keyboard.on_release_key(
                            record_key,
                            lambda _: self._on_release_key(),
                            suppress=False,
                        )
                        self._store_hotkey_handle(
                            f"{record_key}:release",
                            release_handle,
                        )
                    except OSError as e:
                        self._log(
                            logging.ERROR,
                            "OS error while registering release hotkey.",
                            event="hotkeys.register_release_os_error",
                            record_key=record_key,
                            error=str(e),
                        )
                        return False
                    except Exception as e:
                        self._log(
                            logging.ERROR,
                            "Unexpected error while registering release hotkey.",
                            event="hotkeys.register_release_error",
                            record_key=record_key,
                            error=str(e),
                            exc_info=True,
                        )
                        return False
                    self._log(
                        logging.INFO,
                        "Release handler registered for record hotkey.",
                        event="hotkeys.register_release_success",
                        record_key=record_key,
                    )

            # Usar on_press_key em vez de add_hotkey para maior confiabilidade
            try:
                press_handle = keyboard.on_press_key(
                    record_key,
                    lambda _: handler(),
                    suppress=True,
                )
                self._store_hotkey_handle(
                    f"{record_key}:press",
                    press_handle,
                )
            except OSError as e:
                self._log(
                    logging.ERROR,
                    "OS error while registering record hotkey.",
                    event="hotkeys.register_press_os_error",
                    record_key=record_key,
                    error=str(e),
                )
                return False
            except Exception as e:
                self._log(
                    logging.ERROR,
                    "Error while registering record hotkey.",
                    event="hotkeys.register_press_error",
                    record_key=record_key,
                    error=str(e),
                    exc_info=True,
                )
                return False
            self._log(
                logging.INFO,
                "Recording hotkey registered.",
                event="hotkeys.register_press_success",
                record_key=record_key,
            )

            # Registrar a tecla de recarga
            self._log(
                logging.INFO,
                "Registering agent hotkey.",
                event="hotkeys.register_agent",
                agent_key=agent_key,
            )
            try:
                agent_handle = keyboard.on_press_key(
                    agent_key,
                    lambda _: self._on_agent_key(),
                    suppress=False,
                )
                self._store_hotkey_handle(
                    f"{agent_key}:press",
                    agent_handle,
                )
            except OSError as e:
                self._log(
                    logging.ERROR,
                    "OS error while registering agent hotkey.",
                    event="hotkeys.register_agent_os_error",
                    agent_key=agent_key,
                    error=str(e),
                )
                return False
            except Exception as e:
                self._log(
                    logging.ERROR,
                    "Unexpected error while registering agent hotkey.",
                    event="hotkeys.register_agent_error",
                    agent_key=agent_key,
                    error=str(e),
                    exc_info=True,
                )
                return False
            self._log(
                logging.INFO,
                "Agent hotkey registered.",
                event="hotkeys.register_agent_success",
                agent_key=agent_key,
            )

            self._log(
                logging.INFO,
                "Hotkeys registered successfully.",
                event="hotkeys.register_complete",
                record_key=record_key,
                agent_key=agent_key,
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
        try:
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
                # Após processar cada entrada, garantir que não haja handles residuais
                self.hotkey_handlers[handle_id] = []

            # Limpar qualquer chave vazia restante
            self.hotkey_handlers.clear()

            self._log(
                logging.INFO,
                "Hotkeys unregistered successfully.",
                event="hotkeys.unregister_success",
            )
            # Garantir que o estado reflita a ausência de hotkeys registradas
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
