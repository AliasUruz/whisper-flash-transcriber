# -*- coding: utf-8 -*-
import os
import json
import time
import threading
import logging
from pathlib import Path

import keyboard

class KeyboardHotkeyManager:
    """
    Gerencia hotkeys usando a biblioteca keyboard.
    Esta classe oferece uma solução mais simples para o gerenciamento de hotkeys.
    """

    def __init__(self, config_file="hotkey_config.json"):
        """
        Inicializa o gerenciador de hotkeys.

        Args:
            config_file (str): Caminho para o arquivo de configuração
        """
        self.config_file = config_file
        self.is_running = False
        self.callback_toggle = None
        self.callback_start = None
        self.callback_stop = None
        self.callback_agent = None
        self.record_key = "f3"  # Tecla padrão
        self.agent_key = "f4"  # Tecla padrão para comando agêntico
        self.record_mode = "toggle"  # Modo padrão
        self.hotkey_handlers = {}

        # Carregar configuração se existir
        self._load_config()

    def _load_config(self):
        """Load configuration from disk, creating the file with defaults when it is missing."""
        try:
            if not os.path.exists(self.config_file):
                logging.info(f"'{self.config_file}' not found. Creating it with default values for the first launch.")
                self._save_config()

            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.record_key = config.get('record_key', self.record_key)
                self.agent_key = config.get('agent_key', self.agent_key)
                self.record_mode = config.get('record_mode', self.record_mode)
                logging.info(
                    "Configuration loaded: record_key=%s, agent_key=%s, record_mode=%s",
                    self.record_key,
                    self.agent_key,
                    self.record_mode,
                )
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(
                f"Error loading or creating hotkey config: {e}. Resetting file to defaults.",
                exc_info=True,
            )
            try:
                self.record_key = "f3"
                self.agent_key = "f4"
                self.record_mode = "toggle"
                self._save_config()
            except Exception as write_error:
                logging.error(
                    f"Failed to rebuild hotkey configuration after corruption: {write_error}",
                    exc_info=True,
                )
                raise
            raise RuntimeError(
                f"Hotkey configuration '{self.config_file}' was corrupted and has been reset. Please restart the application."
            ) from e
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading hotkey config: {e}", exc_info=True)
            raise

    def _save_config(self):
        """Persist the current hotkey configuration to disk."""
        try:
            config = {
                'record_key': self.record_key,
                'agent_key': self.agent_key,
                'record_mode': self.record_mode
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logging.info("Configuration saved: record_key=%s, agent_key=%s, record_mode=%s",
                         self.record_key, self.agent_key, self.record_mode)
        except Exception as e:
            logging.error(f"Failed to save hotkey configuration: {e}")
            raise RuntimeError(
                f"Unable to persist hotkey configuration '{self.config_file}': {e}"
            ) from e

    def start(self):
        """Inicia o gerenciador de hotkeys."""
        if self.is_running:
            logging.warning("KeyboardHotkeyManager is already running.")
            return True

        try:
            # Registrar as hotkeys e verificar o resultado
            success = self._register_hotkeys()
            if not success:
                logging.error("Failed to register hotkeys.")
                self.stop()
                return False

            self.is_running = True
            logging.info("KeyboardHotkeyManager started successfully.")
            return True
        except Exception as e:
            logging.error(f"Error starting KeyboardHotkeyManager: {e}", exc_info=True)
            self.stop()
            return False

    def stop(self):
        """Para o gerenciador de hotkeys."""
        # Sempre tente remover as hotkeys, mesmo que o estado esteja incorreto
        self._unregister_hotkeys()
        self.is_running = False
        logging.info("KeyboardHotkeyManager stopped.")

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
                result = self._register_hotkeys()
                if not result:
                    logging.error("Failed to register hotkeys after applying the update.")
                    self.is_running = False
                    return False
                # Retomar o estado de execução se o registro foi bem-sucedido
                self.is_running = True

            logging.info(f"Configuration saved: record_key={self.record_key}, agent_key={self.agent_key}, record_mode={self.record_mode}")
            return True

        except Exception as e:
            logging.error(f"Failed to update hotkey configuration: {e}")
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
            logging.warning(
                "Hotkey handle '%s' is None; the hook may not have been registered correctly.",
                handle_id,
            )
            return
        self.hotkey_handlers.setdefault(handle_id, []).append(handle)

    def _register_hotkeys(self):
        """Registra as hotkeys no sistema."""
        try:
            logging.info("Iniciando registro de hotkeys...")

            # Desregistrar hotkeys existentes para evitar duplicação
            self._unregister_hotkeys()

            # Registrar a tecla de gravação
            logging.info("Registering recording hotkey: %s", self.record_key)

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
                        logging.error(
                            f"Specific failure while registering release hotkey: {e}"
                        )
                        return False
                    except Exception as e:
                        logging.error(
                            f"Error while registering release hotkey: {e}", exc_info=True
                        )
                        return False
                    logging.info("Release handler registered for: %s", self.record_key)

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
                logging.error(
                    f"Specific failure while registering record hotkey: {e}"
                )
                return False
            except Exception as e:
                logging.error(
                    f"Failed to register record hotkey: {e}", exc_info=True
                )
                return False
            logging.info(
                "Recording hotkey registered successfully (on_press_key): %s",
                self.record_key,
            )

            # Registrar a tecla de recarga
            logging.info("Registering command hotkey: %s", self.agent_key)
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
                logging.error(
                    f"Specific failure while registering agent hotkey: {e}"
                )
                return False
            except Exception as e:
                logging.error(
                    f"Error while registering agent hotkey: {e}", exc_info=True
                )
                return False
            logging.info(
                "Command hotkey registered successfully (on_press_key): %s",
                self.agent_key,
            )

            logging.info("Hotkeys registered successfully: record=%s, command=%s", self.record_key, self.agent_key)
            return True

        except Exception as e:
            logging.error(f"Error registering hotkeys: {e}", exc_info=True)
            return False

    def _unregister_hotkeys(self):
        """Desregistra as hotkeys do sistema."""
        try:
            for handle_id, handles in list(self.hotkey_handlers.items()):
                for handle in handles:
                    try:
                        keyboard.unhook(handle)
                        logging.debug("Hotkey handle '%s' removed.", handle_id)
                    except (KeyError, ValueError):
                        logging.warning(
                            "Hotkey handle '%s' was already removed or is invalid. Skipping.",
                            handle_id,
                        )
                    except Exception as e:
                        logging.error(
                            "Error while removing hook '%s': %s",
                            handle_id,
                            e,
                            exc_info=True,
                        )
                # Após processar cada entrada, garantir que não haja handles residuais
                self.hotkey_handlers[handle_id] = []

            # Limpar qualquer chave vazia restante
            self.hotkey_handlers.clear()

            logging.info("Hotkeys unregistered successfully.")
            # Garantir que o estado reflita a ausência de hotkeys registradas
            self.is_running = False

        except Exception as e:
            logging.error(f"Error unregistering hotkeys: {e}", exc_info=True)

    def _on_toggle_key(self):
        """Handler para a tecla de toggle."""
        try:
            if self.callback_toggle:
                threading.Thread(target=self.callback_toggle, daemon=True, name="ToggleCallback").start()
                logging.info("Toggle callback invoked.")
        except Exception as e:
            logging.error(f"Error while invoking toggle callback: {e}", exc_info=True)

    def _on_press_key(self):
        """Handler para a tecla de press."""
        try:
            if self.callback_start:
                threading.Thread(target=self.callback_start, daemon=True, name="StartCallback").start()
                logging.info("Start callback invoked.")
        except Exception as e:
            logging.error(f"Error while invoking start callback: {e}", exc_info=True)

    def _on_agent_key(self):
        """Handler for the agent command hotkey."""
        try:
            if self.callback_agent:
                threading.Thread(target=self.callback_agent, daemon=True, name="AgentCallback").start()
                logging.info("Agent callback invoked.")
        except Exception as e:
            logging.error(f"Error while invoking agent callback: {e}", exc_info=True)

    def _on_release_key(self):
        """Handler para o evento de soltar a tecla no modo press."""
        try:
            if self.callback_stop:
                threading.Thread(target=self.callback_stop, daemon=True, name="StopCallback").start()
                logging.info("Stop callback invoked (release key).")
        except Exception as e:
            logging.error(f"Error while invoking stop callback (release): {e}", exc_info=True)

    def restart(self):
        """Reinicia o gerenciador de hotkeys."""
        logging.info("Restarting KeyboardHotkeyManager...")
        self.stop()
        time.sleep(0.5)  # Pequeno delay para garantir que tudo foi encerrado

        time.sleep(0.5)  # Delay adicional para garantir limpeza completa

        # Iniciar novamente
        result = self.start()
        if result:
            logging.info("KeyboardHotkeyManager restarted successfully.")
        else:
            logging.error("Failed to restart KeyboardHotkeyManager.")

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
            logging.info(f"Starting key detection with a timeout of {timeout} seconds...")

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

                logging.info(f"Tecla detectada: {e.name}")
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
                    logging.info(f"Tecla detectada e retornada: {detected_key[0]}")
                    return detected_key[0]
                else:
                    logging.info("Nenhuma tecla detectada dentro do timeout.")
                    return None
            finally:
                # Remover o hook
                keyboard.unhook(hook)

        except Exception as e:
            logging.error(f"Error while detecting key: {e}", exc_info=True)
            return None

    def detect_single_key(self, timeout=5.0):
        """Mantida para compatibilidade: delega para ``detect_key``."""
        logging.info("detect_single_key foi chamado; delegando para detect_key.")
        return self.detect_key(timeout=timeout)
