# -*- coding: utf-8 -*-
import os
import json
import time
import threading
import logging
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
        """Carrega a configuração do arquivo."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.record_key = config.get('record_key', self.record_key)
                    self.agent_key = config.get('agent_key', self.agent_key)
                    self.record_mode = config.get('record_mode', self.record_mode)
                    logging.info(f"Configuration loaded: record_key={self.record_key}, agent_key={self.agent_key}, record_mode={self.record_mode}")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")

    def _save_config(self):
        """Salva a configuração no arquivo."""
        try:
            config = {
                'record_key': self.record_key,
                'agent_key': self.agent_key,
                'record_mode': self.record_mode
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logging.info(f"Configuration saved: {config}")
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")

    def start(self):
        """Starts the hotkey manager."""
        if self.is_running:
            logging.warning("KeyboardHotkeyManager is already running.")
            return True

        try:
            # Registrar as hotkeys
            self._register_hotkeys()
            self.is_running = True
            logging.info("KeyboardHotkeyManager started successfully.")
            return True
        except Exception as e:
            logging.error(f"Error starting KeyboardHotkeyManager: {e}", exc_info=True)
            self.stop()
            return False

    def stop(self):
        """Stops the hotkey manager."""
        if not self.is_running:
            return

        # Desregistrar as hotkeys
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
                self._register_hotkeys()

            logging.info(f"Configuration updated: record_key={self.record_key}, agent_key={self.agent_key}, record_mode={self.record_mode}")
            return True

        except Exception as e:
            logging.error(f"Error updating configuration: {e}")
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

    def _register_hotkeys(self):
        """Registers hotkeys in the system."""
        try:
            logging.info("Registering hotkeys...")

            # Desregistrar hotkeys existentes para evitar duplicação
            self._unregister_hotkeys()

            # Limpar todos os hooks do keyboard para garantir um estado limpo
            keyboard.unhook_all()
            time.sleep(0.2)  # Pequeno delay para garantir que os hooks foram removidos

            # Registrar a tecla de gravação
            logging.info(f"Registering record hotkey: {self.record_key}")

            # Definir o handler para a tecla de gravação
            if self.record_mode == "toggle":
                handler = self._on_toggle_key
            else:
                # Para o modo press, registramos dois handlers: um para pressionar e outro para soltar
                handler = self._on_press_key
                # Registrar handler para soltar a tecla no modo press
                if self.record_mode == "press":
                    keyboard.on_release_key(self.record_key, lambda _: self._on_release_key(), suppress=True)
                    logging.info(f"Release handler registered for: {self.record_key}")

            # Usar on_press_key em vez de add_hotkey para maior confiabilidade
            keyboard.on_press_key(self.record_key, lambda _: handler(), suppress=True)
            self.hotkey_handlers[self.record_key] = handler
            logging.info(f"Record hotkey registered successfully (on_press_key): {self.record_key}")

            # Registrar a tecla de recarga
            logging.info(f"Registering command hotkey: {self.agent_key}")
            keyboard.on_press_key(self.agent_key, lambda _: self._on_agent_key(), suppress=True)
            self.hotkey_handlers[self.agent_key] = self._on_agent_key
            logging.info(f"Command hotkey registered successfully (on_press_key): {self.agent_key}")

            logging.info(f"Hotkeys registered successfully: record={self.record_key}, command={self.agent_key}")
            return True

        except Exception as e:
            logging.error(f"Error registering hotkeys: {e}", exc_info=True)
            return False

    def _unregister_hotkeys(self):
        """Unregisters hotkeys from the system."""
        try:
            # Não tentamos remover hotkeys individuais, pois isso pode causar erros
            # quando usamos on_press_key. Em vez disso, vamos direto para unhook_all()
            # que é mais confiável.

            # Limpar o dicionário de handlers primeiro
            self.hotkey_handlers.clear()

            # Remover todos os hooks de uma vez
            try:
                keyboard.unhook_all()
                logging.info("All keyboard hooks have been removed.")
            except Exception as e:
                logging.error(f"Error removing all hooks: {e}")
                # Mesmo com erro, continuamos para garantir que o estado seja consistente

            logging.info("Hotkeys unregistered successfully.")

        except Exception as e:
            logging.error(f"Error unregistering hotkeys: {e}", exc_info=True)

    def _on_toggle_key(self):
        """Handler for the toggle hotkey."""
        try:
            if self.callback_toggle:
                threading.Thread(target=self.callback_toggle, daemon=True, name="ToggleCallback").start()
                logging.info("Toggle callback invoked.")
        except Exception as e:
            logging.error(f"Error calling toggle callback: {e}", exc_info=True)

    def _on_press_key(self):
        """Handler for the press hotkey."""
        try:
            if self.callback_start:
                threading.Thread(target=self.callback_start, daemon=True, name="StartCallback").start()
                logging.info("Start callback invoked.")
        except Exception as e:
            logging.error(f"Error calling start callback: {e}", exc_info=True)

    def _on_agent_key(self):
        """Handler for the agent command hotkey."""
        try:
            if self.callback_agent:
                threading.Thread(target=self.callback_agent, daemon=True, name="AgentCallback").start()
                logging.info("Command callback invoked.")
        except Exception as e:
            logging.error(f"Error calling command callback: {e}", exc_info=True)

    def _on_release_key(self):
        """Handler for key release event in press mode."""
        try:
            if self.callback_stop:
                threading.Thread(target=self.callback_stop, daemon=True, name="StopCallback").start()
                logging.info("Stop callback invoked (release key).")
        except Exception as e:
            logging.error(f"Error calling stop callback (release): {e}", exc_info=True)

    def restart(self):
        """Restarts the hotkey manager."""
        logging.info("Restarting KeyboardHotkeyManager...")
        self.stop()
        time.sleep(0.5)  # Pequeno delay para garantir que tudo foi encerrado

        # Garantir que todos os hooks foram removidos
        try:
            keyboard.unhook_all()
            logging.info("All keyboard hooks removed during restart.")
        except Exception as e:
            logging.error(f"Error removing all hooks during restart: {e}")

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
        Detects a key press.

        Args:
            timeout (float): Maximum wait time in seconds

        Returns:
            str: Detected key or None if no key detected
        """
        try:
            logging.info(f"Starting key detection with timeout {timeout} seconds...")

            # Variáveis para armazenar a tecla detectada
            detected_key = [None]
            key_detected = threading.Event()

            # Função para capturar a tecla
            def on_key_event(e):
                # Ignorar teclas especiais como shift, ctrl, alt
                if e.name in ['shift', 'ctrl', 'alt', 'left shift', 'right shift', 'left ctrl', 'right ctrl', 'left alt', 'right alt']:
                    return

                logging.info(f"Key detected: {e.name}")
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
                    logging.info(f"Detected key returned: {detected_key[0]}")
                    return detected_key[0]
                else:
                    logging.info("No key detected within the timeout.")
                    return None
            finally:
                # Remover o hook
                keyboard.unhook(hook)

        except Exception as e:
            logging.error(f"Error detecting key: {e}", exc_info=True)
            return None
