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
        self.callback_reload = None
        self.record_key = "f3"  # Tecla padrão
        self.reload_key = "f4"  # Tecla padrão
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
                    self.reload_key = config.get('reload_key', self.reload_key)
                    self.record_mode = config.get('record_mode', self.record_mode)
                    logging.info(f"Configuração carregada: record_key={self.record_key}, reload_key={self.reload_key}, record_mode={self.record_mode}")
        except Exception as e:
            logging.error(f"Erro ao carregar configuração: {e}")

    def _save_config(self):
        """Salva a configuração no arquivo."""
        try:
            config = {
                'record_key': self.record_key,
                'reload_key': self.reload_key,
                'record_mode': self.record_mode
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logging.info(f"Configuração salva: {config}")
        except Exception as e:
            logging.error(f"Erro ao salvar configuração: {e}")

    def start(self):
        """Inicia o gerenciador de hotkeys."""
        if self.is_running:
            logging.warning("KeyboardHotkeyManager já está em execução.")
            return True

        try:
            # Registrar as hotkeys
            self._register_hotkeys()
            self.is_running = True
            logging.info("KeyboardHotkeyManager iniciado com sucesso.")
            return True
        except Exception as e:
            logging.error(f"Erro ao iniciar KeyboardHotkeyManager: {e}", exc_info=True)
            self.stop()
            return False

    def stop(self):
        """Para o gerenciador de hotkeys."""
        if not self.is_running:
            return

        # Desregistrar as hotkeys
        self._unregister_hotkeys()
        self.is_running = False
        logging.info("KeyboardHotkeyManager encerrado.")

    def update_config(self, record_key=None, reload_key=None, record_mode=None):
        """
        Atualiza a configuração do gerenciador de hotkeys.

        Args:
            record_key (str): Tecla de gravação
            reload_key (str): Tecla de recarga
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

            if reload_key is not None:
                self.reload_key = reload_key.lower()

            if record_mode is not None:
                self.record_mode = record_mode

            # Salvar configuração
            self._save_config()

            # Registrar novas hotkeys se estava em execução
            if was_running:
                self._register_hotkeys()

            logging.info(f"Configuração atualizada: record_key={self.record_key}, reload_key={self.reload_key}, record_mode={self.record_mode}")
            return True

        except Exception as e:
            logging.error(f"Erro ao atualizar configuração: {e}")
            return False

    def set_callbacks(self, toggle=None, start=None, stop=None, reload=None):
        """
        Define os callbacks para os eventos de hotkey.

        Args:
            toggle (callable): Callback para o evento de toggle
            start (callable): Callback para o evento de início de gravação
            stop (callable): Callback para o evento de fim de gravação
            reload (callable): Callback para o evento de recarga
        """
        if toggle is not None:
            self.callback_toggle = toggle

        if start is not None:
            self.callback_start = start

        if stop is not None:
            self.callback_stop = stop

        if reload is not None:
            self.callback_reload = reload

    def _register_hotkeys(self):
        """Registra as hotkeys no sistema."""
        try:
            logging.info("Iniciando registro de hotkeys...")

            # Desregistrar hotkeys existentes para evitar duplicação
            self._unregister_hotkeys()

            # Limpar todos os hooks do keyboard para garantir um estado limpo
            keyboard.unhook_all()
            time.sleep(0.2)  # Pequeno delay para garantir que os hooks foram removidos

            # Registrar a tecla de gravação
            logging.info(f"Registrando hotkey de gravação: {self.record_key}")

            # Definir o handler para a tecla de gravação
            if self.record_mode == "toggle":
                handler = self._on_toggle_key
            else:
                # Para o modo press, registramos dois handlers: um para pressionar e outro para soltar
                handler = self._on_press_key
                # Registrar handler para soltar a tecla no modo press
                if self.record_mode == "press":
                    keyboard.on_release_key(self.record_key, lambda _: self._on_release_key(), suppress=True)
                    logging.info(f"Handler de release registrado para: {self.record_key}")

            # Usar on_press_key em vez de add_hotkey para maior confiabilidade
            keyboard.on_press_key(self.record_key, lambda _: handler(), suppress=True)
            self.hotkey_handlers[self.record_key] = handler
            logging.info(f"Hotkey de gravação registrada com sucesso (on_press_key): {self.record_key}")

            # Registrar a tecla de recarga
            logging.info(f"Registrando hotkey de recarga: {self.reload_key}")
            keyboard.on_press_key(self.reload_key, lambda _: self._on_reload_key(), suppress=True)
            self.hotkey_handlers[self.reload_key] = self._on_reload_key
            logging.info(f"Hotkey de recarga registrada com sucesso (on_press_key): {self.reload_key}")

            logging.info(f"Hotkeys registradas com sucesso: gravação={self.record_key}, recarga={self.reload_key}")
            return True

        except Exception as e:
            logging.error(f"Erro ao registrar hotkeys: {e}", exc_info=True)
            return False

    def _unregister_hotkeys(self):
        """Desregistra as hotkeys do sistema."""
        try:
            # Não tentamos remover hotkeys individuais, pois isso pode causar erros
            # quando usamos on_press_key. Em vez disso, vamos direto para unhook_all()
            # que é mais confiável.

            # Limpar o dicionário de handlers primeiro
            self.hotkey_handlers.clear()

            # Remover todos os hooks de uma vez
            try:
                keyboard.unhook_all()
                logging.info("Todos os hooks do keyboard foram removidos.")
            except Exception as e:
                logging.error(f"Erro ao remover todos os hooks: {e}")
                # Mesmo com erro, continuamos para garantir que o estado seja consistente

            logging.info("Hotkeys desregistradas com sucesso.")

        except Exception as e:
            logging.error(f"Erro ao desregistrar hotkeys: {e}", exc_info=True)

    def _on_toggle_key(self):
        """Handler para a tecla de toggle."""
        try:
            if self.callback_toggle:
                threading.Thread(target=self.callback_toggle, daemon=True, name="ToggleCallback").start()
                logging.info("Callback de toggle chamado.")
        except Exception as e:
            logging.error(f"Erro ao chamar callback de toggle: {e}", exc_info=True)

    def _on_press_key(self):
        """Handler para a tecla de press."""
        try:
            if self.callback_start:
                threading.Thread(target=self.callback_start, daemon=True, name="StartCallback").start()
                logging.info("Callback de start chamado.")
        except Exception as e:
            logging.error(f"Erro ao chamar callback de start: {e}", exc_info=True)

    def _on_reload_key(self):
        """Handler para a tecla de recarga."""
        try:
            if self.callback_reload:
                threading.Thread(target=self.callback_reload, daemon=True, name="ReloadCallback").start()
                logging.info("Callback de reload chamado.")
        except Exception as e:
            logging.error(f"Erro ao chamar callback de reload: {e}", exc_info=True)

    def _on_release_key(self):
        """Handler para o evento de soltar a tecla no modo press."""
        try:
            if self.callback_stop:
                threading.Thread(target=self.callback_stop, daemon=True, name="StopCallback").start()
                logging.info("Callback de stop chamado (release key).")
        except Exception as e:
            logging.error(f"Erro ao chamar callback de stop (release): {e}", exc_info=True)

    def restart(self):
        """Reinicia o gerenciador de hotkeys."""
        logging.info("Reiniciando KeyboardHotkeyManager...")
        self.stop()
        time.sleep(0.5)  # Pequeno delay para garantir que tudo foi encerrado

        # Garantir que todos os hooks foram removidos
        try:
            keyboard.unhook_all()
            logging.info("Todos os hooks do keyboard foram removidos durante o restart.")
        except Exception as e:
            logging.error(f"Erro ao remover todos os hooks durante restart: {e}")

        time.sleep(0.5)  # Delay adicional para garantir limpeza completa

        # Iniciar novamente
        result = self.start()
        if result:
            logging.info("KeyboardHotkeyManager reiniciado com sucesso.")
        else:
            logging.error("Falha ao reiniciar KeyboardHotkeyManager.")

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
            logging.info(f"Iniciando detecção de tecla com timeout de {timeout} segundos...")

            # Variáveis para armazenar a tecla detectada
            detected_key = [None]
            key_detected = threading.Event()

            # Função para capturar a tecla
            def on_key_event(e):
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
            logging.error(f"Erro ao detectar tecla: {e}", exc_info=True)
            return None
