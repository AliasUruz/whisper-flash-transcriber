# -*- coding: utf-8 -*-
import os
import json
import time
import subprocess
import threading
import logging
from pathlib import Path

class AutoHotkeyManager:
    """
    Gerencia a comunicação com o AutoHotkey para captura de hotkeys.
    Esta classe substitui as bibliotecas keyboard e pynput, oferecendo uma
    solução mais robusta para o Windows 11.
    """
    
    def __init__(self, config_file="ahk_config.json", script_file="whisper_hotkeys.ahk", signal_file="hotkey_signals.txt"):
        """
        Inicializa o gerenciador AutoHotkey.
        
        Args:
            config_file (str): Caminho para o arquivo de configuração do AutoHotkey
            script_file (str): Caminho para o script AutoHotkey
            signal_file (str): Caminho para o arquivo de comunicação
        """
        self.config_file = config_file
        self.script_file = script_file
        self.signal_file = signal_file
        self.ahk_process = None
        self.is_running = False
        self.last_signal_time = 0
        self.callback_toggle = None
        self.callback_start = None
        self.callback_stop = None
        self.callback_reload = None
        self.signal_check_thread = None
        self.stop_signal_check = threading.Event()
        
        # Verificar se o AutoHotkey está instalado
        self._check_ahk_installed()
    
    def _check_ahk_installed(self):
        """Verifica se o AutoHotkey está instalado no sistema."""
        try:
            # Tenta encontrar o executável do AutoHotkey
            ahk_paths = [
                r"C:\Program Files\AutoHotkey\AutoHotkey.exe",
                r"C:\Program Files (x86)\AutoHotkey\AutoHotkey.exe",
                # Adicione outros caminhos comuns se necessário
            ]
            
            self.ahk_exe = None
            for path in ahk_paths:
                if os.path.exists(path):
                    self.ahk_exe = path
                    break
            
            if not self.ahk_exe:
                # Tenta encontrar pelo PATH
                try:
                    result = subprocess.run(["where", "AutoHotkey.exe"], 
                                           capture_output=True, 
                                           text=True, 
                                           check=True)
                    if result.stdout.strip():
                        self.ahk_exe = result.stdout.strip().split('\n')[0]
                except subprocess.CalledProcessError:
                    pass
            
            if not self.ahk_exe:
                logging.error("AutoHotkey não encontrado. Por favor, instale o AutoHotkey.")
                raise RuntimeError("AutoHotkey não encontrado. Por favor, instale o AutoHotkey.")
            
            logging.info(f"AutoHotkey encontrado em: {self.ahk_exe}")
            
        except Exception as e:
            logging.error(f"Erro ao verificar instalação do AutoHotkey: {e}")
            raise
    
    def start(self):
        """Inicia o script AutoHotkey e o monitoramento de sinais."""
        if self.is_running:
            logging.warning("AutoHotkey já está em execução.")
            return
        
        try:
            # Garantir que o arquivo de sinais não existe ou está vazio
            if os.path.exists(self.signal_file):
                open(self.signal_file, 'w').close()
            
            # Iniciar o processo do AutoHotkey
            logging.info(f"Iniciando AutoHotkey com o script: {self.script_file}")
            self.ahk_process = subprocess.Popen(
                [self.ahk_exe, self.script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Verificar se o processo iniciou corretamente
            if self.ahk_process.poll() is not None:
                stderr = self.ahk_process.stderr.read().decode('utf-8', errors='ignore')
                raise RuntimeError(f"Falha ao iniciar AutoHotkey: {stderr}")
            
            self.is_running = True
            
            # Iniciar thread de monitoramento de sinais
            self.stop_signal_check.clear()
            self.signal_check_thread = threading.Thread(
                target=self._monitor_signals,
                daemon=True,
                name="AHKSignalMonitor"
            )
            self.signal_check_thread.start()
            
            logging.info("AutoHotkey iniciado com sucesso.")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao iniciar AutoHotkey: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Para o script AutoHotkey e o monitoramento de sinais."""
        if not self.is_running:
            return
        
        # Parar o monitoramento de sinais
        if self.signal_check_thread and self.signal_check_thread.is_alive():
            self.stop_signal_check.set()
            self.signal_check_thread.join(timeout=1.0)
        
        # Encerrar o processo do AutoHotkey
        if self.ahk_process:
            try:
                self.ahk_process.terminate()
                self.ahk_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.ahk_process.kill()
            except Exception as e:
                logging.error(f"Erro ao encerrar processo do AutoHotkey: {e}")
            
            self.ahk_process = None
        
        self.is_running = False
        logging.info("AutoHotkey encerrado.")
    
    def update_config(self, record_key=None, reload_key=None, record_mode=None):
        """
        Atualiza a configuração do AutoHotkey.
        
        Args:
            record_key (str): Tecla de gravação
            reload_key (str): Tecla de recarga
            record_mode (str): Modo de gravação ('toggle' ou 'press')
        """
        try:
            # Carregar configuração atual
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            
            # Atualizar valores
            if record_key is not None:
                config['record_key'] = record_key
            
            if reload_key is not None:
                config['reload_key'] = reload_key
            
            if record_mode is not None:
                config['record_mode'] = record_mode
            
            # Salvar configuração
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"Configuração do AutoHotkey atualizada: {config}")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao atualizar configuração do AutoHotkey: {e}")
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
    
    def _monitor_signals(self):
        """Monitora o arquivo de sinais em busca de eventos do AutoHotkey."""
        logging.info("Iniciando monitoramento de sinais do AutoHotkey.")
        
        while not self.stop_signal_check.is_set():
            try:
                if os.path.exists(self.signal_file):
                    # Verificar se o arquivo foi modificado
                    mod_time = os.path.getmtime(self.signal_file)
                    
                    if mod_time > self.last_signal_time:
                        self.last_signal_time = mod_time
                        
                        # Ler o sinal
                        with open(self.signal_file, 'r') as f:
                            signal_data = f.read().strip()
                        
                        if signal_data:
                            self._process_signal(signal_data)
            
            except Exception as e:
                logging.error(f"Erro ao monitorar sinais do AutoHotkey: {e}")
            
            # Aguardar um pouco antes de verificar novamente
            time.sleep(0.1)
        
        logging.info("Monitoramento de sinais do AutoHotkey encerrado.")
    
    def _process_signal(self, signal_data):
        """
        Processa um sinal recebido do AutoHotkey.
        
        Args:
            signal_data (str): Dados do sinal no formato "COMANDO|TIMESTAMP"
        """
        try:
            parts = signal_data.split('|')
            if len(parts) < 1:
                return
            
            command = parts[0]
            logging.debug(f"Sinal recebido do AutoHotkey: {command}")
            
            # Processar o comando
            if command == "TOGGLE" and self.callback_toggle:
                threading.Thread(target=self.callback_toggle, daemon=True, name="AHKToggleCallback").start()
            
            elif command == "START" and self.callback_start:
                threading.Thread(target=self.callback_start, daemon=True, name="AHKStartCallback").start()
            
            elif command == "STOP" and self.callback_stop:
                threading.Thread(target=self.callback_stop, daemon=True, name="AHKStopCallback").start()
            
            elif command == "RELOAD" and self.callback_reload:
                threading.Thread(target=self.callback_reload, daemon=True, name="AHKReloadCallback").start()
            
            elif command.startswith("LOG"):
                logging.info(f"AutoHotkey: {command[4:]}")
            
            elif command.startswith("ERROR"):
                logging.error(f"AutoHotkey Error: {command[6:]}")
        
        except Exception as e:
            logging.error(f"Erro ao processar sinal do AutoHotkey: {e}")
    
    def is_ahk_running(self):
        """Verifica se o processo do AutoHotkey ainda está em execução."""
        if not self.ahk_process:
            return False
        
        return self.ahk_process.poll() is None
    
    def restart(self):
        """Reinicia o script AutoHotkey."""
        self.stop()
        time.sleep(0.5)  # Pequeno delay para garantir que o processo foi encerrado
        return self.start()
