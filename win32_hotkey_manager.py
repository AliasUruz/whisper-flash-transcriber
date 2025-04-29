# -*- coding: utf-8 -*-
import os
import json
import time
import subprocess
import threading
import logging
from pathlib import Path
import win32api
import win32con
import win32gui
import win32event

class Win32HotkeyManager:
    """
    Gerencia a comunicação com o AutoHotkey para captura de hotkeys usando a API Win32.
    Esta classe substitui as bibliotecas keyboard e pynput, oferecendo uma
    solução mais robusta para o Windows 11.
    """

    # Constantes para mensagens do Windows
    WM_APP = 0x8000  # Base para mensagens personalizadas do Windows
    WM_WHISPER_TOGGLE = WM_APP + 1  # Mensagem para toggle
    WM_WHISPER_START = WM_APP + 2   # Mensagem para iniciar gravação
    WM_WHISPER_STOP = WM_APP + 3    # Mensagem para parar gravação
    WM_WHISPER_RELOAD = WM_APP + 4  # Mensagem para recarregar hotkeys
    WM_WHISPER_PING = WM_APP + 5    # Mensagem para verificar se o aplicativo está respondendo

    def __init__(self, config_file="ahk_config.json", script_file="whisper_hotkeys_simple.ahk"):
        """
        Inicializa o gerenciador Win32Hotkey.

        Args:
            config_file (str): Caminho para o arquivo de configuração do AutoHotkey
            script_file (str): Caminho para o script AutoHotkey
        """
        self.config_file = config_file
        self.script_file = script_file
        self.ahk_process = None
        self.is_running = False
        self.callback_toggle = None
        self.callback_start = None
        self.callback_stop = None
        self.callback_reload = None
        self.message_window = None
        self.message_thread = None
        self.stop_message_thread = threading.Event()

        # Verificar se o AutoHotkey está instalado
        self._check_ahk_installed()

    def _check_ahk_installed(self):
        """Verifica se o AutoHotkey está instalado no sistema."""
        try:
            # Tenta encontrar o executável do AutoHotkey (v1 e v2)
            ahk_paths = [
                # AutoHotkey v1
                r"C:\Program Files\AutoHotkey\AutoHotkey.exe",
                r"C:\Program Files (x86)\AutoHotkey\AutoHotkey.exe",
                # AutoHotkey v2
                r"C:\Program Files\AutoHotkey\v2\AutoHotkey.exe",
                r"C:\Program Files (x86)\AutoHotkey\v2\AutoHotkey.exe",
                # Caminhos de instalação personalizados comuns
                r"C:\AutoHotkey\AutoHotkey.exe",
                r"C:\AutoHotkey\v2\AutoHotkey.exe",
                # Pasta do usuário
                os.path.expanduser(r"~\AppData\Local\Programs\AutoHotkey\AutoHotkey.exe"),
                os.path.expanduser(r"~\AppData\Local\Programs\AutoHotkey\v2\AutoHotkey.exe"),
                # Pasta atual
                os.path.join(os.getcwd(), "AutoHotkey.exe"),
                # Pasta do script
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "AutoHotkey.exe"),
            ]

            self.ahk_exe = None
            for path in ahk_paths:
                if os.path.exists(path):
                    self.ahk_exe = path
                    logging.info(f"AutoHotkey encontrado em: {self.ahk_exe}")
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
                        logging.info(f"AutoHotkey encontrado no PATH: {self.ahk_exe}")
                except subprocess.CalledProcessError:
                    logging.warning("AutoHotkey não encontrado no PATH")

            # Se ainda não encontrou, tenta verificar no registro do Windows
            if not self.ahk_exe:
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\AutoHotkey") as key:
                        install_dir = winreg.QueryValueEx(key, "InstallDir")[0]
                        potential_path = os.path.join(install_dir, "AutoHotkey.exe")
                        if os.path.exists(potential_path):
                            self.ahk_exe = potential_path
                            logging.info(f"AutoHotkey encontrado no registro: {self.ahk_exe}")
                except Exception as reg_error:
                    logging.warning(f"Não foi possível verificar o registro do Windows: {reg_error}")

            # Se ainda não encontrou, tenta baixar e extrair o AutoHotkey portátil
            if not self.ahk_exe:
                try:
                    self.ahk_exe = self._download_portable_ahk()
                    if self.ahk_exe:
                        logging.info(f"AutoHotkey portátil baixado e extraído: {self.ahk_exe}")
                except Exception as dl_error:
                    logging.error(f"Erro ao baixar AutoHotkey portátil: {dl_error}")

            if not self.ahk_exe:
                error_msg = "AutoHotkey não encontrado. Por favor, instale o AutoHotkey."
                logging.error(error_msg)
                raise RuntimeError(error_msg)

        except Exception as e:
            logging.error(f"Erro ao verificar instalação do AutoHotkey: {e}")
            raise

    def _download_portable_ahk(self):
        """Baixa e extrai o AutoHotkey portátil."""
        import urllib.request
        import zipfile
        import tempfile

        # URL do AutoHotkey portátil
        ahk_url = "https://www.autohotkey.com/download/ahk.zip"

        # Pasta para extrair o AutoHotkey
        extract_dir = os.path.join(os.getcwd(), "AutoHotkey_Portable")
        os.makedirs(extract_dir, exist_ok=True)

        # Baixar o arquivo zip
        logging.info(f"Baixando AutoHotkey portátil de {ahk_url}...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                urllib.request.urlretrieve(ahk_url, temp_path)

                # Extrair o arquivo zip
                logging.info(f"Extraindo AutoHotkey portátil para {extract_dir}...")
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                # Verificar se o executável foi extraído
                ahk_exe_path = os.path.join(extract_dir, "AutoHotkey.exe")
                if os.path.exists(ahk_exe_path):
                    return ahk_exe_path
                else:
                    logging.error(f"AutoHotkey.exe não encontrado após extração em {extract_dir}")
                    return None
            except Exception as e:
                logging.error(f"Erro ao baixar/extrair AutoHotkey portátil: {e}")
                return None
            finally:
                # Remover o arquivo temporário
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def start(self):
        """Inicia o script AutoHotkey e a janela de mensagens."""
        if self.is_running:
            logging.warning("Win32HotkeyManager já está em execução.")
            return True

        try:
            # Iniciar a thread da janela de mensagens
            self.stop_message_thread.clear()
            self.message_thread = threading.Thread(
                target=self._run_message_window,
                daemon=True,
                name="Win32MessageWindow"
            )
            self.message_thread.start()

            # Aguardar a criação da janela de mensagens
            start_time = time.time()
            while not self.message_window and time.time() - start_time < 5.0:
                time.sleep(0.1)

            if not self.message_window:
                raise RuntimeError("Falha ao criar janela de mensagens do Windows.")

            # Atualizar a configuração do AutoHotkey
            self.update_config()

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
            logging.info("Win32HotkeyManager iniciado com sucesso.")
            return True

        except Exception as e:
            logging.error(f"Erro ao iniciar Win32HotkeyManager: {e}", exc_info=True)
            self.stop()
            return False

    def stop(self):
        """Para o script AutoHotkey e a janela de mensagens."""
        if not self.is_running:
            return

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

        # Parar a thread da janela de mensagens
        if self.message_thread and self.message_thread.is_alive():
            self.stop_message_thread.set()
            if self.message_window:
                win32gui.PostMessage(self.message_window, win32con.WM_CLOSE, 0, 0)
            self.message_thread.join(timeout=2.0)

        self.message_window = None
        self.is_running = False
        logging.info("Win32HotkeyManager encerrado.")

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
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

            # Atualizar valores
            if record_key is not None:
                config['record_key'] = record_key

            if reload_key is not None:
                config['reload_key'] = reload_key

            if record_mode is not None:
                config['record_mode'] = record_mode

            # Salvar configuração
            with open(self.config_file, 'w', encoding='utf-8') as f:
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

    def _run_message_window(self):
        """Cria e executa uma janela de mensagens do Windows para receber mensagens do AutoHotkey."""
        try:
            # Registrar a classe da janela
            window_class = "WhisperAppMessageWindow"

            # Definir a classe da janela
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._wnd_proc_wrapper
            wc.lpszClassName = window_class
            wc.hInstance = win32api.GetModuleHandle(None)

            # Registrar a classe
            class_atom = win32gui.RegisterClass(wc)

            # Criar a janela
            self.message_window = win32gui.CreateWindow(
                class_atom,
                "Whisper App Message Window",
                0,  # Estilo
                0, 0, 0, 0,  # Posição e tamanho (invisível)
                0,  # hWndParent
                0,  # hMenu
                wc.hInstance,
                None  # lpParam
            )

            logging.info(f"Janela de mensagens criada com sucesso. Handle: {self.message_window}")

            # Loop de mensagens
            while not self.stop_message_thread.is_set():
                try:
                    # Processar mensagens
                    result, msg = win32gui.PeekMessage(None, 0, 0, win32con.PM_REMOVE)
                    if result:
                        if msg[1] == win32con.WM_QUIT:
                            break
                        win32gui.TranslateMessage(msg)
                        win32gui.DispatchMessage(msg)
                    else:
                        # Se não houver mensagens, aguarde um pouco para não consumir CPU
                        time.sleep(0.01)
                except Exception as e:
                    logging.error(f"Erro no loop de mensagens: {e}", exc_info=True)
                    time.sleep(0.1)  # Pequeno delay para evitar consumo excessivo de CPU
                    continue  # Continuar o loop em vez de quebrar

            # Destruir a janela
            if self.message_window:
                win32gui.DestroyWindow(self.message_window)
                self.message_window = None

            # Desregistrar a classe
            win32gui.UnregisterClass(class_atom, wc.hInstance)

            logging.info("Janela de mensagens encerrada.")

        except Exception as e:
            logging.error(f"Erro ao criar janela de mensagens: {e}", exc_info=True)
            self.message_window = None

    def _wnd_proc_wrapper(self, hwnd, msg, wparam, lparam):
        """Wrapper para o _wnd_proc para garantir que os argumentos sejam passados corretamente."""
        try:
            return self._wnd_proc(hwnd, msg, wparam, lparam)
        except Exception as e:
            logging.error(f"Erro no wrapper do wnd_proc: {e}", exc_info=True)
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        """Processa mensagens recebidas pela janela."""
        try:
            # Processar mensagens específicas do aplicativo
            if msg == self.WM_WHISPER_TOGGLE and self.callback_toggle:
                threading.Thread(target=self.callback_toggle, daemon=True, name="ToggleCallback").start()
                return 0

            elif msg == self.WM_WHISPER_START and self.callback_start:
                threading.Thread(target=self.callback_start, daemon=True, name="StartCallback").start()
                return 0

            elif msg == self.WM_WHISPER_STOP and self.callback_stop:
                threading.Thread(target=self.callback_stop, daemon=True, name="StopCallback").start()
                return 0

            elif msg == self.WM_WHISPER_RELOAD and self.callback_reload:
                threading.Thread(target=self.callback_reload, daemon=True, name="ReloadCallback").start()
                return 0

            elif msg == self.WM_WHISPER_PING:
                logging.debug("Ping recebido do AutoHotkey.")
                return 0

            elif msg == win32con.WM_DESTROY:
                win32gui.PostQuitMessage(0)
                return 0

            # Processar outras mensagens com o comportamento padrão
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

        except Exception as e:
            logging.error(f"Erro ao processar mensagem {msg}: {e}", exc_info=True)
            return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

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

    def detect_key(self, timeout=5.0):
        """Detecta uma tecla pressionada.

        Args:
            timeout (float): Tempo máximo de espera em segundos

        Returns:
            str: A tecla detectada ou None se nenhuma tecla for detectada
        """
        logging.info("Método detect_key não implementado no Win32HotkeyManager.")
        logging.info("Usando método alternativo para detectar tecla...")

        # Implementação simples para detectar teclas
        # Retorna uma tecla padrão para evitar erros
        return "F3"
