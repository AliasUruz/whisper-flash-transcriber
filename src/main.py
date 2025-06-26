import sys
import tkinter as tk
import logging
import atexit
import os # Adicionado para manipulação de caminhos

# Configuração de logging (mantida aqui para o ponto de entrada)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s', encoding='utf-8')

# Adicionar o diretório pai (WhisperTeste) ao sys.path para importações absolutas
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar os módulos da aplicação
from src.core import AppCore
from src.ui_manager import UIManager

# --- Ajuste para evitar erros "main thread is not in main loop" ao destruir
# variáveis Tkinter quando a aplicação encerra. Mantemos o destrutor original
# em `_original_variable_del` para preservar comportamentos internos e facilitar
# a manutenção em futuras versões do Tkinter.
_original_variable_del = tk.Variable.__del__

def _safe_variable_del(self):
    tk_root = getattr(self, "_tk", None)
    if not tk_root: return
    try:
        if tk_root.getboolean(tk_root.call("info", "exists", self._name)):
            tk_root.globalunsetvar(self._name)
    except Exception: pass
    cmds = getattr(self, "_tclCommands", None)
    if cmds:
        for name in cmds:
            try:
                tk_root.deletecommand(name)
            except Exception:
                pass
            self._tclCommands = None
    try:
        # Garante que eventuais comportamentos futuros do destrutor original
        # sejam preservados.
        _original_variable_del(self)
    except Exception:
        pass

tk.Variable.__del__ = _safe_variable_del

# Variáveis globais para referências (necessárias para pystray e callbacks)
app_core_instance = None
ui_manager_instance = None

def on_exit_app_enhanced(*_):
    logging.info("Saída solicitada pelo ícone da bandeja.")
    if app_core_instance:
        app_core_instance.shutdown()
    if ui_manager_instance and ui_manager_instance.tray_icon:
        ui_manager_instance.tray_icon.stop()
    # Agenda o encerramento do mainloop na thread principal sem gerar exceção
    main_tk_root.after(0, main_tk_root.quit)

if __name__ == "__main__":
    atexit.register(on_exit_app_enhanced)

    main_tk_root = tk.Tk()
    main_tk_root.withdraw()

    # Inicializar a lógica principal (AppCore)
    app_core_instance = AppCore(main_tk_root)

    # Inicializar o gerenciador de UI
    ui_manager_instance = UIManager(main_tk_root, app_core_instance.config_manager, app_core_instance)
    app_core_instance.ui_manager = ui_manager_instance # Conectar UI Manager ao AppCore

    # Definir callbacks do AppCore para o UI Manager
    app_core_instance.set_state_update_callback(ui_manager_instance.update_tray_icon)

    # Configurar e iniciar o ícone da bandeja
    ui_manager_instance.setup_tray_icon()

    # Sobrescrever a função on_exit_app original (para o menu do pystray)
    ui_manager_instance.on_exit_app = on_exit_app_enhanced

    logging.info("Iniciando o mainloop do Tkinter na thread principal.")
    main_tk_root.mainloop()
    logging.info("Mainloop do Tkinter finalizado. A aplicação será encerrada.")
    sys.exit(0)
