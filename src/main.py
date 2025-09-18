import sys
import tkinter as tk
import logging
import atexit
import os  # Adicionado para manipulação de caminhos
import importlib

# Configuração de logging (mantida aqui para o ponto de entrada)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s', encoding='utf-8')

# Se não houver DISPLAY, encerra a aplicação antes de carregar módulos que exigem interface gráfica
if os.name != "nt" and not os.environ.get("DISPLAY"):
    logging.warning(
        "Variável DISPLAY ausente; encerrando a execução em modo de teste."
    )
    sys.exit(0)

# Adicionar o diretório pai ao sys.path para importações absolutas
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Importar os módulos da aplicação
from src.core import AppCore  # noqa: E402
from src.ui_manager import UIManager  # noqa: E402

# Ativar cudnn.benchmark quando CUDA estiver disponível (pode acelerar convoluções do encoder de áudio)
try:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
                logging.info("cudnn.benchmark ativado (CUDA disponível).")
            except Exception as e:
                logging.warning(f"Não foi possível ativar cudnn.benchmark: {e}")

            # Logs detalhados de capacidades CUDA (T5.2)
            try:
                num_gpus = torch.cuda.device_count()
                logging.info(f"CUDA disponível: {torch.version.cuda}; GPUs detectadas: {num_gpus}")
                for i in range(num_gpus):
                    try:
                        name = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        total_mem_gb = props.total_memory / (1024 ** 3)
                        capability = f"{props.major}.{props.minor}"
                        logging.info(f"GPU {i}: {name} | VRAM total: {total_mem_gb:.2f} GB | CC: {capability}")
                    except Exception as eg:
                        logging.warning(f"Falha ao consultar propriedades da GPU {i}: {eg}")

                # Detectar presença de FlashAttention 2
                try:
                    fa_spec = importlib.util.find_spec("flash_attn")
                    has_flash_attn = fa_spec is not None
                except Exception:
                    has_flash_attn = False
                logging.info(f"FlashAttention 2 detectado: {has_flash_attn}")
            except Exception as el:
                logging.warning(f"Não foi possível coletar capacidades CUDA: {el}")
except Exception as e:
    # torch pode não estar instalado em alguns ambientes de teste; seguir silenciosamente
    logging.debug(f"torch não disponível para configurar cudnn.benchmark: {e}")

# Log explícito de CPU quando nenhuma GPU disponível (TODO 4.3/5.2)
try:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            logging.info("[METRIC] stage=device_select device=cpu reason=no_cuda_available")
except Exception:
    pass
# --- Ajuste para evitar erros "main thread is not in main loop" ao destruir
# variáveis Tkinter quando a aplicação encerra. Mantemos o destrutor original
# em `_original_variable_del` para preservar comportamentos internos e facilitar
# a manutenção em futuras versões do Tkinter.
_original_variable_del = tk.Variable.__del__

def _safe_variable_del(self):
    tk_root = getattr(self, "_tk", None)
    if not tk_root:
        return
    try:
        if tk_root.getboolean(tk_root.call("info", "exists", self._name)):
            tk_root.globalunsetvar(self._name)
    except Exception:
        pass
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
    logging.info("Exit requested from tray icon.")
    if app_core_instance:
        app_core_instance.shutdown()
    if ui_manager_instance and ui_manager_instance.tray_icon:
        ui_manager_instance.tray_icon.stop()
    # Agenda o encerramento do mainloop na thread principal sem gerar exceção
    main_tk_root.after(0, main_tk_root.quit)


if __name__ == "__main__":
    atexit.register(lambda: logging.info("Application terminated."))

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
    app_core_instance.flush_pending_ui_notifications()

    # Sobrescrever a função on_exit_app original (para o menu do pystray)
    ui_manager_instance.on_exit_app = on_exit_app_enhanced

    logging.info("Starting the Tkinter mainloop on the main thread.")
    main_tk_root.mainloop()
    logging.info("Tkinter mainloop finished. The application will exit.")
    sys.exit(0)
