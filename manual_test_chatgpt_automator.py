"""Teste manual para o ChatGPTAutomator."""
import logging
import time

from src.config_manager import ConfigManager
from src.chatgpt_automator import ChatGPTAutomator


def main() -> None:
    """Executa um ciclo simples de abertura e fechamento do navegador."""
    logging.basicConfig(level=logging.DEBUG)
    cfg = ConfigManager()
    automator = ChatGPTAutomator(cfg, user_data_dir="/tmp/chatgpt_profile", headless=True)
    try:
        automator.start()
        logging.info("Navegador iniciado com sucesso. Aguardando alguns segundos...")
        time.sleep(2)
    finally:
        automator.close()
        logging.info("Navegador encerrado sem erros.")


if __name__ == "__main__":
    main()
