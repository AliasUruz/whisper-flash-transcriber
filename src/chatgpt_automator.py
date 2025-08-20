import logging
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Page, BrowserContext, Playwright

from .config_manager import (
    CHATGPT_URL_CONFIG_KEY,
    CHATGPT_SELECTORS_CONFIG_KEY,
)

class ChatGPTAutomator:
    """
    Gerencia a automação da interface web do ChatGPT usando Playwright.
    """
    def __init__(self, user_data_dir: Path, config_manager):
        self.user_data_dir = user_data_dir
        self.config_manager = config_manager
        self.playwright: Optional["Playwright"] = None
        self.browser: Optional["BrowserContext"] = None
        self.page: Optional["Page"] = None

    def _lista_seletores(self, chave: str, padrao: str) -> list:
        """Recupera lista de seletores a partir da configuração."""
        seletores = self.config_manager.get(CHATGPT_SELECTORS_CONFIG_KEY, {})
        valor = seletores.get(chave, padrao)
        return valor if isinstance(valor, list) else [valor]

    def _esperar_seletor(self, seletores: list, timeout: int = 30000) -> str:
        """Retorna o primeiro seletor disponível, testando em cascata."""
        ultimo_erro = None
        for seletor in seletores:
            try:
                self.page.wait_for_selector(seletor, timeout=timeout)
                return seletor
            except Exception as e:
                ultimo_erro = e
        raise ultimo_erro or ValueError("Nenhum seletor válido encontrado.")

    def start(self):
        """Inicia o Playwright e abre o navegador com um contexto persistente."""
        try:
            from playwright.sync_api import sync_playwright

            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            headless = self.config_manager.get("chatgpt_headless", False)
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=headless,
                slow_mo=50
            )
            self.page = self.browser.pages[0] if self.browser.pages else self.browser.new_page()
            logging.info("Navegador com contexto persistente iniciado.")
        except Exception as e:
            logging.error(f"Falha ao iniciar o Playwright: {e}")
            raise

    def ensure_chatgpt_open(self):
        """Garante que a página do ChatGPT esteja aberta e pronta."""
        if not self.page or self.page.is_closed():
            raise ConnectionError("A página do navegador não está disponível.")

        try:
            chatgpt_url = self.config_manager.get(CHATGPT_URL_CONFIG_KEY, "https://chatgpt.com/")
            if chatgpt_url not in self.page.url:
                self.page.goto(chatgpt_url, timeout=60000)

            seletor_prompt = self._esperar_seletor(self._lista_seletores("prompt", "#prompt-textarea"))
            self.page.wait_for_selector(seletor_prompt, timeout=30000)
            logging.info("Página do ChatGPT carregada.")
        except Exception as e:
            logging.error(f"Não foi possível carregar a página do ChatGPT. O usuário pode precisar fazer login. Erro: {e}")
            raise

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Faz o upload do arquivo de áudio e captura a transcrição resultante."""
        try:
            self.ensure_chatgpt_open()

            seletor_resposta = self._esperar_seletor(
                self._lista_seletores("resp", "div[data-message-author-role='assistant']")
            )
            seletor_plus = self._esperar_seletor(
                self._lista_seletores("plus", 'button[data-testid="composer-plus-btn"]')
            )
            seletor_upload = self._esperar_seletor(
                self._lista_seletores("upload", 'input[type="file"]')
            )
            seletor_send = self._esperar_seletor(
                self._lista_seletores("send", 'button[data-testid="send-button"]')
            )
            contagem_inicial = self.page.locator(seletor_resposta).count()

            self.page.click(seletor_plus)
            self.page.set_input_files(seletor_upload, audio_file_path)

            self.page.wait_for_selector(f"{seletor_send}:not([disabled])", timeout=20000)
            self.page.click(seletor_send)

            self.page.locator(seletor_resposta).nth(contagem_inicial).wait_for(timeout=60000)

            transcribed_text = self.page.locator(f"{seletor_resposta} .markdown").last.inner_text()
            logging.info("Transcrição via ChatGPT (Web) capturada com sucesso.")
            return transcribed_text
        except Exception as e:
            logging.error(f"Falha no processo de transcrição automatizada: {e}")
            return None

    def close(self):
        """Encerra a sessão de automação de forma segura."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        logging.info("Sessão de automação web encerrada.")
