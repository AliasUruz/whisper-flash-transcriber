import logging
from typing import Optional

from playwright.sync_api import BrowserContext, Page, sync_playwright

from .config_manager import (
    ConfigManager,
    CHATGPT_SELECTORS_CONFIG_KEY,
    CHATGPT_URL_CONFIG_KEY,
)


class ChatGPTAutomator:
    """Automatiza interações com a interface web do ChatGPT."""

    def __init__(self, config_manager: ConfigManager, user_data_dir: str, headless: bool = False) -> None:
        """Inicializa o automator com o gerenciador de configuração."""
        self._config_manager = config_manager
        self._user_data_dir = user_data_dir
        self._headless = headless
        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Inicia o navegador persistente e garante a abertura do ChatGPT."""
        self._logger.debug("Iniciando Playwright com user_data_dir=%s", self._user_data_dir)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            self._user_data_dir,
            headless=self._headless,
        )
        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self.ensure_chatgpt_open()

    def ensure_chatgpt_open(self) -> None:
        """Garante que a página do ChatGPT esteja aberta."""
        if not self._page:
            raise RuntimeError("Contexto não inicializado. Chame start() antes.")
        url = self._config_manager.config.get(CHATGPT_URL_CONFIG_KEY, "https://chatgpt.com/")
        self._logger.debug("Acessando URL do ChatGPT: %s", url)
        if not self._page.url.startswith(url):
            self._page.goto(url, wait_until="domcontentloaded")

    def transcribe_audio(self, audio_path: str) -> str:
        """Envia um arquivo de áudio e retorna o texto transcrito."""
        if not self._page:
            raise RuntimeError("Contexto não inicializado. Chame start() antes.")
        selectors = self._config_manager.config.get(CHATGPT_SELECTORS_CONFIG_KEY, {})
        file_input = selectors.get("file_input")
        response_block = selectors.get("response_block")
        if not file_input or not response_block:
            raise KeyError("Seletores necessários não configurados.")
        self._logger.debug("Enviando arquivo de áudio %s", audio_path)
        self._page.set_input_files(file_input, audio_path)
        self._logger.debug("Aguardando transcrição...")
        self._page.wait_for_selector(response_block)
        texto = self._page.inner_text(response_block)
        self._logger.debug("Transcrição recebida: %.60s", texto)
        return texto

    def close(self) -> None:
        """Fecha o navegador e libera recursos."""
        self._logger.debug("Encerrando contexto do navegador.")
        if self._context:
            self._context.close()
            self._context = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._page = None
        self._logger.debug("Navegador fechado com sucesso.")
