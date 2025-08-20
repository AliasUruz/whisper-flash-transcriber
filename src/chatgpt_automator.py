import logging
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, BrowserContext, Playwright
from typing import Optional

class ChatGPTAutomator:
    """
    Gerencia a automação da interface web do ChatGPT usando Playwright.
    """
    def __init__(self, user_data_dir: Path, config_manager):
        self.user_data_dir = user_data_dir
        self.config_manager = config_manager
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def start(self):
        """Inicia o Playwright e abre o navegador com um contexto persistente."""
        try:
            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=False,
                slow_mo=50
            )
            self.page = self.browser.pages[0] if self.browser.pages else self.browser.new_page()
            logging.info("Navegador com contexto persistente iniciado.")
        except Exception as e:
            logging.error(f"Falha ao iniciar o Playwright: {e}")
            raise

    def _handle_login_or_consent(self) -> bool:
        """Detecta telas de **login** ou de *consentimento* e redireciona o usuário."""
        if not self.page:
            return False

        labels_tipicas = [
            "Log in",
            "Sign in",
            "Entrar",
            "Aceitar",
            "Concordo",
        ]

        if self.page.locator("form").count() > 0:
            login_url = "https://chatgpt.com/auth/login"
            logging.warning(
                "Tela de **login/consentimento** detectada. Abrindo a URL de autenticação: %s",
                login_url,
            )
            self.page.goto(login_url, timeout=60000)
            return True

        for rotulo in labels_tipicas:
            if self.page.get_by_role("button", name=rotulo, exact=False).count() > 0:
                login_url = "https://chatgpt.com/auth/login"
                logging.warning(
                    "Elemento '%s' detectado. Redirecionando para a página de **login**: %s",
                    rotulo,
                    login_url,
                )
                self.page.goto(login_url, timeout=60000)
                return True

        return False

    def ensure_chatgpt_open(self):
        """Garante que a página do ChatGPT esteja aberta e pronta."""
        if not self.page or self.page.is_closed():
            raise ConnectionError("A página do navegador não está disponível.")

        try:
            if "chatgpt.com" not in self.page.url:
                self.page.goto("https://chatgpt.com/", timeout=60000)

            if self._handle_login_or_consent():
                raise ConnectionError("Autenticação necessária para prosseguir.")

            prompt_selector = self.config_manager.get("chatgpt_selectors", {}).get("prompt_textarea", "#prompt-textarea")
            self.page.wait_for_selector(prompt_selector, timeout=30000)
            logging.info("Página do ChatGPT carregada.")
        except Exception as e:
            logging.error(f"Não foi possível carregar a página do ChatGPT. O usuário pode precisar fazer login. Erro: {e}")
            raise

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Faz o upload do arquivo de áudio e captura a transcrição resultante."""
        try:
            self.ensure_chatgpt_open()

            selectors = self.config_manager.get("chatgpt_selectors", {})
            response_selector = selectors.get("response_container", "div[data-message-author-role='assistant']")
            attach_button_selector = selectors.get("attach_button", 'button[data-testid="composer-plus-btn"]')
            send_button_selector = selectors.get("send_button", 'button[data-testid="send-button"]')

            initial_response_count = self.page.locator(response_selector).count()

            with self.page.expect_file_chooser() as fc_info:
                self.page.click(attach_button_selector)
            file_chooser = fc_info.value
            file_chooser.set_files(audio_file_path)

            self.page.wait_for_selector(f"{send_button_selector}:not([disabled])", timeout=20000)
            self.page.click(send_button_selector)

            self.page.locator(response_selector).nth(initial_response_count).wait_for(timeout=60000)

            transcribed_text = self.page.locator(f"{response_selector} .markdown").last.inner_text()
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
