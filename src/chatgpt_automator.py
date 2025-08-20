import logging
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, BrowserContext, Playwright
from typing import Optional, Callable

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

    def _executar_com_retry(self, acao: Callable, tentativas: int, intervalo: float):
        """Executa ``acao`` com retries e backoff exponencial."""
        for tentativa in range(1, tentativas + 1):
            try:
                return acao()
            except Exception as e:
                if tentativa == tentativas:
                    raise
                espera = intervalo * (2 ** (tentativa - 1))
                logging.warning(
                    f"Ação sensível falhou (tentativa {tentativa}/{tentativas}): {e}. "
                    f"Nova tentativa em {espera}s."
                )
                time.sleep(espera)

    def start(self):
        """Inicia o Playwright e abre o navegador com um contexto persistente."""
        sucesso = False
        try:
            from playwright.sync_api import sync_playwright

            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            headless = self.config_manager.get("chatgpt_headless", False)
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=False,
                slow_mo=50,
            )
            self.page = self.browser.pages[0] if self.browser.pages else self.browser.new_page()
            logging.info("Navegador com contexto persistente iniciado.")
            sucesso = True
        except Exception as e:
            logging.error(f"Falha ao iniciar o Playwright: {e}")
            raise
        finally:
            if not sucesso:
                try:
                    self.close()
                except Exception as err:
                    logging.error(f"Falha ao encerrar sessão após erro de inicialização: {err}")

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
            chatgpt_url = self.config_manager.get(CHATGPT_URL_CONFIG_KEY, "https://chatgpt.com/")
            if chatgpt_url not in self.page.url:
                self.page.goto(chatgpt_url, timeout=60000)

            if self._handle_login_or_consent():
                raise ConnectionError("Autenticação necessária para prosseguir.")

            prompt_selector = self.config_manager.get("chatgpt_selectors", {}).get("prompt_textarea", "#prompt-textarea")
            logging.info("Página do ChatGPT carregada.")
        except Exception as e:
            logging.error(f"Não foi possível carregar a página do ChatGPT. O usuário pode precisar fazer login. Erro: {e}")
            raise

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Faz o upload do arquivo de áudio e captura a transcrição resultante."""
        sucesso = False
        try:
            self.ensure_chatgpt_open()

            selectors = self.config_manager.get("chatgpt_selectors", {})
            response_selector = selectors.get("response_container", "div[data-message-author-role='assistant']")
            attach_button_selector = selectors.get("attach_button", 'button[data-testid="composer-plus-btn"]')
            send_button_selector = selectors.get("send_button", 'button[data-testid="send-button"]')

            tentativas = self.config_manager.get("chatgpt_retry_attempts", 3)
            intervalo = self.config_manager.get("chatgpt_retry_interval", 1.0)

            initial_response_count = self.page.locator(response_selector).count()

            def abrir_menu():
                self.page.click(attach_button_selector)

            def selecionar_upload():
                with self.page.expect_file_chooser() as fc_info:
                    self.page.get_by_text("Upload files").click()
                fc_info.value.set_files(audio_file_path)

            def esperar_botao_enviar():
                self.page.wait_for_selector(
                    f"{send_button_selector}:not([disabled])", timeout=20000
                )

            def clicar_enviar():
                self.page.click(send_button_selector)

            def aguardar_resposta():
                self.page.locator(response_selector).nth(initial_response_count).wait_for(
                    timeout=60000
                )

            self._executar_com_retry(abrir_menu, tentativas, intervalo)
            self._executar_com_retry(selecionar_upload, tentativas, intervalo)
            self._executar_com_retry(esperar_botao_enviar, tentativas, intervalo)
            self._executar_com_retry(clicar_enviar, tentativas, intervalo)
            self._executar_com_retry(aguardar_resposta, tentativas, intervalo)

            transcribed_text = self.page.locator(f"{seletor_resposta} .markdown").last.inner_text()
            logging.info("Transcrição via ChatGPT (Web) capturada com sucesso.")
            sucesso = True
            return transcribed_text
        except Exception as e:
            logging.error(f"Falha no processo de transcrição automatizada: {e}")
            return None
        finally:
            if not sucesso:
                try:
                    self.close()
                except Exception as err:
                    logging.error(f"Erro ao encerrar sessão após falha de transcrição: {err}")

    def _wait_for_assistant_finalization(self, response_selector: str, appear_timeout: int, finalize_timeout: int):
        """Aguarda o surgimento do último bloco de resposta do assistente e sua finalização."""
        # Localizador para o bloco mais recente do assistente
        response_block = self.page.locator(response_selector).last
        # Aguarda o bloco aparecer na página
        response_block.wait_for(timeout=appear_timeout)
        # Aguarda desaparecer qualquer indicador de carregamento ou rascunho
        self.page.wait_for_function(
            """
            (el) => {
                const html = el.innerHTML.toLowerCase();
                const hasDraft = html.includes('draft');
                const hasSpinner = el.querySelector('.spinner, [data-testid="loading-spinner"], svg[aria-label="Stop generating"]');
                return !(hasDraft || hasSpinner);
            }
            """,
            arg=response_block,
            timeout=finalize_timeout,
        )
        return response_block

    def close(self):
        """Encerra a sessão de automação de forma segura."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        self.browser = None
        self.playwright = None
        self.page = None
        logging.info("Sessão de automação web encerrada.")
