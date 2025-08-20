import logging
import time
from pathlib import Path
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - apenas para *type hints*
    from playwright.sync_api import Page, BrowserContext, Playwright

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

    def _get_selector(self, keys: list) -> Optional[str]:
        """Retorna o primeiro seletor disponível em ``chatgpt_selectors``.

        Útil para permitir *fallback* de múltiplas chaves de configuração.
        """
        seletores = self.config_manager.get(CHATGPT_SELECTORS_CONFIG_KEY, {})
        for chave in keys:
            seletor = seletores.get(chave)
            if seletor:
                return seletor
        return None

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

    def _executar_com_retry(self, acao: Callable, tentativas: int = 3, intervalo: float = 0.5) -> int:
        """Executa ``acao`` com *retries* e **backoff exponencial**.

        Retorna o número de tentativas necessárias até o sucesso.
        """
        for tentativa in range(1, tentativas + 1):
            try:
                acao()
                return tentativa
            except Exception as e:
                if tentativa == tentativas:
                    raise
                espera = intervalo * (2 ** (tentativa - 1))
                logging.warning(
                    "Ação sensível falhou (tentativa %d/%d): %s. Nova tentativa em %.1fs.",
                    tentativa,
                    tentativas,
                    e,
                    espera,
                )
                time.sleep(espera)
        return tentativas

    def start(self):
        """Inicia o Playwright e abre o navegador com um contexto persistente."""
        sucesso = False
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

            self.user_data_dir.mkdir(parents=True, exist_ok=True)
            headless = bool(self.config_manager.get("chatgpt_headless", False))
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch_persistent_context(
                self.user_data_dir,
                headless=headless,
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

    def _handle_login_or_consent(self) -> None:
        """Detecta telas de **login** ou de *consentimento* e aguarda resolução.

        Se após **duas** tentativas de espera (30s cada) a página continuar
        exigindo autenticação, lança ``ConnectionError`` orientando o usuário
        a realizar o login manualmente ou configurar cookies/credenciais.
        """
        if not self.page:
            return

        labels_tipicas = [
            "Log in",
            "Sign in",
            "Entrar",
            "Aceitar",
            "Concordo",
        ]

        for tentativa in range(3):
            possui_form = self.page.locator("form").count() > 0
            possui_botao = any(
                self.page.get_by_role("button", name=rot, exact=False).count() > 0
                for rot in labels_tipicas
            )
            if not (possui_form or possui_botao):
                return

            logging.warning(
                "Tela de **login/consentimento** detectada. "
                "Realize a autenticação manualmente ou configure cookies. "
                "Tentativa %d/3.",
                tentativa + 1,
            )
            try:
                self.page.wait_for_timeout(30_000)
            except Exception:
                pass

        raise ConnectionError("Autenticação necessária para prosseguir.")

    def ensure_chatgpt_open(self):
        """Garante que a página do ChatGPT esteja aberta e pronta."""
        if not self.page or self.page.is_closed():
            raise ConnectionError("A página do navegador não está disponível.")

        try:
            chatgpt_url = self.config_manager.get(CHATGPT_URL_CONFIG_KEY, "https://chatgpt.com/")
            if chatgpt_url not in self.page.url:
                self.page.goto(chatgpt_url, timeout=60000)

            self._handle_login_or_consent()

            prompt_selector = self.config_manager.get("chatgpt_selectors", {}).get("prompt_textarea", "#prompt-textarea")
            logging.info("Página do ChatGPT carregada.")
        except Exception as e:
            logging.error(f"Não foi possível carregar a página do ChatGPT. O usuário pode precisar fazer login. Erro: {e}")
            raise

    def _esperar_resposta_streaming(
        self,
        response_selector: str,
        inicio_envio: float,
        timeout: int = 120_000,
        estabilidade_ms: int = 500,
    ) -> tuple[str, float | None]:
        """Acompanha o *streaming* da resposta do assistente.

        Retorna o texto final e o tempo até o **primeiro token** em segundos.
        O algoritmo monitora se um novo ``data-message-id`` surge ou se o
        conteúdo atual permanece estável por ``estabilidade_ms``.
        """

        ultimo_id = None
        ultimo_tamanho = 0
        tempo_primeiro_token = None
        inicio_estavel = time.time()
        inicio_total = time.time()
        atraso = 0.25

        while (time.time() - inicio_total) * 1000 < timeout:
            blocos = self.page.locator(response_selector)
            if blocos.count() == 0:
                time.sleep(atraso)
                atraso = min(atraso * 1.5, 5)
                continue

            bloco = blocos.last
            data_id = bloco.get_attribute("data-message-id")
            texto = bloco.inner_text()

            if data_id != ultimo_id:
                ultimo_id = data_id
                ultimo_tamanho = len(texto)
                inicio_estavel = time.time()
                if tempo_primeiro_token is None and texto:
                    tempo_primeiro_token = time.time() - inicio_envio
            elif len(texto) != ultimo_tamanho:
                ultimo_tamanho = len(texto)
                inicio_estavel = time.time()
                if tempo_primeiro_token is None and texto:
                    tempo_primeiro_token = time.time() - inicio_envio

            if (time.time() - inicio_estavel) * 1000 >= estabilidade_ms:
                return texto, tempo_primeiro_token

            time.sleep(atraso)
            atraso = min(atraso * 1.5, 5)

        # Timeout: retorna o texto atual mesmo assim
        return self.page.locator(response_selector).last.inner_text(), tempo_primeiro_token

    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """Faz o upload do arquivo de áudio e captura a transcrição resultante."""
        sucesso = False
        total_retries = 0
        try:
            self.ensure_chatgpt_open()

            response_selector = self._get_selector(["response_container"]) or "div[data-message-author-role='assistant']"
            attach_button_selector = self._get_selector(["attach_button"]) or 'button[data-testid="composer-plus-btn"]'
            send_button_selector = self._get_selector(["send_button"]) or 'button[data-testid="send-button"]'
            file_input_selector = self._get_selector(["file_input"])

            tentativas = int(self.config_manager.get("chatgpt_retry_attempts", 3))
            intervalo = float(self.config_manager.get("chatgpt_retry_interval", 0.5))

            logging.info("Iniciando upload do áudio '%s'.", audio_file_path)
            inicio_upload = time.time()

            if file_input_selector:
                # Preferível: ``file_input`` deve apontar para ``input[type=file]`` do DOM.
                def enviar_arquivo():
                    self.page.locator(file_input_selector).set_input_files(audio_file_path)

                total_retries += self._executar_com_retry(enviar_arquivo, tentativas, intervalo) - 1
            else:
                # *Fallback* para o método antigo usando ``expect_file_chooser``.
                def abrir_menu():
                    self.page.click(attach_button_selector)

                def selecionar_upload():
                    with self.page.expect_file_chooser() as fc_info:
                        self.page.get_by_text("Upload files").click()
                    fc_info.value.set_files(audio_file_path)

                total_retries += self._executar_com_retry(abrir_menu, tentativas, intervalo) - 1
                total_retries += self._executar_com_retry(selecionar_upload, tentativas, intervalo) - 1

            logging.info("Upload concluído em %.2fs.", time.time() - inicio_upload)

            def esperar_botao_enviar():
                self.page.wait_for_selector(
                    f"{send_button_selector}:not([disabled])", timeout=20_000
                )

            total_retries += self._executar_com_retry(esperar_botao_enviar, tentativas, intervalo) - 1

            def clicar_enviar():
                self.page.click(send_button_selector)

            total_retries += self._executar_com_retry(clicar_enviar, tentativas, intervalo) - 1
            inicio_envio = time.time()

            texto, tempo_primeiro_token = self._esperar_resposta_streaming(
                response_selector, inicio_envio
            )
            if tempo_primeiro_token is not None:
                logging.info("Tempo até o primeiro token: %.2fs.", tempo_primeiro_token)

            logging.info("Total de retries executados: %d.", total_retries)
            logging.info("Transcrição via ChatGPT (Web) capturada com sucesso.")
            sucesso = True
            return texto
        except Exception as e:
            logging.error(f"Falha no processo de transcrição automatizada: {e}")
            return None
        finally:
            if not sucesso:
                try:
                    self.close()
                except Exception as err:
                    logging.error(f"Erro ao encerrar sessão após falha de transcrição: {err}")

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
