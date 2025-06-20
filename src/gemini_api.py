import logging
import time
from typing import Optional

import google.generativeai as genai


from .config_manager import ConfigManager  # Importar ConfigManager


class GeminiAPI:
    """
    A client for the Google Gemini API to correct transcribed text.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Gemini API client.

        Args:
            config_manager: Instância do ConfigManager para acessar as
                configurações.
            api_key: Opcional. A chave da API a ser usada. Se não for
                fornecida, será obtida do ConfigManager.
        """
        self.config_manager = config_manager
        self.client = None  # Inicializa o cliente Gemini
        self.model = None
        self.current_api_key = None
        self.current_model_id = None
        self.current_prompt = None
        # Armazena a chave passada na inicialização
        self.last_api_key = api_key
        self.last_model_id = None
        self.last_prompt = None
        # Novo atributo para indicar se a API está configurada e válida
        self.is_valid: bool = False

        # Inicializa o cliente Gemini
        self.reinitialize_client()

    def reinitialize_client(self):
        """
        Recarrega o cliente Gemini com as configurações mais recentes.
        Útil quando as configurações mudam em tempo de execução.
        """
        logging.info(
            "Gemini API client re/initializing due to external request."
        )
        self._load_model_from_config()

    def _load_model_from_config(self):
        """Recarrega o modelo Gemini quando chave ou modelo mudam.

        A chave da API é obtida da instância ou do ``ConfigManager``. O prompt
        é lido diretamente nos métodos públicos, portanto não participa do
        critério de reinicialização.
        """
        # Prioriza a chave da API passada no construtor, depois a do config
        self.current_api_key = (
            self.last_api_key
            or self.config_manager.get("gemini_api_key")
        )
        self.current_model_id = self.config_manager.get('gemini_model')
        self.current_prompt = self.config_manager.get('gemini_prompt')

        # A reinicialização considera apenas chave e modelo. O prompt é lido a
        # cada chamada pública, portanto não dispara reload.
        config_changed = (
            self.current_api_key != self.last_api_key
            or self.current_model_id != self.last_model_id
        )

        if self.model is None or config_changed:

            if not self.current_api_key or "SUA_CHAVE" in self.current_api_key:
                logging.warning(
                    "Gemini API Key não configurada ou inválida. "
                    "Correção de texto desativada."
                )
                self.model = None
                self.is_valid = False  # Chave inválida, marca como inválido
                self.last_api_key = self.current_api_key
                return

            try:
                genai.configure(api_key=self.current_api_key)
                self.model = genai.GenerativeModel(self.current_model_id)
                self.last_api_key = self.current_api_key
                self.last_model_id = self.current_model_id
                self.last_prompt = self.current_prompt
                # Sucesso na configuração, marca como válido
                self.is_valid = True
                logging.info(
                    "Gemini API client re/initialized with model: %s",
                    self.last_model_id,
                )
            except Exception as e:
                logging.error(
                    "Falha ao inicializar o cliente Gemini API: %s",
                    e,
                )
                self.model = None
                # Falha na configuração, marca como inválido
                self.is_valid = False
                # Não limpa a chave aqui para que o erro de chave inválida
                # persista até ser corrigido
                self.last_api_key = self.current_api_key
                self.last_model_id = None
                self.last_prompt = None

    def _execute_request(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: int = 1,
    ) -> str:
        """
        Executa uma requisição para a API Gemini com lógica de retry.
        Este é o método central para todas as chamadas da API.
        """
        self._load_model_from_config()
        if not prompt or not self.is_valid or not self.model:
            logging.warning(
                "Não é possível executar a requisição: prompt vazio, "
                "cliente inválido ou modelo não carregado."
            )
            return ""

        for attempt in range(max_retries):
            try:
                logging.info(
                    "Enviando prompt para a API Gemini usando o modelo %s "
                    "(tentativa %s/%s)",
                    self.last_model_id,
                    attempt + 1,
                    max_retries,
                )
                response = self.model.generate_content(prompt)

                if hasattr(response, 'text') and response.text:
                    generated_text = response.text.strip()
                    logging.info(
                        "Resposta recebida com sucesso da API Gemini."
                    )
                    return generated_text
                else:
                    logging.warning(
                        "API Gemini retornou resposta vazia (tentativa %s/%s)",
                        attempt + 1,
                        max_retries,
                    )

            except Exception as e:
                logging.error(
                    "Erro durante a geração de conteúdo da API Gemini "
                    "(tentativa %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
                    logging.info(
                        "Tentando novamente em %s segundos...",
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        logging.error(
            "Todas as tentativas de geração de conteúdo da API "
            "Gemini falharam."
        )
        return ""

    def get_correction(self, text: str) -> str:
        """
        Formata e executa uma requisição de correção de texto.
        """
        if not text:
            return ""
        correction_prompt_template = self.config_manager.get('gemini_prompt')
        full_prompt = correction_prompt_template.format(text=text)
        corrected_text = self._execute_request(full_prompt)
        return corrected_text if corrected_text else text

    def get_agent_response(self, text: str) -> str:
        """
        Formata e executa uma requisição do modo agente.
        """
        if not text:
            return ""
        agent_prompt_template = self.config_manager.get('prompt_agentico')
        full_prompt = f"{agent_prompt_template}\n\n{text}"
        original_model = self.current_model_id
        self.current_model_id = self.config_manager.get('gemini_agent_model')
        self.last_model_id = None
        agent_response = self._execute_request(full_prompt)
        self.current_model_id = original_model
        self.last_model_id = None
        return agent_response if agent_response else text
