import json
import logging
import time
from typing import Any, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import (
        helper_types,
        BrokenResponseError,
        IncompleteIterationError,
    )
    try:
        from google.api_core import exceptions as google_api_exceptions
    except ImportError:
        google_api_exceptions = None
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False
    google_api_exceptions = None
    # Crie classes dummy para evitar erros de tipo se a biblioteca não estiver instalada

    class BrokenResponseError(Exception):
        pass


    class IncompleteIterationError(Exception):
        pass

    class helper_types:
        class RequestOptions:
            def __init__(self, timeout=None):
                pass

if google_api_exceptions is not None:
    GoogleAPIError = google_api_exceptions.GoogleAPIError
    GoogleAPITimeoutError = getattr(
        google_api_exceptions,
        "DeadlineExceeded",
        google_api_exceptions.GoogleAPIError,
    )
else:
    class GoogleAPIError(Exception):
        """Fallback genérico quando google-api-core não está disponível."""


    class GoogleAPITimeoutError(GoogleAPIError):
        """Fallback genérico para timeouts quando google-api-core não está disponível."""

from .config_manager import (
    ConfigManager,
    GEMINI_PROMPT_CONFIG_KEY,
    GEMINI_TIMEOUT_CONFIG_KEY,
)


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
        self.client = None
        self.model = None
        self.current_api_key = None
        self.current_model_id = None
        self.current_prompt = None
        self.last_api_key = api_key
        self.last_model_id = None
        self.last_prompt = None
        self.is_valid: bool = False

        if not GEMINI_API_AVAILABLE:
            logging.warning("Google Generative AI SDK not found. Gemini features will be disabled.")
            self.is_valid = False
        else:
            self.reinitialize_client()

    def reinitialize_client(self):
        """
        Recarrega o cliente Gemini com as configurações mais recentes.
        """
        if not GEMINI_API_AVAILABLE:
            self.is_valid = False
            return

        logging.info(
            "Gemini API client re/initializing due to external request."
        )
        self._load_model_from_config()

    def _load_model_from_config(self):
        """Recarrega o modelo Gemini quando chave ou modelo mudam."""
        if not GEMINI_API_AVAILABLE:
            self.is_valid = False
            return

        self.current_api_key = (
            self.last_api_key or self.config_manager.get("gemini_api_key")
        )
        self.current_model_id = self.config_manager.get('gemini_model')
        self.current_prompt = self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY)

        key_changed = self.current_api_key != self.last_api_key
        model_changed = self.current_model_id != self.last_model_id
        config_changed = key_changed or model_changed

        if self.model is None or config_changed:
            if not self.current_api_key or "SUA_CHAVE" in self.current_api_key:
                logging.warning(
                    "Gemini API key is missing or invalid. Text correction disabled."
                )
                self.model = None
                self.is_valid = False
                self.last_api_key = self.current_api_key
                return

            try:
                genai.configure(api_key=self.current_api_key)
                self.model = genai.GenerativeModel(self.current_model_id)
                self.last_api_key = self.current_api_key
                self.last_model_id = self.current_model_id
                self.last_prompt = self.current_prompt
                self.is_valid = True
                logging.info(
                    "Gemini API client re/initialized with model: %s",
                    self.last_model_id,
                )
            except Exception as e:
                logging.error(
                    "Failed to initialize the Gemini API client: %s",
                    e,
                )
                self.model = None
                self.is_valid = False
                self.last_api_key = self.current_api_key
                self.last_model_id = self.current_model_id
                self.last_prompt = self.current_prompt

    def _resolve_timeout(self, default_timeout: float | int) -> float:
        """Obtém o timeout configurado garantindo um valor positivo."""
        configured_timeout = self.config_manager.get_timeout(
            GEMINI_TIMEOUT_CONFIG_KEY,
            default_timeout,
        )
        return float(configured_timeout)

    @staticmethod
    def _format_response_for_logging(response: Any) -> str:
        """Serializa respostas arbitrárias em uma string amigável para logs."""
        if response is None:
            return "None"

        if hasattr(response, "to_dict"):
            try:
                response_dict = response.to_dict()
                return json.dumps(response_dict, ensure_ascii=False)
            except Exception:
                return repr(response)

        try:
            return json.dumps(response, ensure_ascii=False)
        except TypeError:
            return repr(response)

    def _execute_request(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: int = 1,
        timeout: int | float = 120,
    ) -> str:
        """
        Executa uma requisição para a API Gemini com lógica de retry.
        """
        if not prompt or not self.is_valid or not self.model:
            logging.warning(
                "Cannot execute request: empty prompt, invalid client, or model not loaded."
            )
            return ""

        timeout_value = self._resolve_timeout(timeout)
        model_for_log = self.current_model_id or self.last_model_id or "unknown"

        for attempt in range(max_retries):
            attempt_number = attempt + 1
            should_retry = False
            try:
                logging.info(
                    "Sending prompt to the Gemini API with model %s (attempt %s/%s)",
                    model_for_log,
                    attempt_number,
                    max_retries,
                )
                logging.debug(
                    "Gemini prompt payload for model '%s' (attempt %s/%s): %s",
                    model_for_log,
                    attempt_number,
                    max_retries,
                    prompt,
                )
                response = self.model.generate_content(
                    prompt,
                    request_options=helper_types.RequestOptions(timeout=timeout_value),
                )
                logging.debug(
                    "Gemini raw response for model '%s' (attempt %s/%s): %s",
                    model_for_log,
                    attempt_number,
                    max_retries,
                    self._format_response_for_logging(response),
                )

                if hasattr(response, 'text') and response.text:
                    generated_text = response.text.strip()
                    logging.info(
                        "Gemini API returned a successful response."
                    )
                    return generated_text

                logging.warning(
                    "Gemini API returned an empty response (attempt %s/%s)",
                    attempt_number,
                    max_retries,
                )
                should_retry = True

            except GoogleAPITimeoutError as e:
                logging.error(
                    "Gemini API request timed out after %.2f seconds (attempt %s/%s): %s",
                    timeout_value,
                    attempt_number,
                    max_retries,
                    e,
                )
                should_retry = True
            except (BrokenResponseError, IncompleteIterationError) as e:
                logging.error(
                    "Gemini API specific error (attempt %s/%s): %s",
                    attempt_number,
                    max_retries,
                    e,
                )
                should_retry = True
            except GoogleAPIError as e:
                logging.error(
                    "Gemini API returned an error response (attempt %s/%s): %s",
                    attempt_number,
                    max_retries,
                    e,
                )
                should_retry = True
            except Exception as e:
                logging.error(
                    "Error while generating content with the Gemini API (attempt %s/%s): %s",
                    attempt_number,
                    max_retries,
                    e,
                    exc_info=True,
                )
                should_retry = True

            if should_retry and attempt < max_retries - 1:
                logging.info(
                    "Retrying in %s seconds...",
                    retry_delay,
                )
                time.sleep(retry_delay)

        logging.error(
            "All attempts to generate content with the Gemini API failed."
        )
        return ""

    def get_correction(self, text: str) -> str:
        """
        Formata e executa uma requisição de correção de texto.
        """
        if not text or not self.is_valid:
            return text
        correction_prompt_template = self.config_manager.get(
            GEMINI_PROMPT_CONFIG_KEY
        )
        full_prompt = correction_prompt_template.format(text=text)
        corrected_text = self._execute_request(full_prompt)
        return corrected_text if corrected_text else text

    def get_agent_response(self, text: str) -> str:
        """
        Formata e executa uma requisição do modo agente.
        """
        if not text or not self.is_valid:
            return text
        agent_prompt_template = self.config_manager.get('prompt_agentico')
        full_prompt = f"{agent_prompt_template}\n\n{text}"
        original_model = self.current_model_id
        original_last_model = self.last_model_id
        original_config_model = self.config_manager.get('gemini_model')
        self.current_model_id = self.config_manager.get('gemini_agent_model')
        self.last_model_id = None
        self.config_manager.set('gemini_model', self.current_model_id)
        self._load_model_from_config()
        agent_response = self._execute_request(full_prompt)
        self.config_manager.set('gemini_model', original_config_model)
        self.current_model_id = original_model
        self.last_model_id = original_last_model
        self._load_model_from_config()
        return agent_response if agent_response else text

    def correct_text_async(self, text: str, prompt: str, api_key: str) -> str:
        if not self.is_valid:
            return text
        self.last_api_key = api_key
        self.current_api_key = api_key
        self.current_prompt = prompt
        full_prompt = prompt.format(text=text)
        return self._execute_request(full_prompt) or text
