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
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False
    # Crie classes dummy para evitar erros de tipo se a biblioteca não estiver instalada

    class BrokenResponseError(Exception):
        pass


    class IncompleteIterationError(Exception):
        pass

    class helper_types:
        class RequestOptions:
            def __init__(self, timeout=None):
                pass

from .config_manager import ConfigManager
from .config_manager import GEMINI_PROMPT_CONFIG_KEY


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
        self.current_api_key: Optional[str] = None
        self.override_api_key: Optional[str] = api_key
        self.correction_model_id: Optional[str] = None
        self.agent_model_id: Optional[str] = None
        self.correction_model: Any | None = None
        self.agent_model: Any | None = None
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
        self._load_models_from_config()

    def _reset_models(self) -> None:
        self.correction_model = None
        self.agent_model = None
        self.correction_model_id = None
        self.agent_model_id = None

    def _initialize_model(self, model_id: Optional[str], purpose: str) -> Any | None:
        if not model_id:
            logging.warning(
                "No Gemini %s model configured.",
                purpose,
            )
            return None

        try:
            model = genai.GenerativeModel(model_id)
            logging.info(
                "Gemini %s model initialized: %s",
                purpose,
                model_id,
            )
            return model
        except Exception as e:
            logging.error(
                "Failed to initialize the Gemini %s model '%s': %s",
                purpose,
                model_id,
                e,
            )
            return None

    def _load_models_from_config(self) -> None:
        """Carrega os modelos de correção e agente usando as configurações atuais."""
        self._reset_models()

        if not GEMINI_API_AVAILABLE:
            self.is_valid = False
            return

        api_key = self.override_api_key or self.config_manager.get("gemini_api_key")

        if not api_key or "SUA_CHAVE" in api_key:
            logging.warning(
                "Gemini API key is missing or invalid. Text correction disabled."
            )
            self.is_valid = False
            self.current_api_key = None
            return

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logging.error(
                "Failed to configure the Gemini API client: %s",
                e,
            )
            self.is_valid = False
            self.current_api_key = None
            return

        self.current_api_key = api_key

        correction_model_id = self.config_manager.get("gemini_model")
        agent_model_id = self.config_manager.get("gemini_agent_model")

        correction_model = self._initialize_model(
            correction_model_id,
            "text correction",
        )
        agent_model = self._initialize_model(
            agent_model_id,
            "agent",
        )

        self.correction_model = correction_model
        self.agent_model = agent_model
        self.correction_model_id = correction_model_id
        self.agent_model_id = agent_model_id
        self.is_valid = self.correction_model is not None

    def _execute_request(
        self,
        prompt: str,
        model: Any | None,
        model_id: Optional[str],
        max_retries: int = 3,
        retry_delay: int = 1,
        timeout: int | float = 120,
    ) -> str:
        """
        Executa uma requisição para a API Gemini com lógica de retry.
        """
        if not prompt:
            logging.warning("Cannot execute request: empty prompt provided.")
            return ""

        if model is None:
            logging.warning(
                "Cannot execute request: model '%s' is not available.",
                model_id or "<unknown>",
            )
            return ""

        for attempt in range(max_retries):
            try:
                logging.info(
                    "Sending prompt to the Gemini API with model %s (attempt %s/%s)",
                    model_id or "<unknown>",
                    attempt + 1,
                    max_retries,
                )
                response = model.generate_content(
                    prompt,
                    request_options=helper_types.RequestOptions(timeout=timeout),
                )

                if hasattr(response, 'text') and response.text:
                    generated_text = response.text.strip()
                    logging.info(
                        "Gemini API returned a successful response."
                    )
                    return generated_text
                else:
                    logging.warning(
                        "Gemini API returned an empty response (attempt %s/%s)",
                        attempt + 1,
                        max_retries,
                    )

            except (BrokenResponseError, IncompleteIterationError) as e:
                logging.error(
                    "Gemini API specific error (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
            except Exception as e:
                logging.error(
                    "Error while generating content with the Gemini API (attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )
                if attempt < max_retries - 1:
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
        if not text or not self.is_valid or not self.correction_model:
            return text

        correction_prompt_template = self.config_manager.get(
            GEMINI_PROMPT_CONFIG_KEY
        )
        if not correction_prompt_template:
            logging.warning("Gemini correction prompt template is empty.")
            return text

        full_prompt = correction_prompt_template.format(text=text)
        corrected_text = self._execute_request(
            full_prompt,
            self.correction_model,
            self.correction_model_id,
        )
        return corrected_text if corrected_text else text

    def get_agent_response(self, text: str) -> str:
        """
        Formata e executa uma requisição do modo agente.
        """
        if not text or not self.agent_model:
            return text
        agent_prompt_template = self.config_manager.get('prompt_agentico')
        if not agent_prompt_template:
            logging.warning("Gemini agent prompt template is empty.")
            return text

        full_prompt = f"{agent_prompt_template}\n\n{text}"
        agent_response = self._execute_request(
            full_prompt,
            self.agent_model,
            self.agent_model_id,
        )
        return agent_response if agent_response else text

    def correct_text_async(self, text: str, prompt: str, api_key: str) -> str:
        if not self.is_valid or not self.correction_model:
            return text

        if not prompt:
            logging.warning("Empty Gemini prompt received for async correction.")
            return text

        if api_key and self.current_api_key and api_key != self.current_api_key:
            logging.debug(
                "Async correction received API key different from configured key. Using configured client instead."
            )

        full_prompt = prompt.format(text=text)
        return (
            self._execute_request(
                full_prompt,
                self.correction_model,
                self.correction_model_id,
            )
            or text
        )
