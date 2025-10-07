import json
import logging
import time
from typing import Any, Optional

from .config_manager import (
    ConfigManager,
    GEMINI_PROMPT_CONFIG_KEY,
    GEMINI_MAX_ATTEMPTS_CONFIG_KEY,
    GEMINI_TIMEOUT_CONFIG_KEY,
)
from .logging_utils import get_logger, log_context
from .retry_utils import RetryableOperationError, retry_with_backoff

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

LOGGER = get_logger('whisper_flash_transcriber.gemini', component='GeminiAPI')

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
            LOGGER.warning('Google Generative AI SDK not found. Gemini features will be disabled.')
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

        LOGGER.info(
            log_context(
                'Gemini API client re/initializing due to external request.',
                event='gemini.client_reinitialized',
            )
        )
        self._load_models_from_config()

    def _reset_models(self) -> None:
        self.correction_model = None
        self.agent_model = None
        self.correction_model_id = None
        self.agent_model_id = None

    def _initialize_model(self, model_id: Optional[str], purpose: str) -> Any | None:
        if not model_id:
            LOGGER.warning(
                "No Gemini %s model configured.",
                purpose,
            )
            return None

        try:
            model = genai.GenerativeModel(model_id)
            LOGGER.info(
                "Gemini %s model initialized: %s",
                purpose,
                model_id,
            )
            return model
        except Exception as e:
            LOGGER.error(
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
            LOGGER.warning(
                "Gemini API key is missing or invalid. Text correction disabled."
            )
            self.is_valid = False
            self.current_api_key = None
            return

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            LOGGER.error(
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
        self.is_valid = any((self.correction_model, self.agent_model))

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
        model: Any | None,
        model_id: Optional[str],
        *,
        base_delay: float = 1.0,
        max_delay: float = 16.0,
        jitter_factor: float = 0.25,
        timeout: int | float = 120,
    ) -> str:
        """Executa uma requisição para a API Gemini com lógica de retry."""
        if not prompt:
            LOGGER.warning("Cannot execute request: empty prompt provided.")
            return ""

        if model is None:
            LOGGER.warning(
                "Cannot execute request: model '%s' is not available.",
                model_id or "<unknown>",
            )
            return ""

        timeout_value = self._resolve_timeout(timeout)
        model_for_log = model_id or "<unknown>"
        default_attempts = self.config_manager.default_config.get(
            GEMINI_MAX_ATTEMPTS_CONFIG_KEY,
            3,
        )
        max_attempts = self.config_manager.get_retry_attempts(
            GEMINI_MAX_ATTEMPTS_CONFIG_KEY,
            default_attempts,
        )
        operation_id = f"gemini:{model_for_log}:{time.monotonic_ns()}"
        retryable_error_codes = (429, 500, 502, 503, 504)

        def _generate(attempt_number: int, total_attempts: int) -> str:
            LOGGER.info(
                "Sending prompt to the Gemini API with model %s (attempt %s/%s)",
                model_for_log,
                attempt_number,
                total_attempts,
            )
            LOGGER.debug(
                "Gemini prompt payload for model '%s' (attempt %s/%s): %s",
                model_for_log,
                attempt_number,
                total_attempts,
                prompt,
            )
            try:
                response = model.generate_content(
                    prompt,
                    request_options=helper_types.RequestOptions(timeout=timeout_value),
                )
            except GoogleAPITimeoutError as exc:
                LOGGER.error(
                    "Gemini API request timed out after %.2f seconds (attempt %s/%s): %s",
                    timeout_value,
                    attempt_number,
                    total_attempts,
                    exc,
                )
                raise RetryableOperationError(
                    "Gemini API request timed out.",
                    error_code=getattr(exc, "code", "timeout"),
                    retryable=True,
                ) from exc
            except (BrokenResponseError, IncompleteIterationError) as exc:
                LOGGER.error(
                    "Gemini API specific error (attempt %s/%s): %s",
                    attempt_number,
                    total_attempts,
                    exc,
                )
                raise RetryableOperationError(
                    "Gemini API returned a broken response.",
                    retryable=True,
                ) from exc
            except GoogleAPIError as exc:
                status_code = getattr(exc, "code", None)
                if status_code is None and hasattr(exc, "response"):
                    status_code = getattr(getattr(exc, "response"), "status_code", None)
                should_retry = True
                if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                    should_retry = False
                LOGGER.error(
                    "Gemini API returned an error response (attempt %s/%s): %s",
                    attempt_number,
                    total_attempts,
                    exc,
                )
                raise RetryableOperationError(
                    "Gemini API returned an error response.",
                    error_code=status_code,
                    retryable=should_retry,
                ) from exc
            except Exception as exc:
                LOGGER.error(
                    "Error while generating content with the Gemini API (attempt %s/%s): %s",
                    attempt_number,
                    total_attempts,
                    exc,
                    exc_info=True,
                )
                raise RetryableOperationError(
                    "Unexpected error while calling Gemini API.",
                    retryable=True,
                ) from exc

            LOGGER.debug(
                "Gemini raw response for model '%s' (attempt %s/%s): %s",
                model_for_log,
                attempt_number,
                total_attempts,
                self._format_response_for_logging(response),
            )

            if hasattr(response, "text") and response.text:
                generated_text = response.text.strip()
                LOGGER.info("Gemini API returned a successful response.")
                return generated_text

            LOGGER.warning(
                "Gemini API returned an empty response (attempt %s/%s)",
                attempt_number,
                total_attempts,
            )
            raise RetryableOperationError(
                "Gemini API returned an empty response.",
                retryable=True,
            )

        try:
            return retry_with_backoff(
                _generate,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                jitter_factor=jitter_factor,
                operation_id=operation_id,
                logger=LOGGER,
                retryable_error_codes=retryable_error_codes,
            )
        except RetryableOperationError:
            LOGGER.error(
                "All attempts to generate content with the Gemini API failed."
            )
        except Exception:
            LOGGER.error(
                "Gemini API request failed with an unexpected error after retries.",
                exc_info=True,
            )
        return ""

    def get_correction(self, text: str) -> str:
        """Formata e executa uma requisição de correção de texto.

        Returns:
            Texto corrigido quando a chamada é bem-sucedida; caso contrário,
            o texto original recebido.
        """
        if not text or not self.is_valid or not self.correction_model:
            return text

        correction_prompt_template = self.config_manager.get(
            GEMINI_PROMPT_CONFIG_KEY
        )
        if not correction_prompt_template:
            LOGGER.warning("Gemini correction prompt template is empty.")
            return text

        full_prompt = correction_prompt_template.format(text=text)
        corrected_text = self._execute_request(
            full_prompt,
            self.correction_model,
            self.correction_model_id,
        )
        return corrected_text if corrected_text else text

    def get_agent_response(self, text: str) -> str:
        """Formata e executa uma requisição do modo agente.

        Returns:
            Resposta gerada pelo agente ou o texto original quando a chamada
            não pôde ser atendida.
        """
        if not text or not self.agent_model:
            return text
        agent_prompt_template = self.config_manager.get('prompt_agentico')
        if not agent_prompt_template:
            LOGGER.warning("Gemini agent prompt template is empty.")
            return text

        full_prompt = f"{agent_prompt_template}\n\n{text}"
        agent_response = self._execute_request(
            full_prompt,
            self.agent_model,
            self.agent_model_id,
        )
        return agent_response if agent_response else text

    def correct_text_async(self, text: str, prompt: str, api_key: str) -> str:
        """Executa correção de texto em modo assíncrono utilizando um prompt customizado.

        Returns:
            Texto corrigido quando a chamada é concluída com sucesso ou
            ``text`` caso ocorra falha.
        """
        if not self.is_valid or not self.correction_model:
            return text

        if not prompt:
            LOGGER.warning("Empty Gemini prompt received for async correction.")
            return text

        if api_key and self.current_api_key and api_key != self.current_api_key:
            LOGGER.debug(
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
