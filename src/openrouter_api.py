from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

from .logging_utils import get_logger, log_context
from .app_identity import APP_DISPLAY_NAME, APP_OFFICIAL_URL, APP_LOG_NAMESPACE

LOGGER = get_logger(f"{APP_LOG_NAMESPACE}.openrouter", component='OpenRouterAPI')


class OpenRouterAPI:
    """Cliente para a API OpenRouter usado na correção de texto.

    Possui o método :py:meth:`reinitialize_client` para atualizar chave
    e modelo dinamicamente, permitindo aplicar novas configurações sem
    recriar a instância.
    """

    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        api_key: str,
        model_id: str = "deepseek/deepseek-chat-v3-0324:free",
        max_tokens: int = 4096,
        request_timeout: float = DEFAULT_TIMEOUT,
        *,
        max_attempts: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 16.0,
        retry_jitter_factor: float = 0.25,
    ) -> None:
        """Inicializa o cliente OpenRouter.

        Args:
            api_key: Chave de acesso para a API.
            model_id: Identificador do modelo em uso.
            max_tokens: Limite máximo de tokens retornados pela API. O padrão
                ``4096`` é compatível com a maioria dos modelos do OpenRouter.
            request_timeout: Tempo máximo (em segundos) aguardado por resposta
                da API em cada tentativa.
        """
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_tokens = max_tokens
        self._last_invalid_timeout_value: object | None = None
        self.request_timeout = self._normalize_timeout(
            request_timeout,
            self.DEFAULT_TIMEOUT,
        )
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))
        self.retry_max_delay = max(float(retry_max_delay), self.retry_base_delay or 0.0)
        self.retry_jitter_factor = max(0.0, float(retry_jitter_factor))
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": APP_OFFICIAL_URL,
            "X-Title": APP_DISPLAY_NAME,
        }

    def _normalize_timeout(
        self,
        value: float | int | str | None,
        fallback: float,
    ) -> float:
        """Converte valores arbitrários de timeout em ``float`` positivos.

        Returns:
            Valor de timeout válido em segundos. Quando ``value`` é inválido,
            retorna ``fallback`` e registra um aviso.
        """
        if value is None:
            return float(fallback)
        try:
            timeout_value = float(value)
            if timeout_value <= 0:
                raise ValueError
        except (TypeError, ValueError):
            if value != self._last_invalid_timeout_value:
                LOGGER.warning(
                    "Invalid OpenRouter timeout '%s'; using fallback %.2f seconds.",
                    value,
                    float(fallback),
                )
                self._last_invalid_timeout_value = value
            return float(fallback)
        self._last_invalid_timeout_value = None
        return timeout_value

    def reinitialize_client(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        request_timeout: Optional[float] = None,
        max_attempts: Optional[int] = None,
        retry_base_delay: Optional[float] = None,
        retry_max_delay: Optional[float] = None,
        retry_jitter_factor: Optional[float] = None,
    ) -> None:
        """Atualiza chave, modelo e cabeçalhos do cliente.

        Args:
            api_key: Nova chave de API. Se ``None``, mantém a atual.
            model_id: Novo identificador de modelo.
                Se ``None``, mantém o atual.
            request_timeout: Novo timeout (em segundos) para requisições. Se
                ``None``, mantém o atual.
        """
        if api_key:
            self.api_key = api_key
        if model_id:
            self.model_id = model_id
        if request_timeout is not None:
            self.request_timeout = self._normalize_timeout(
                request_timeout,
                self.request_timeout,
            )
        if max_attempts is not None:
            try:
                attempts = int(max_attempts)
            except (TypeError, ValueError):
                attempts = self.max_attempts
            else:
                if attempts > 0:
                    self.max_attempts = attempts
        if retry_base_delay is not None:
            try:
                self.retry_base_delay = max(0.0, float(retry_base_delay))
            except (TypeError, ValueError):
                pass
        if retry_max_delay is not None:
            try:
                self.retry_max_delay = max(
                    float(retry_max_delay),
                    self.retry_base_delay or 0.0,
                )
            except (TypeError, ValueError):
                pass
        if retry_jitter_factor is not None:
            try:
                self.retry_jitter_factor = max(0.0, float(retry_jitter_factor))
            except (TypeError, ValueError):
                pass
        self.headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers["HTTP-Referer"] = APP_OFFICIAL_URL
        self.headers["X-Title"] = APP_DISPLAY_NAME
        LOGGER.info(
            "OpenRouter API client re/initialized with model '%s'",
            self.model_id,
        )
        LOGGER.debug('OpenRouter request timeout configured for %.2f seconds.', self.request_timeout)

    def _perform_single_attempt(
        self,
        payload_json: str,
        attempt_number: int,
        max_attempts: int,
    ) -> dict:
        """Executa uma tentativa única de requisição ao OpenRouter."""
        LOGGER.info(
            "Sending request to OpenRouter API with model '%s' (attempt %s/%s)",
            self.model_id,
            attempt_number,
            max_attempts,
        )
        LOGGER.debug(
            "OpenRouter payload for model '%s' (attempt %s/%s): %s",
            self.model_id,
            attempt_number,
            max_attempts,
            payload_json,
        )
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=payload_json,
                timeout=self.request_timeout,
            )
        except requests.exceptions.Timeout as exc:
            LOGGER.error(
                "OpenRouter API request timed out after %.2f seconds (attempt %s/%s): %s",
                self.request_timeout,
                attempt_number,
                max_attempts,
                exc,
            )
            raise RetryableOperationError(
                "OpenRouter API request timed out.",
                error_code="timeout",
                retryable=True,
            ) from exc
        except requests.exceptions.RequestException as exc:
            LOGGER.error(
                "Network error when calling OpenRouter API (attempt %s/%s): %s",
                attempt_number,
                max_attempts,
                exc,
            )
            raise RetryableOperationError(
                "OpenRouter API network error.",
                error_code="network",
                retryable=True,
            ) from exc

        LOGGER.debug(
            "OpenRouter raw response (status %s, attempt %s/%s): %s",
            response.status_code,
            attempt_number,
            max_attempts,
            response.text,
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status_code = (
                exc.response.status_code if exc.response is not None else "unknown"
            )
            LOGGER.error(
                "OpenRouter API HTTP error (status %s) on attempt %s/%s: %s",
                status_code,
                attempt_number,
                max_attempts,
                exc,
            )
            if exc.response is not None:
                LOGGER.debug(
                    "OpenRouter HTTP error response body (attempt %s/%s): %s",
                    attempt_number,
                    max_attempts,
                    exc.response.text,
                )
            retryable = True
            if isinstance(status_code, int) and 400 <= status_code < 500 and status_code != 429:
                retryable = False
            raise RetryableOperationError(
                "OpenRouter API returned an HTTP error.",
                error_code=status_code,
                retryable=retryable,
            ) from exc

        try:
            result = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            LOGGER.error(
                "Failed to decode OpenRouter API response JSON (attempt %s/%s): %s",
                attempt_number,
                max_attempts,
                exc,
            )
            raise RetryableOperationError(
                "Failed to decode OpenRouter API response JSON.",
                retryable=True,
            ) from exc
        except Exception as exc:
            LOGGER.error(
                "Unexpected error while parsing OpenRouter API response (attempt %s/%s): %s",
                attempt_number,
                max_attempts,
                exc,
                exc_info=True,
            )
            raise RetryableOperationError(
                "Unexpected error while parsing OpenRouter API response.",
                retryable=False,
            ) from exc

        return result

    def correct_text(
        self,
        text: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> str:
        """
        Corrige o texto transcrito usando o modelo Deepseek.

        Args:
            text: Texto a ser corrigido.
            max_retries: Número máximo de tentativas.
            retry_delay: Atraso entre tentativas em segundos.

        Returns:
            str: Texto corrigido ou original caso falhe.
        """
        if not text or not text.strip():
            LOGGER.warning(
                "Empty text provided to OpenRouter API; skipping correction"
            )
            return text

        # Create the prompt for the model
        system_message = (
            "You are a text correction assistant. "
            "Your task is to correct the following transcribed text:\n"
            "1. Fix punctuation (commas, periods, question marks, etc.)\n"
            "2. Maintain the original meaning and all content\n"
            "3. Do not add, edit, or remove information/words.\n"
            "4. Return only the corrected text without any explanations or "
            "additional comments"
        )

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Please correct this transcribed text: {text}",
                }
            ],
            "temperature": 0.0,
            # Low temperature para maior determinismo
            "max_tokens": self.max_tokens
        }

        payload_json = json.dumps(payload, ensure_ascii=False)
        start_time = time.time()
        attempts = max_retries if max_retries is not None else self.max_attempts
        base_delay = retry_delay if retry_delay is not None else self.retry_base_delay
        operation_id = f"openrouter:{self.model_id}:{time.monotonic_ns()}"
        retryable_error_codes = (408, 409, 425, 429, 500, 502, 503, 504)

        def _invoke(attempt_number: int, total_attempts: int) -> str:
            result = self._perform_single_attempt(
                payload_json,
                attempt_number,
                total_attempts,
            )
            if result is not None and result.get("choices"):
                return result["choices"][0]["message"]["content"]

            LOGGER.error(
                "Unexpected response format from OpenRouter API: %s",
                result,
            )
            LOGGER.debug(
                "OpenRouter unexpected response payload (attempt %s/%s): %s",
                attempt_number,
                total_attempts,
                json.dumps(result, ensure_ascii=False),
            )
            raise RetryableOperationError(
                "OpenRouter API returned an unexpected payload.",
                retryable=True,
            )

        try:
            corrected_text = retry_with_backoff(
                _invoke,
                max_attempts=attempts,
                base_delay=base_delay,
                max_delay=self.retry_max_delay,
                jitter_factor=self.retry_jitter_factor,
                operation_id=operation_id,
                logger=LOGGER,
                retryable_error_codes=retryable_error_codes,
            )
        except RetryableOperationError:
            LOGGER.warning(
                "Failed to correct text with OpenRouter API, returning original text (operation_id=%s)",
                operation_id,
            )
            return text
        except Exception:
            LOGGER.error(
                "OpenRouter correction failed with unexpected error after retries (operation_id=%s)",
                operation_id,
                exc_info=True,
            )
            return text

        LOGGER.info(
            log_context(
                "Successfully received corrected text from OpenRouter API.",
                event="openrouter.correction_success",
                latency_ms=int((time.time() - start_time) * 1000),
                operation_id=operation_id,
            )
        )
        if corrected_text != text:
            LOGGER.info("OpenRouter API made corrections to the text")
        else:
            LOGGER.info("OpenRouter API returned text unchanged")
        return corrected_text

    def correct_text_async(
        self, text: str, prompt: str, api_key: str, model: str
    ) -> str:
        """Corrige texto usando prompt customizado de forma assíncrona.

        Esta função reusa a lógica de :py:meth:`correct_text`, mas permite
        especificar um *prompt* diferente, do qual são derivados tanto a
        mensagem de sistema quanto a do usuário.

        Returns:
            Texto revisado quando a chamada é bem-sucedida; caso contrário,
            devolve ``text`` sem alterações.
        """
        self.reinitialize_client(api_key=api_key, model_id=model)

        if not text or not text.strip():
            LOGGER.warning(
                "Empty text provided to OpenRouter API; skipping correction"
            )
            return text

        system_message = prompt.replace("{text}", "").strip()
        user_message = text

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.0,
            "max_tokens": 10000,
        }

        payload_json = json.dumps(payload, ensure_ascii=False)
        attempts = self.max_attempts
        operation_id = f"openrouter:{self.model_id}:async:{time.monotonic_ns()}"
        retryable_error_codes = (408, 409, 425, 429, 500, 502, 503, 504)

        def _invoke_async(attempt_number: int, total_attempts: int) -> str:
            result = self._perform_single_attempt(
                payload_json,
                attempt_number,
                total_attempts,
            )
            if result is not None and result.get("choices"):
                return result["choices"][0]["message"]["content"]

            LOGGER.error(
                "Unexpected response format from OpenRouter API: %s",
                result,
            )
            LOGGER.debug(
                "OpenRouter unexpected response payload (attempt %s/%s): %s",
                attempt_number,
                total_attempts,
                json.dumps(result, ensure_ascii=False),
            )
            raise RetryableOperationError(
                "OpenRouter API returned an unexpected payload.",
                retryable=True,
            )

        try:
            return retry_with_backoff(
                _invoke_async,
                max_attempts=attempts,
                base_delay=self.retry_base_delay,
                max_delay=self.retry_max_delay,
                jitter_factor=self.retry_jitter_factor,
                operation_id=operation_id,
                logger=LOGGER,
                retryable_error_codes=retryable_error_codes,
            )
        except RetryableOperationError:
            LOGGER.warning(
                "Failed to correct text with OpenRouter API, returning original text (operation_id=%s)",
                operation_id,
            )
        except Exception:
            LOGGER.error(
                "OpenRouter async correction failed with unexpected error after retries (operation_id=%s)",
                operation_id,
                exc_info=True,
            )
        return text
