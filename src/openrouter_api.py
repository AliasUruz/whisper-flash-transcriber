import requests
import json
import logging
import time
from typing import Optional


class OpenRouterAPI:
    """Cliente para a API OpenRouter usado na correção de texto.

    Possui o método :py:meth:`reinitialize_client` para atualizar chave
    e modelo dinamicamente, permitindo aplicar novas configurações sem
    recriar a instância.
    """
    def __init__(
        self,
        api_key: str,
        model_id: str = "deepseek/deepseek-chat-v3-0324:free",
    ) -> None:
        """Inicializa o cliente OpenRouter."""
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://whisper-recorder.app",
            "X-Title": "Whisper Recorder",
        }

    def reinitialize_client(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        """Atualiza chave, modelo e cabeçalhos do cliente.

        Args:
            api_key: Nova chave de API. Se ``None``, mantém a atual.
            model_id: Novo identificador de modelo.
                Se ``None``, mantém o atual.
        """
        if api_key:
            self.api_key = api_key
        if model_id:
            self.model_id = model_id
        self.headers["Authorization"] = f"Bearer {self.api_key}"
        logging.info(
            "OpenRouter API client re/initialized with model '%s'",
            self.model_id,
        )

    def correct_text(
        self,
        text: str,
        max_retries: int = 3,
        retry_delay: float = 1,
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
            logging.warning(
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
            # Low temperature for mais determinismo
            "max_tokens": 10000   # Adjust based on your expected output length
        }

        # Try to make the API call with retries
        for attempt in range(max_retries):
            try:
                logging.info(
                    "Sending text to OpenRouter API (attempt %s/%s)",
                    attempt + 1,
                    max_retries,
                )
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30  # 30 second timeout
                )

                response.raise_for_status()  # Exceção para erros 4XX/5XX

                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    corrected_text = result['choices'][0]['message']['content']
                    logging.info(
                        "Successfully received corrected text from "
                        "OpenRouter API"
                    )
                    if corrected_text != text:
                        logging.info(
                            "OpenRouter API made corrections to the text"
                        )
                    else:
                        logging.info("OpenRouter API returned text unchanged")
                    return corrected_text
                else:
                    logging.error(
                        "Unexpected response format from OpenRouter API: %s",
                        result,
                    )
                    logging.debug(
                        "Full response: %s",
                        json.dumps(result, indent=2),
                    )

            except requests.exceptions.RequestException as e:
                logging.error("Error calling OpenRouter API: %s", e)
                if attempt < max_retries - 1:
                    logging.info("Retrying in %s seconds...", retry_delay)
                    time.sleep(retry_delay)
                    # Increase retry delay for subsequent attempts
                    retry_delay *= 2

            except Exception as e:
                logging.error("Unexpected error with OpenRouter API: %s", e)
                break

        # If we get here, all attempts failed
        logging.warning(
            "Failed to correct text with OpenRouter API, "
            "returning original text",
        )
        return text
