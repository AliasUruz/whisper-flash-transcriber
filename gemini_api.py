import logging
import time
from typing import Optional

import google.generativeai as genai


class GeminiAPI:
    """
    A client for the Google Gemini API to correct transcribed text.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.0-flash-001",
        prompt: str = "",
    ) -> None:
        """
        Initialize the Gemini API client.

        Args:
            api_key: Chave da API do Gemini.
            model_id: ID do modelo (padrão: gemini-2.0-flash-001).
            prompt: Prompt usado na correção (padrão: "").
        """
        self.api_key = api_key
        self.model_id = model_id
        self.model = None
        self.prompt = prompt

        try:
            # Configure the Gemini API with the provided API key
            genai.configure(api_key=self.api_key)

            # Initialize the model
            self.model = genai.GenerativeModel(self.model_id)
            logging.info(
                "Gemini API client initialized with model: %s",
                self.model_id,
            )
        except Exception as e:
            logging.error(f"Failed to initialize Gemini API client: {e}")
            self.model = None

    def correct_text(
        self,
        text: str,
        max_retries: int = 3,
        retry_delay: int = 1,
        override_prompt: Optional[str] = None,
    ) -> str:
        """
        Correct the transcribed text using the Gemini API.

        Args:
            text: The text to correct
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            override_prompt: Prompt opcional para substituir o padrão
                nesta correção.

        Returns:
            The corrected text or the original text if correction fails
        """
        if not text or not self.model:
            return text

        # Cria o prompt formatando self.prompt com o texto a ser corrigido
        if override_prompt is not None:
            prompt_to_format = override_prompt
        else:
            prompt_to_format = self.prompt

        prompt = prompt_to_format.format(text=text)

        for attempt in range(max_retries):
            try:
                logging.info(
                    "Sending text to Gemini API for correction "
                    "(attempt %s/%s)",
                    attempt + 1,
                    max_retries,
                )

                # Generate content with the model
                response = self.model.generate_content(prompt)

                # Check if the response is valid
                if hasattr(response, 'text') and response.text:
                    corrected_text = response.text.strip()
                    logging.info(
                        "Successfully received corrected text from Gemini API"
                    )

                    if corrected_text != text:
                        logging.info("Gemini API made corrections to the text")
                    else:
                        logging.info("Gemini API returned text unchanged")

                    return corrected_text
                else:
                    logging.warning(
                        "Gemini API returned empty response (attempt %s/%s)",
                        attempt + 1,
                        max_retries,
                    )

            except Exception as e:
                logging.error(
                    "Error during Gemini API text correction "
                    "(attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

                # Wait before retrying
                if attempt < max_retries - 1:
                    logging.info("Retrying in %s seconds...", retry_delay)
                    time.sleep(retry_delay)

        # If all retries failed, return the original text
        logging.error(
            "All Gemini API correction attempts failed, "
            "returning original text",
        )
        return text
