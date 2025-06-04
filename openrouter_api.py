import requests
import json
import logging
import time

class OpenRouterAPI:
    """
    Class to handle interactions with the OpenRouter API for text correction using Deepseek model.
    """
    def __init__(self, api_key, model_id="deepseek/deepseek-chat-v3-0324:free"):
        """
        Initialize the OpenRouter API client.

        Args:
            api_key (str): Your OpenRouter API key
            model_id (str): The model ID to use (default: "deepseek/deepseek-chat-v3-0324:free")
        """
        self.api_key = api_key
        self.model_id = model_id
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://whisper-recorder.app",  # Replace with your actual app URL
            "X-Title": "Whisper Recorder"  # Replace with your actual app name
        }

    def correct_text(self, text: str, max_retries: int = 3, retry_delay: float = 1) -> str:
        """
        Send text to the Deepseek model for correction of punctuation, capitalization, and names.

        Args:
            text (str): The text to correct
            max_retries (int): Maximum number of retries on failure
            retry_delay (float): Delay between retries in seconds

        Returns:
            str: The corrected text, or the original text if the API call fails
        """
        if not text or not text.strip():
            logging.warning("Empty text provided to OpenRouter API, skipping correction")
            return text

        # Create the prompt for the model
        system_message = """You are a text correction assistant. Your task is to correct the following transcribed text:
1. Fix punctuation (commas, periods, question marks, etc.)
2. Maintain the original meaning and all content
3. Do not add, edit, or remove information/words.
4. Return only the corrected text without any explanations or additional comments"""

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Please correct this transcribed text: {text}"}
            ],
            "temperature": 0.0,  # Low temperature for more deterministic outputs
            "max_tokens": 10000   # Adjust based on your expected output length
        }

        # Try to make the API call with retries
        for attempt in range(max_retries):
            try:
                logging.info(f"Sending text to OpenRouter API (attempt {attempt+1}/{max_retries})")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30  # 30 second timeout
                )

                response.raise_for_status()  # Raise exception for 4XX/5XX responses

                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    corrected_text = result['choices'][0]['message']['content']
                    logging.info("Successfully received corrected text from OpenRouter API")
                    if corrected_text != text:
                        logging.info("OpenRouter API made corrections to the text")
                    else:
                        logging.info("OpenRouter API returned text unchanged")
                    return corrected_text
                else:
                    logging.error(f"Unexpected response format from OpenRouter API: {result}")
                    logging.debug(f"Full response: {json.dumps(result, indent=2)}")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error calling OpenRouter API: {e}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Increase retry delay for subsequent attempts
                    retry_delay *= 2

            except Exception as e:
                logging.error(f"Unexpected error with OpenRouter API: {e}")
                break

        # If we get here, all attempts failed
        logging.warning("Failed to correct text with OpenRouter API, returning original text")
        return text
