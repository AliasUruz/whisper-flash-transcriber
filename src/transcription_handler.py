import logging
import torch
from transformers import pipeline
import numpy as np
from typing import Optional

from openrouter_api import OpenRouterAPI
from gemini_api import GeminiAPI

class TranscriptionHandler:
    def __init__(self, config: dict):
        self.config = config
        self.pipe = None
        self.openrouter_client: Optional[OpenRouterAPI] = None
        self.gemini_client: Optional[GeminiAPI] = None
        self._init_clients()
        self._load_model()

    def _init_clients(self):
        if self.config.get("openrouter_api_key"):
            self.openrouter_client = OpenRouterAPI(
                api_key=self.config.get("openrouter_api_key"),
                model_id=self.config.get("openrouter_model")
            )
        if self.config.get("gemini_api_key"):
            self.gemini_client = GeminiAPI(
                api_key=self.config.get("gemini_api_key"),
                model_id=self.config.get("gemini_model"),
                prompt=self.config.get("gemini_prompt", "")
            )

    def _load_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        device_param = 0 if device == "cuda" else "cpu"
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                device=device_param,
                torch_dtype=dtype,
            )
            logging.info("Pipeline Whisper carregado")
        except Exception as e:
            logging.error(f"Erro ao carregar pipeline: {e}")
            self.pipe = None

    def transcribe(self, audio: np.ndarray) -> str:
        if self.pipe is None:
            logging.error("Pipeline não disponível")
            return ""
        try:
            result = self.pipe(audio, chunk_length_s=30, batch_size=self.config.get("batch_size", 16), return_timestamps=False)
            text = result.get("text", "").strip()
            return self._correct(text)
        except Exception as e:
            logging.error(f"Erro na transcrição: {e}")
            return ""

    def _correct(self, text: str) -> str:
        if not text:
            return text
        if not self.config.get("text_correction_enabled"):
            return text
        service = self.config.get("text_correction_service")
        if service == "openrouter" and self.openrouter_client:
            return self.openrouter_client.correct_text(text)
        if service == "gemini" and self.gemini_client:
            prompt = self.config.get("gemini_prompt", "")
            return self.gemini_client.correct_text(text, override_prompt=prompt)
        return text
