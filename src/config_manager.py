import json
import os
import logging

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "record_key": "F3",
    "record_mode": "toggle",
    "auto_paste": True,
    "min_record_duration": 0.5,
    "sound_enabled": True,
    "sound_frequency": 400,
    "sound_duration": 0.3,
    "sound_volume": 0.5,
    "agent_key": "F5",
    "agent_record_duration": 5.0,
    "keyboard_library": "win32",
    "text_correction_enabled": False,
    "text_correction_service": "none",
    "openrouter_api_key": "",
    "openrouter_model": "deepseek/deepseek-chat-v3-0324:free",
    "gemini_api_key": "",
    "gemini_model": "gemini-2.0-flash-001",
    "gemini_mode": "correction",
    "gemini_general_prompt": "Based on the following text, generate a short response: {text}",
    "gemini_agent_prompt": "You are a helpful assistant. Reply to: {text}",
    "gemini_agent_model": "gemini-2.0-flash-001",
    "agent_auto_paste": True,
    "gemini_prompt": "",
    "batch_size": 16,
    "gpu_index": 0,
    "auto_reregister_hotkeys": True,
    "gemini_model_options": [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-pro"
    ],
    "save_audio_for_debug": False
}

class ConfigManager:
    def __init__(self, path: str = CONFIG_FILE):
        self.path = path
        self.config = DEFAULT_CONFIG.copy()

    def load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.config.update(data)
                    logging.info(f"Configuração carregada de {self.path}")
            except Exception as e:
                logging.error(f"Erro ao carregar {self.path}: {e}")
        else:
            logging.info(f"{self.path} não encontrado. Utilizando valores padrão.")
        return self.config

    def save(self, data: dict):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            logging.info(f"Configuração salva em {self.path}")
        except Exception as e:
            logging.error(f"Erro ao salvar configuração: {e}")
