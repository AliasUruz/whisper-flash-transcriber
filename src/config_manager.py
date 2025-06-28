import os
import json
import logging
import copy
import hashlib
try:
    from distutils.util import strtobool
except Exception:  # Python >= 3.12
    from setuptools._distutils.util import strtobool


def _parse_bool(value, default=False):
    """Converte diferentes representações de booleanos em objetos ``bool``.

    Se o valor for uma string não reconhecida, retorna ``default``.
    """
    if isinstance(value, str):
        try:
            return bool(strtobool(value))
        except ValueError:
            return None
    return bool(value)

# --- Constantes de Configuração (movidas de whisper_tkinter.py) ---
CONFIG_FILE = "config.json"
SECRETS_FILE = "secrets.json" # Nova constante para o arquivo de segredos
DEFAULT_CONFIG = {
    "record_key": "F3",
    "record_mode": "toggle",
    "auto_paste": True,
    "min_record_duration": 0.5,
    "sound_enabled": True,
    "sound_frequency": 400,
    "sound_duration": 0.3,
    "sound_volume": 0.5,
    "agent_key": "F4",
    "keyboard_library": "win32",
    "text_correction_enabled": False,
    "text_correction_service": "none",
    "openrouter_api_key": "",
    "openrouter_model": "deepseek/deepseek-chat-v3-0324:free",
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash-lite-preview-06-17",
    "gemini_agent_model": "gemini-2.5-flash-lite-preview-06-17",
    # Ativado por padrão; desabilite se necessário
    "use_flash_attention_2": True,
    "ai_provider": "gemini",
    "openrouter_agent_prompt": "",
    "openrouter_prompt": "",
    "prompt_agentico": "You are an AI assistant that executes text commands. The user will provide an instruction followed by the text to be processed. Your task is to execute the instruction on the text and return ONLY the final result. Do not add explanations, greetings, or any extra text. The user's instruction is your top priority. The output language should match the main language of the provided text.",
    "gemini_prompt": """You are a meticulous speech-to-text correction AI. Your primary task is to correct punctuation, capitalization, and minor transcription errors in the text below while preserving the original content and structure as closely as possible.
Key instructions:
- Correct punctuation, such as adding commas, periods, and question marks where appropriate.
- Fix capitalization at the beginning of sentences.
- Remove only obvious speech disfluencies like stutters (e.g., \"I-I mean\") and false starts, but preserve the natural flow of speech.
- DO NOT summarize, paraphrase, or change the original meaning of the sentences.
- DO NOT remove any content, even if it seems redundant.
- Preserve all language transitions (e.g., Portuguese/English) exactly as they occur.
- Return only the corrected text, with no additional comments or explanations.
Transcribed speech: {text}""",
    "batch_size": 16, # Valor padrão para o modo automático
    "batch_size_mode": "auto", # Novo: 'auto' ou 'manual'
    "manual_batch_size": 8, # Novo: Valor para o modo manual
    "gpu_index": 0,
    "use_turbo": False,
    "hotkey_stability_service_enabled": True, # Nova configuração unificada
    "use_vad": False,
    "vad_threshold": 0.5,
    # Duração máxima da pausa preservada antes que o silêncio seja descartado
    "vad_silence_duration": 1.0,
    "display_transcripts_in_terminal": False,
    "gemini_model_options": [
        "gemini-2.5-flash-lite-preview-06-17",
        "gemini-2.5-flash",
        "gemini-2.5-pro"
    ],
    "text_correction_timeout": 30,
    "save_temp_recordings": False,
    "min_transcription_duration": 1.0, # Nova configuração
}

# Outras constantes de configuração (movidas de whisper_tkinter.py)
MIN_RECORDING_DURATION_CONFIG_KEY = "min_record_duration"
MIN_TRANSCRIPTION_DURATION_CONFIG_KEY = "min_transcription_duration"
AGENT_KEY_CONFIG_KEY = "agent_key"
SOUND_ENABLED_CONFIG_KEY = "sound_enabled"
SOUND_FREQUENCY_CONFIG_KEY = "sound_frequency"
SOUND_DURATION_CONFIG_KEY = "sound_duration"
SOUND_VOLUME_CONFIG_KEY = "sound_volume"
HOTKEY_STABILITY_SERVICE_ENABLED_CONFIG_KEY = "hotkey_stability_service_enabled" # Nova constante unificada
BATCH_SIZE_CONFIG_KEY = "batch_size" # Agora é o batch size padrão para o modo auto
BATCH_SIZE_MODE_CONFIG_KEY = "batch_size_mode" # Novo
MANUAL_BATCH_SIZE_CONFIG_KEY = "manual_batch_size" # Novo
GPU_INDEX_CONFIG_KEY = "gpu_index"
USE_TURBO_CONFIG_KEY = "use_turbo"
SAVE_TEMP_RECORDINGS_CONFIG_KEY = "save_temp_recordings"
USE_FLASH_ATTENTION_2_CONFIG_KEY = "use_flash_attention_2"
DISPLAY_TRANSCRIPTS_KEY = "display_transcripts_in_terminal"
USE_VAD_CONFIG_KEY = "use_vad"
VAD_THRESHOLD_CONFIG_KEY = "vad_threshold"
VAD_SILENCE_DURATION_CONFIG_KEY = "vad_silence_duration"
DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY = DISPLAY_TRANSCRIPTS_KEY
KEYBOARD_LIBRARY_CONFIG_KEY = "keyboard_library"
KEYBOARD_LIB_WIN32 = "win32"
TEXT_CORRECTION_ENABLED_CONFIG_KEY = "text_correction_enabled"
TEXT_CORRECTION_SERVICE_CONFIG_KEY = "text_correction_service"
ENABLE_AI_CORRECTION_CONFIG_KEY = TEXT_CORRECTION_ENABLED_CONFIG_KEY
SERVICE_NONE = "none"
SERVICE_OPENROUTER = "openrouter"
SERVICE_GEMINI = "gemini"
OPENROUTER_API_KEY_CONFIG_KEY = "openrouter_api_key"
OPENROUTER_MODEL_CONFIG_KEY = "openrouter_model"
GEMINI_API_KEY_CONFIG_KEY = "gemini_api_key"
GEMINI_MODEL_CONFIG_KEY = "gemini_model"
GEMINI_AGENT_MODEL_CONFIG_KEY = "gemini_agent_model"
WHISPER_MODEL_ID_CONFIG_KEY = "whisper_model_id"
GEMINI_MODEL_OPTIONS_CONFIG_KEY = "gemini_model_options"
AI_PROVIDER_CONFIG_KEY = TEXT_CORRECTION_SERVICE_CONFIG_KEY
GEMINI_AGENT_PROMPT_CONFIG_KEY = "prompt_agentico"
OPENROUTER_PROMPT_CONFIG_KEY = "openrouter_agent_prompt"
OPENROUTER_AGENT_PROMPT_CONFIG_KEY = OPENROUTER_PROMPT_CONFIG_KEY
GEMINI_PROMPT_CONFIG_KEY = "gemini_prompt"
TEXT_CORRECTION_TIMEOUT_CONFIG_KEY = "text_correction_timeout"
SETTINGS_WINDOW_GEOMETRY = "550x700"
REREGISTER_INTERVAL_SECONDS = 60
MAX_HOTKEY_FAILURES = 3
HOTKEY_HEALTH_CHECK_INTERVAL = 10

class ConfigManager:
    def __init__(self, config_file=CONFIG_FILE, default_config=DEFAULT_CONFIG):
        self.config_file = config_file
        self.default_config = default_config
        self.config = {}
        self._config_hash = None
        self._secrets_hash = None
        self.load_config()

    def _compute_hash(self, data) -> str:
        """Gera um hash SHA256 determinístico para o dicionário informado."""
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    def load_config(self):
        cfg = copy.deepcopy(self.default_config) # Usar deepcopy para evitar modificações no default

        # 1. Carregar config.json (configurações do usuário)
        loaded_config_from_file = {}
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding='utf-8') as f:
                    loaded_config_from_file = json.load(f)
                self._config_hash = self._compute_hash(loaded_config_from_file)

                if "vad_enabled" in loaded_config_from_file:
                    logging.info("Migrating legacy 'vad_enabled' key to 'use_vad'.")
                    loaded_config_from_file["use_vad"] = loaded_config_from_file.pop("vad_enabled")
                cfg.update(loaded_config_from_file)
                logging.info(f"Configuration loaded from {self.config_file}.")

                # --- Migração do Prompt Agêntico ---
                old_agent_prompt = "Você é um assistente de IA que integra um sistema operacional. Se o usuário pedir uma ação que possa ser resolvida por um comando de terminal (como listar arquivos, verificar o IP, etc.), responda exclusivamente com o comando dentro das tags <cmd>comando</cmd>. Para todas as outras solicitações, responda normalmente."
                current_agent_prompt = cfg.get("prompt_agentico", "")
                if current_agent_prompt == old_agent_prompt:
                    cfg["prompt_agentico"] = self.default_config["prompt_agentico"]
                    logging.info("Prompt agêntico antigo detectado e migrado para o novo padrão.")
                # --- Fim da Migração ---
            else:
                logging.info(f"{self.config_file} not found. Using defaults.")
                self._config_hash = None
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error reading or decoding {self.config_file}: {e}. Using default configuration.")
            # Em caso de erro, garantir que o config.json seja recriado com defaults
            loaded_config_from_file = {}
            cfg = copy.deepcopy(self.default_config) # Resetar para defaults limpos
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {self.config_file}: {e}. Using default configuration.", exc_info=True)
            loaded_config_from_file = {}
            cfg = copy.deepcopy(self.default_config)

        # 2. Carregar secrets.json (chaves de API e segredos)
        secrets_loaded = {}
        try:
            if os.path.exists(SECRETS_FILE):
                with open(SECRETS_FILE, "r", encoding='utf-8') as f:
                    secrets_loaded = json.load(f)
                cfg.update(secrets_loaded)  # Secrets sobrescrevem configs se houver conflito
                self._secrets_hash = self._compute_hash(secrets_loaded)
                logging.info(f"Secrets loaded from {SECRETS_FILE}.")
            else:
                logging.info(f"{SECRETS_FILE} not found. API keys might be missing.")
                self._secrets_hash = None
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error reading or decoding {SECRETS_FILE}: {e}. API keys might be missing or invalid.")
            secrets_loaded = {}  # Resetar segredos em caso de erro
            self._secrets_hash = None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {SECRETS_FILE}: {e}. API keys might be missing or invalid.", exc_info=True)
            secrets_loaded = {}
            self._secrets_hash = None

        self.config = cfg
        # Aplicar validação e conversão de tipo
        self._validate_and_apply_config(loaded_config_from_file) # Passar o config.json carregado para validação de 'specified'
        
        # Salvar a configuração (apenas as não sensíveis) após o carregamento e validação
        self.save_config()

    def _validate_and_apply_config(self, loaded_config):
        # Validate and convert sound settings
        try:
            self.config["sound_frequency"] = int(self.config.get("sound_frequency", self.default_config["sound_frequency"]))
        except (ValueError, TypeError):
            logging.warning(f"Invalid sound_frequency value '{self.config.get('sound_frequency')}' in config. Using default ({self.default_config['sound_frequency']}).")
            self.config["sound_frequency"] = self.default_config["sound_frequency"]

        try:
            self.config["sound_duration"] = float(self.config.get("sound_duration", self.default_config["sound_duration"]))
        except (ValueError, TypeError):
            logging.warning(f"Invalid sound_duration value '{self.config.get('sound_duration')}' in config. Using default ({self.default_config['sound_duration']}).")
            self.config["sound_duration"] = self.default_config["sound_duration"]

        try:
            self.config["sound_volume"] = float(self.config.get("sound_volume", self.default_config["sound_volume"]))
            if not (0.0 <= self.config["sound_volume"] <= 1.0):
                logging.warning(f"Invalid sound_volume value '{self.config['sound_volume']}'. Must be between 0.0 and 1.0. Using default ({self.default_config['sound_volume']}).")
                self.config["sound_volume"] = self.default_config["sound_volume"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid sound_volume value '{self.config.get('sound_volume')}' in config. Using default ({self.default_config['sound_volume']}).")
            self.config["sound_volume"] = self.default_config["sound_volume"]

        # Validate and convert batch_size settings
        try:
            self.config["batch_size"] = int(self.config.get("batch_size", self.default_config["batch_size"]))
            if self.config["batch_size"] <= 0:
                logging.warning(f"Invalid batch_size value '{self.config['batch_size']}'. Must be positive. Using default ({self.default_config['batch_size']}).")
                self.config["batch_size"] = self.default_config["batch_size"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid batch_size value '{self.config.get('batch_size')}' in config. Using default ({self.default_config['batch_size']}).")
            self.config["batch_size"] = self.default_config["batch_size"]

        try:
            self.config["manual_batch_size"] = int(self.config.get("manual_batch_size", self.default_config["manual_batch_size"]))
            if self.config["manual_batch_size"] <= 0:
                logging.warning(f"Invalid manual_batch_size value '{self.config['manual_batch_size']}'. Must be positive. Using default ({self.default_config['manual_batch_size']}).")
                self.config["manual_batch_size"] = self.default_config["manual_batch_size"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid manual_batch_size value '{self.config.get('manual_batch_size')}' in config. Using default ({self.default_config['manual_batch_size']}).")
            self.config["manual_batch_size"] = self.default_config["manual_batch_size"]

        # Validate hotkey_stability_service_enabled
        hse_val = _parse_bool(
            self.config.get(
                "hotkey_stability_service_enabled",
                self.default_config["hotkey_stability_service_enabled"],
            )
        )
        if hse_val is None:
            logging.warning(
                f"Invalid hotkey_stability_service_enabled '{self.config.get('hotkey_stability_service_enabled')}'. Using default ({self.default_config['hotkey_stability_service_enabled']})."
            )
            hse_val = self.default_config["hotkey_stability_service_enabled"]
        self.config["hotkey_stability_service_enabled"] = hse_val

        # Validate text_correction_enabled
        tce_val = _parse_bool(
            self.config.get(
                "text_correction_enabled",
                self.default_config["text_correction_enabled"],
            )
        )
        if tce_val is None:
            logging.warning(
                f"Invalid text_correction_enabled '{self.config.get('text_correction_enabled')}'. Using default ({self.default_config['text_correction_enabled']})."
            )
            tce_val = self.default_config["text_correction_enabled"]
        self.config["text_correction_enabled"] = tce_val

        # Validate text_correction_service
        if self.config.get("text_correction_service") not in [SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI]:
            logging.warning(f"Invalid text_correction_service '{self.config.get('text_correction_service')}'. Using default ('{self.default_config['text_correction_service']}').")
            self.config["text_correction_service"] = self.default_config["text_correction_service"]

        # Validate gemini_model_options
        if not isinstance(self.config.get("gemini_model_options"), list):
            logging.warning(f"Invalid gemini_model_options. Must be a list. Using default.")
            self.config["gemini_model_options"] = self.default_config["gemini_model_options"]
        else:
            # Ensure all elements in the list are strings
            self.config["gemini_model_options"] = [str(m) for m in self.config["gemini_model_options"]]

        try:
            raw_timeout = self.config.get(
                TEXT_CORRECTION_TIMEOUT_CONFIG_KEY,
                self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY],
            )
            timeout_val = float(raw_timeout)
            if timeout_val <= 0:
                logging.warning(
                    f"Invalid text_correction_timeout '{timeout_val}'. Using default ({self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]})."
                )
                timeout_val = self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]
            self.config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY] = timeout_val
        except (ValueError, TypeError):
            logging.warning(
                f"Invalid text_correction_timeout value '{self.config.get(TEXT_CORRECTION_TIMEOUT_CONFIG_KEY)}' in config. Using default ({self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]})."
            )
            self.config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY] = self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]

        # Validate min_record_duration
        try:
            self.config["min_record_duration"] = float(self.config.get("min_record_duration", self.default_config["min_record_duration"]))
            if self.config["min_record_duration"] < 0:
                logging.warning(f"Invalid min_record_duration '{self.config['min_record_duration']}'. Must be non-negative. Using default ({self.default_config['min_record_duration']}).")
                self.config["min_record_duration"] = self.default_config["min_record_duration"]
        except (ValueError, TypeError):
            logging.warning(f"Invalid min_record_duration value '{self.config.get('min_record_duration')}' in config. Using default ({self.default_config['min_record_duration']}).")
            self.config["min_record_duration"] = self.default_config["min_record_duration"]

        # Validate keyboard_library
        if self.config.get("keyboard_library") not in ["win32", "linux", "darwin"]: # Assuming these are the only valid values
            logging.warning(f"Invalid keyboard_library '{self.config.get('keyboard_library')}'. Using default ('{self.default_config['keyboard_library']}').")
            self.config["keyboard_library"] = self.default_config["keyboard_library"]

        # Validate AI provider related models and keys (basic check for existence)
        for key in ["openrouter_model", "gemini_model", "gemini_agent_model"]:
            if not isinstance(self.config.get(key), str):
                logging.warning(f"Invalid type for '{key}'. Must be a string. Using default.")
                self.config[key] = self.default_config[key]

        for key in ["openrouter_api_key", "gemini_api_key"]:
            if not isinstance(self.config.get(key), str):
                logging.warning(f"Invalid type for '{key}'. Must be a string. Using empty string.")
                self.config[key] = ""

        # Validate prompt strings
        for key in ["openrouter_agent_prompt", "openrouter_prompt", "prompt_agentico", "gemini_prompt"]:
            if not isinstance(self.config.get(key), str):
                logging.warning(f"Invalid type for '{key}'. Must be a string. Using default.")
                self.config[key] = self.default_config[key]

        self.config["record_key"] = str(self.config.get("record_key", self.default_config["record_key"])).lower()
        self.config["record_mode"] = str(self.config.get("record_mode", self.default_config["record_mode"])).lower()
        if self.config["record_mode"] not in ["toggle", "press"]:
            logging.warning(f"Invalid record_mode '{self.config['record_mode']}'. Falling back to '{self.default_config['record_mode']}'.")
            self.config["record_mode"] = self.default_config['record_mode']
        
        # Unificar auto_paste e agent_auto_paste
        self.config["auto_paste"] = _parse_bool(
            self.config.get("auto_paste", self.default_config["auto_paste"]),
            default=self.default_config["auto_paste"],
        )
        self.config["agent_auto_paste"] = self.config["auto_paste"]  # Garante que agent_auto_paste seja sempre igual a auto_paste

        # Flag para exibir transcrições brutas no log
        self.config[DISPLAY_TRANSCRIPTS_KEY] = _parse_bool(
            self.config.get(
                DISPLAY_TRANSCRIPTS_KEY,
                self.default_config[DISPLAY_TRANSCRIPTS_KEY],
            ),
            default=self.default_config[DISPLAY_TRANSCRIPTS_KEY],
        )

        # Persistência opcional de gravações temporárias
        self.config[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = _parse_bool(
            self.config.get(
                SAVE_TEMP_RECORDINGS_CONFIG_KEY,
                self.default_config[SAVE_TEMP_RECORDINGS_CONFIG_KEY],
            ),
            default=self.default_config[SAVE_TEMP_RECORDINGS_CONFIG_KEY],
        )

        # Uso opcional do Flash Attention 2
        self.config[USE_FLASH_ATTENTION_2_CONFIG_KEY] = _parse_bool(
            self.config.get(
                USE_FLASH_ATTENTION_2_CONFIG_KEY,
                self.default_config[USE_FLASH_ATTENTION_2_CONFIG_KEY],
            ),
            default=self.default_config[USE_FLASH_ATTENTION_2_CONFIG_KEY],
        )
        if not isinstance(self.config.get(USE_FLASH_ATTENTION_2_CONFIG_KEY), bool):
            logging.warning(
                f"Invalid type for '{USE_FLASH_ATTENTION_2_CONFIG_KEY}'. Must be a boolean. Using default."
            )
            self.config[USE_FLASH_ATTENTION_2_CONFIG_KEY] = self.default_config[
                USE_FLASH_ATTENTION_2_CONFIG_KEY
            ]
    
        # Para gpu_index_specified e batch_size_specified
        self.config["batch_size_specified"] = BATCH_SIZE_CONFIG_KEY in loaded_config
        self.config["gpu_index_specified"] = GPU_INDEX_CONFIG_KEY in loaded_config
        
        # Lógica de validação para gpu_index
        try:
            raw_gpu_idx_val = loaded_config.get(GPU_INDEX_CONFIG_KEY, -1)
            gpu_idx_val = int(raw_gpu_idx_val)
            if gpu_idx_val < -1:
                logging.warning(f"Invalid GPU index '{gpu_idx_val}'. Must be -1 (auto) or >= 0. Using auto (-1).")
                self.config[GPU_INDEX_CONFIG_KEY] = -1
            else:
                self.config[GPU_INDEX_CONFIG_KEY] = gpu_idx_val
        except (ValueError, TypeError):
            logging.warning(f"Invalid GPU index value '{self.config.get(GPU_INDEX_CONFIG_KEY)}' in config. Falling back to automatic selection (-1).")
            self.config[GPU_INDEX_CONFIG_KEY] = -1
            self.config["gpu_index_specified"] = False # Se falhou a leitura, não foi especificado corretamente

        # Lógica para uso do modelo Turbo
        turbo_val = _parse_bool(
            self.config.get(USE_TURBO_CONFIG_KEY, self.default_config[USE_TURBO_CONFIG_KEY]),
            default=self.default_config[USE_TURBO_CONFIG_KEY],
        )
        self.config[USE_TURBO_CONFIG_KEY] = turbo_val

        # Lógica de validação para min_transcription_duration
        try:
            raw_min_duration_val = loaded_config.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY, self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY])
            min_duration_val = float(raw_min_duration_val)
            if not (0.1 <= min_duration_val <= 10.0): # Exemplo de range razoável
                logging.warning(f"Invalid min_transcription_duration '{min_duration_val}'. Must be between 0.1 and 10.0. Using default ({self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]}).")
                self.config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY] = self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]
            else:
                self.config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY] = min_duration_val
        except (ValueError, TypeError):
            logging.warning(f"Invalid min_transcription_duration value '{self.config.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY)}' in config. Falling back to default ({self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]}).")
            self.config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY] = self.default_config[MIN_TRANSCRIPTION_DURATION_CONFIG_KEY]

        # Lógica de validação para text_correction_timeout
        try:
            raw_timeout_val = loaded_config.get(
                TEXT_CORRECTION_TIMEOUT_CONFIG_KEY,
                self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY],
            )
            timeout_val = float(raw_timeout_val)
            if timeout_val <= 0:
                logging.warning(
                    f"Invalid text_correction_timeout '{timeout_val}'. Using default ({self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]})."
                )
                self.config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY] = self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]
            else:
                self.config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY] = timeout_val
        except (ValueError, TypeError):
            logging.warning(
                f"Invalid text_correction_timeout value '{self.config.get(TEXT_CORRECTION_TIMEOUT_CONFIG_KEY)}' in config. Falling back to default ({self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]})."
            )
            self.config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY] = self.default_config[TEXT_CORRECTION_TIMEOUT_CONFIG_KEY]

        # Lógica para uso do VAD
        self.config[USE_VAD_CONFIG_KEY] = _parse_bool(
            self.config.get(USE_VAD_CONFIG_KEY, self.default_config[USE_VAD_CONFIG_KEY]),
            default=self.default_config[USE_VAD_CONFIG_KEY],
        )
        self.config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY] = _parse_bool(
            self.config.get(
                DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY,
                self.default_config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY],
            ),
            default=self.default_config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY],
        )
        try:
            raw_threshold = self.config.get(VAD_THRESHOLD_CONFIG_KEY, self.default_config[VAD_THRESHOLD_CONFIG_KEY])
            self.config[VAD_THRESHOLD_CONFIG_KEY] = float(raw_threshold)
        except (ValueError, TypeError):
            logging.warning(
                f"Invalid vad_threshold value '{self.config.get(VAD_THRESHOLD_CONFIG_KEY)}' in config. Using default ({self.default_config[VAD_THRESHOLD_CONFIG_KEY]})."
            )
            self.config[VAD_THRESHOLD_CONFIG_KEY] = self.default_config[VAD_THRESHOLD_CONFIG_KEY]

        try:
            raw_silence = self.config.get(VAD_SILENCE_DURATION_CONFIG_KEY, self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY])
            silence_val = float(raw_silence)
            if silence_val < 0.1:
                logging.warning(
                    f"Invalid vad_silence_duration '{silence_val}'. Must be >= 0.1. Using default ({self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]})."
                )
                silence_val = self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = silence_val
        except (ValueError, TypeError):
            logging.warning(
                f"Invalid vad_silence_duration value '{self.config.get(VAD_SILENCE_DURATION_CONFIG_KEY)}' in config. Using default ({self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]})."
            )
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]

        safe_config = self.config.copy()
        safe_config.pop(GEMINI_API_KEY_CONFIG_KEY, None)
        safe_config.pop(OPENROUTER_API_KEY_CONFIG_KEY, None)
        logging.info(f"Configurações aplicadas: {safe_config}")


    def save_config(self):
        """Salva as configurações não sensíveis no config.json e as sensíveis no secrets.json."""
        config_to_save = copy.deepcopy(self.config)
        secrets_to_save = {}

        secret_keys = [GEMINI_API_KEY_CONFIG_KEY, OPENROUTER_API_KEY_CONFIG_KEY]

        # Separar segredos da configuração principal
        for key in secret_keys:
            if key in config_to_save:
                secrets_to_save[key] = config_to_save.pop(key)

        # Remover chaves não persistentes
        keys_to_ignore = ["tray_menu_items", "hotkey_manager"]
        for key in keys_to_ignore:
            if key in config_to_save:
                del config_to_save[key]

        # Salvar config.json apenas se mudar
        new_config_hash = self._compute_hash(config_to_save)
        if new_config_hash != self._config_hash:
            temp_file_config = self.config_file + ".tmp"
            try:
                with open(temp_file_config, "w", encoding='utf-8') as f:
                    json.dump(config_to_save, f, indent=4)
                os.replace(temp_file_config, self.config_file)
                self._config_hash = new_config_hash
                logging.info(f"Configuration saved to {self.config_file}")
            except Exception as e:
                logging.error(f"Error saving configuration to {self.config_file}: {e}")
                if os.path.exists(temp_file_config):
                    os.remove(temp_file_config)
        else:
            logging.info(f"Nenhuma alteração detectada em {self.config_file}.")

        # Salvar secrets.json somente se houver mudanças
        temp_file_secrets = SECRETS_FILE + ".tmp"
        existing_secrets = {}
        if os.path.exists(SECRETS_FILE):
            try:
                with open(SECRETS_FILE, "r", encoding='utf-8') as f:
                    existing_secrets = json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Could not decode {SECRETS_FILE}, will overwrite.")
            except FileNotFoundError:
                pass

        existing_secrets.update(secrets_to_save)
        new_secrets_hash = self._compute_hash(existing_secrets)

        if new_secrets_hash != self._secrets_hash:
            if existing_secrets or os.path.exists(SECRETS_FILE):
                try:
                    with open(temp_file_secrets, "w", encoding='utf-8') as f:
                        json.dump(existing_secrets, f, indent=4)
                    os.replace(temp_file_secrets, SECRETS_FILE)
                    logging.info(f"Secrets saved to {SECRETS_FILE}")
                except Exception as e:
                    logging.error(f"Error saving secrets to {SECRETS_FILE}: {e}")
                    if os.path.exists(temp_file_secrets):
                        os.remove(temp_file_secrets)
                else:
                    self._secrets_hash = new_secrets_hash
            else:
                # Não há arquivo nem segredos a salvar
                self._secrets_hash = new_secrets_hash
        else:
            logging.info(f"Nenhuma alteração detectada em {SECRETS_FILE}.")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def get_api_key(self, provider: str) -> str:
        if provider == SERVICE_GEMINI:
            return self.get(GEMINI_API_KEY_CONFIG_KEY)
        if provider == SERVICE_OPENROUTER:
            return self.get(OPENROUTER_API_KEY_CONFIG_KEY)
        return ""

    def get_use_vad(self):
        return self.config.get(USE_VAD_CONFIG_KEY, self.default_config[USE_VAD_CONFIG_KEY])

    def set_use_vad(self, value: bool):
        self.config[USE_VAD_CONFIG_KEY] = bool(value)

    def get_vad_threshold(self):
        return self.config.get(VAD_THRESHOLD_CONFIG_KEY, self.default_config[VAD_THRESHOLD_CONFIG_KEY])

    def set_vad_threshold(self, value: float):
        try:
            self.config[VAD_THRESHOLD_CONFIG_KEY] = float(value)
        except (ValueError, TypeError):
            self.config[VAD_THRESHOLD_CONFIG_KEY] = self.default_config[VAD_THRESHOLD_CONFIG_KEY]

    def get_vad_silence_duration(self):
        return self.config.get(VAD_SILENCE_DURATION_CONFIG_KEY, self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY])

    def set_vad_silence_duration(self, value: float):
        try:
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = float(value)
        except (ValueError, TypeError):
            self.config[VAD_SILENCE_DURATION_CONFIG_KEY] = self.default_config[VAD_SILENCE_DURATION_CONFIG_KEY]

    def get_use_flash_attention_2(self):
        return self.config.get(USE_FLASH_ATTENTION_2_CONFIG_KEY, self.default_config[USE_FLASH_ATTENTION_2_CONFIG_KEY])

    def set_use_flash_attention_2(self, value: bool):
        self.config[USE_FLASH_ATTENTION_2_CONFIG_KEY] = bool(value)

    def get_display_transcripts_in_terminal(self):
        return self.config.get(
            DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY,
            self.default_config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY]
        )

    def set_display_transcripts_in_terminal(self, value: bool):
        self.config[DISPLAY_TRANSCRIPTS_IN_TERMINAL_CONFIG_KEY] = bool(value)

    def get_save_temp_recordings(self):
        return self.config.get(
            SAVE_TEMP_RECORDINGS_CONFIG_KEY,
            self.default_config[SAVE_TEMP_RECORDINGS_CONFIG_KEY],
        )

    def set_save_temp_recordings(self, value: bool):
        self.config[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = bool(value)

    def get_use_turbo(self):
        return self.config.get(USE_TURBO_CONFIG_KEY, self.default_config[USE_TURBO_CONFIG_KEY])

    def set_use_turbo(self, value: bool):
        self.config[USE_TURBO_CONFIG_KEY] = bool(value)
