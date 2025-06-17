import os
import json
import logging
import copy

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
    "gemini_model": "gemini-2.5-flash-preview-05-20",
    "gemini_agent_model": "gemini-2.5-flash-preview-05-20",
    "prompt_agentico": "Você é um assistente de IA que executa comandos de texto. O usuário fornecerá uma instrução seguida do texto a ser processado. Sua tarefa é executar a instrução sobre o texto e retornar APENAS o resultado final. Não adicione explicações, saudações ou qualquer texto extra. A instrução do usuário é a prioridade máxima. O idioma de saída deve corresponder ao idioma principal do texto fornecido.",
    "gemini_prompt": """You are a speech-to-text correction specialist. Your task is to refine the following transcribed speech.
Key instructions:
- Remove self-corrections (when I say something wrong and then correct myself)
- Focus on removing speech-specific redundancies (repeated words, filler phrases, false starts)
- Make the text MUCH MORE FLUID AND COHERENT
- Remove possible errors in speech
- Maintain the speaker's emotional tone
- Keep the text as fluid as possible (IMPORTANT!)
- Preserve all language transitions (Portuguese/Spanish/English) exactly as they occur
- Connect related thoughts that may be fragmented in the transcription
- Maintain the core message and meaning, but fix speech errors and disfluencies
Return only the improved text without explanations.
Transcribed speech: {text}""",
    "batch_size": 16, # Valor padrão para o modo automático
    "batch_size_mode": "auto", # Novo: 'auto' ou 'manual'
    "manual_batch_size": 8, # Novo: Valor para o modo manual
    "gpu_index": 0,
    "hotkey_stability_service_enabled": True, # Nova configuração unificada
    "use_vad": False,
    "vad_threshold": 0.5,
    # Duração máxima da pausa preservada antes que o silêncio seja descartado
    "vad_silence_duration": 1.0,
    "gemini_model_options": [
        "gemini-2.0-flash-001",
        "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro-preview-06-05"
    ],
    "save_audio_for_debug": False,
    "min_transcription_duration": 1.0, # Nova configuração
    "display_transcripts": False
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
SAVE_AUDIO_FOR_DEBUG_CONFIG_KEY = "save_audio_for_debug"
USE_VAD_CONFIG_KEY = "use_vad"
VAD_THRESHOLD_CONFIG_KEY = "vad_threshold"
VAD_SILENCE_DURATION_CONFIG_KEY = "vad_silence_duration"
KEYBOARD_LIBRARY_CONFIG_KEY = "keyboard_library"
KEYBOARD_LIB_WIN32 = "win32"
TEXT_CORRECTION_ENABLED_CONFIG_KEY = "text_correction_enabled"
TEXT_CORRECTION_SERVICE_CONFIG_KEY = "text_correction_service"
SERVICE_NONE = "none"
SERVICE_OPENROUTER = "openrouter"
SERVICE_GEMINI = "gemini"
OPENROUTER_API_KEY_CONFIG_KEY = "openrouter_api_key"
OPENROUTER_MODEL_CONFIG_KEY = "openrouter_model"
GEMINI_API_KEY_CONFIG_KEY = "gemini_api_key"
GEMINI_MODEL_CONFIG_KEY = "gemini_model"
GEMINI_AGENT_MODEL_CONFIG_KEY = "gemini_agent_model"
GEMINI_MODEL_OPTIONS_CONFIG_KEY = "gemini_model_options"
SETTINGS_WINDOW_GEOMETRY = "550x700"
REREGISTER_INTERVAL_SECONDS = 60
MAX_HOTKEY_FAILURES = 3
HOTKEY_HEALTH_CHECK_INTERVAL = 10
DISPLAY_TRANSCRIPTS_KEY = "display_transcripts"

class ConfigManager:
    def __init__(self, config_file=CONFIG_FILE, default_config=DEFAULT_CONFIG):
        self.config_file = config_file
        self.default_config = default_config
        self.config = {}
        self.load_config()

    def load_config(self):
        cfg = copy.deepcopy(self.default_config) # Usar deepcopy para evitar modificações no default

        # 1. Carregar config.json (configurações do usuário)
        loaded_config_from_file = {}
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding='utf-8') as f:
                    loaded_config_from_file = json.load(f)
                if "new_prompt_agentico" in loaded_config_from_file:
                    logging.info("Removing obsolete 'new_prompt_agentico' key from config file.")
                    loaded_config_from_file.pop("new_prompt_agentico", None)
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
                cfg.update(secrets_loaded) # Secrets sobrescrevem configs se houver conflito
                logging.info(f"Secrets loaded from {SECRETS_FILE}.")
            else:
                logging.info(f"{SECRETS_FILE} not found. API keys might be missing.")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.warning(f"Error reading or decoding {SECRETS_FILE}: {e}. API keys might be missing or invalid.")
            secrets_loaded = {} # Resetar segredos em caso de erro
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading {SECRETS_FILE}: {e}. API keys might be missing or invalid.", exc_info=True)
            secrets_loaded = {}

        self.config = cfg
        # Aplicar validação e conversão de tipo
        self._validate_and_apply_config(loaded_config_from_file) # Passar o config.json carregado para validação de 'specified'
        
        # Salvar a configuração (apenas as não sensíveis) após o carregamento e validação
        self.save_config()

    def _validate_and_apply_config(self, loaded_config):
        self.config["record_key"] = str(self.config.get("record_key", self.default_config["record_key"])).lower()
        self.config["record_mode"] = str(self.config.get("record_mode", self.default_config["record_mode"])).lower()
        if self.config["record_mode"] not in ["toggle", "press"]:
            logging.warning(f"Invalid record_mode '{self.config['record_mode']}'. Falling back to '{self.default_config['record_mode']}'.")
            self.config["record_mode"] = self.default_config['record_mode']
        
        # Unificar auto_paste e agent_auto_paste
        self.config["auto_paste"] = bool(self.config.get("auto_paste", self.default_config["auto_paste"]))
        self.config["agent_auto_paste"] = self.config["auto_paste"] # Garante que agent_auto_paste seja sempre igual a auto_paste

        # Flag para exibir transcrições brutas no log
        self.config[DISPLAY_TRANSCRIPTS_KEY] = bool(
            self.config.get(DISPLAY_TRANSCRIPTS_KEY, self.default_config[DISPLAY_TRANSCRIPTS_KEY])
        )
    
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

        # Lógica para uso do VAD
        self.config[USE_VAD_CONFIG_KEY] = bool(self.config.get(USE_VAD_CONFIG_KEY, self.default_config[USE_VAD_CONFIG_KEY]))
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

        logging.info(f"Configurações aplicadas: {self.config}")


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

        # Salvar config.json
        temp_file_config = self.config_file + ".tmp"
        try:
            with open(temp_file_config, "w", encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
            os.replace(temp_file_config, self.config_file)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Error saving configuration to {self.config_file}: {e}")
            if os.path.exists(temp_file_config):
                os.remove(temp_file_config)

        # Salvar secrets.json se houver segredos
        if secrets_to_save:
            temp_file_secrets = SECRETS_FILE + ".tmp"
            # Ler segredos existentes para não sobrescrevê-los
            existing_secrets = {}
            if os.path.exists(SECRETS_FILE):
                try:
                    with open(SECRETS_FILE, "r", encoding='utf-8') as f:
                        existing_secrets = json.load(f)
                except json.JSONDecodeError:
                    logging.warning(f"Could not decode {SECRETS_FILE}, will overwrite.")
                except FileNotFoundError:
                    pass # Não faz nada se o arquivo não existe, será criado

            existing_secrets.update(secrets_to_save)

            try:
                with open(temp_file_secrets, "w", encoding='utf-8') as f:
                    json.dump(existing_secrets, f, indent=4)
                os.replace(temp_file_secrets, SECRETS_FILE)
                logging.info(f"Secrets saved to {SECRETS_FILE}")
            except Exception as e:
                logging.error(f"Error saving secrets to {SECRETS_FILE}: {e}")
                if os.path.exists(temp_file_secrets):
                    os.remove(temp_file_secrets)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

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
