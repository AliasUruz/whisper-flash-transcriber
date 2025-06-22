import logging
import threading
import concurrent.futures
import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from .openrouter_api import OpenRouterAPI # Assumindo que está na raiz ou em path acessível
import numpy as np # Necessário para o audio_input

# Importar constantes de configuração
from utils import select_batch_size
from .config_manager import (
    BATCH_SIZE_CONFIG_KEY, GPU_INDEX_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY, MANUAL_BATCH_SIZE_CONFIG_KEY, # Novos
    TEXT_CORRECTION_ENABLED_CONFIG_KEY, TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI,
    OPENROUTER_API_KEY_CONFIG_KEY, OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY, DISPLAY_TRANSCRIPTS_KEY, # Nova constante
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
)

class TranscriptionHandler:
    def __init__(
        self,
        config_manager,
        gemini_api_client,
        on_model_ready_callback,
        on_model_error_callback,
        on_transcription_result_callback,
        on_agent_result_callback,
        on_segment_transcribed_callback,
        is_state_transcribing_fn,
        on_transcription_cancelled_callback=None,
    ):
        self.config_manager = config_manager
        self.gemini_client = gemini_api_client # Instância da API Gemini injetada
        self.on_model_ready_callback = on_model_ready_callback
        self.on_model_error_callback = on_model_error_callback
        self.on_transcription_result_callback = on_transcription_result_callback # Para resultado final
        self.on_agent_result_callback = on_agent_result_callback # Para resultado do agente
        self.on_segment_transcribed_callback = on_segment_transcribed_callback # Para segmentos em tempo real
        self.is_state_transcribing_fn = is_state_transcribing_fn
        self.on_transcription_cancelled_callback = on_transcription_cancelled_callback
        # Alias para manter compatibilidade com referências existentes
        self.state_check_callback = is_state_transcribing_fn
        self.correction_cancel_event = threading.Event()
        self.transcription_cancel_event = threading.Event()
        self.correction_in_progress = False

        self.pipe = None
        self.transcription_in_progress = False
        self.transcription_lock = threading.RLock()
        # Executor dedicado para a tarefa de transcrição em background
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )

        # Configurações de modelo e API (carregadas do config_manager)
        self.batch_size = self.config_manager.get(BATCH_SIZE_CONFIG_KEY) # Agora é o batch_size padrão para o modo auto
        self.batch_size_mode = self.config_manager.get(BATCH_SIZE_MODE_CONFIG_KEY) # Novo
        self.manual_batch_size = self.config_manager.get(MANUAL_BATCH_SIZE_CONFIG_KEY) # Novo
        self.gpu_index = self.config_manager.get(GPU_INDEX_CONFIG_KEY)
        self.batch_size_specified = self.config_manager.get("batch_size_specified") # Ainda usado para validação
        self.gpu_index_specified = self.config_manager.get("gpu_index_specified") # Ainda usado para validação

        self.text_correction_enabled = self.config_manager.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY)
        self.text_correction_service = self.config_manager.get(TEXT_CORRECTION_SERVICE_CONFIG_KEY)
        self.openrouter_api_key = self.config_manager.get(OPENROUTER_API_KEY_CONFIG_KEY)
        self.openrouter_model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
        self.gemini_api_key = self.config_manager.get(GEMINI_API_KEY_CONFIG_KEY)
        self.gemini_agent_model = self.config_manager.get('gemini_agent_model')
        self.gemini_prompt = self.config_manager.get("gemini_prompt")
        self.min_transcription_duration = self.config_manager.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY)

        self.openrouter_client = None
        # self.gemini_client é injetado
        self.device_in_use = None # Nova variável para armazenar o dispositivo em uso

        self._init_api_clients()
        # Removido: self._initialize_model_and_processor() # Chamada para inicializar o modelo e o processador

    def _init_api_clients(self):
        # Lógica de inicialização de OpenRouterAPI e GeminiAPI
        # (movida de WhisperCore._init_openrouter_client e _init_gemini_client)
        # ...
        self.openrouter_client = None
        if self.text_correction_enabled and self.text_correction_service == SERVICE_OPENROUTER and self.openrouter_api_key and OpenRouterAPI:
            try:
                self.openrouter_client = OpenRouterAPI(api_key=self.openrouter_api_key, model_id=self.openrouter_model)
                logging.info("OpenRouter API client initialized.")
            except Exception as e:
                logging.error(f"Error initializing OpenRouter API client: {e}")

        # O cliente Gemini agora é injetado, então sua inicialização foi removida daqui.
        # A inicialização do OpenRouter é mantida.

    def update_config(self):
        """Atualiza as configurações do handler a partir do config_manager."""
        self.batch_size = self.config_manager.get(BATCH_SIZE_CONFIG_KEY)
        self.batch_size_mode = self.config_manager.get(BATCH_SIZE_MODE_CONFIG_KEY)
        self.manual_batch_size = self.config_manager.get(MANUAL_BATCH_SIZE_CONFIG_KEY)
        self.gpu_index = self.config_manager.get(GPU_INDEX_CONFIG_KEY)
        self.text_correction_enabled = self.config_manager.get(TEXT_CORRECTION_ENABLED_CONFIG_KEY)
        self.text_correction_service = self.config_manager.get(TEXT_CORRECTION_SERVICE_CONFIG_KEY)
        self.openrouter_api_key = self.config_manager.get(OPENROUTER_API_KEY_CONFIG_KEY)
        self.openrouter_model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
        self.gemini_api_key = self.config_manager.get(GEMINI_API_KEY_CONFIG_KEY)
        self.gemini_agent_model = self.config_manager.get('gemini_agent_model')
        self.gemini_prompt = self.config_manager.get("gemini_prompt")
        self.min_transcription_duration = self.config_manager.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY)
        logging.info("TranscriptionHandler: Configurações atualizadas.")

    def _initialize_model_and_processor(self):
        # Este método será chamado para orquestrar o carregamento do modelo e a criação da pipeline
        # Ele será chamado por start_model_loading
        try:
            model, processor = self._load_model_task()
            if model and processor:
                device = f"cuda:{self.gpu_index}" if self.gpu_index >= 0 and torch.cuda.is_available() else "cpu"
                # Forçar a detecção de idioma na inicialização da pipeline
                generate_kwargs_init = {
                    "task": "transcribe",
                    "language": None
                }
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    chunk_length_s=30,
                    batch_size=self.batch_size, # Usar o batch_size configurado
                    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                    generate_kwargs=generate_kwargs_init
                )
                logging.info("Pipeline de transcrição inicializada com sucesso.")
                self.on_model_ready_callback()
            else:
                error_message = "Falha ao carregar modelo ou processador."
                logging.error(error_message)
                self.on_model_error_callback(error_message)
        except Exception as e:
            error_message = f"Erro na inicialização da pipeline: {e}"
            logging.error(error_message, exc_info=True)
            self.on_model_error_callback(error_message)

    def _get_text_correction_service(self):
        if not self.text_correction_enabled: return SERVICE_NONE
        if self.text_correction_service == SERVICE_OPENROUTER and self.openrouter_client: return SERVICE_OPENROUTER
        # Verifica se o cliente Gemini existe E se a chave é válida
        if self.text_correction_service == SERVICE_GEMINI and self.gemini_client and self.gemini_client.is_valid: return SERVICE_GEMINI
        return SERVICE_NONE

    def _correct_text_with_openrouter(self, text):
        if not self.openrouter_client or not text: return text
        try: return self.openrouter_client.correct_text(text)
        except Exception as e: logging.error(f"Error correcting text with OpenRouter API: {e}"); return text

    def _correct_text_with_gemini(self, text: str) -> str:
        """Chama o novo método de correção da API Gemini."""
        if not self.gemini_client or not text or not self.gemini_client.is_valid:
            return text
        try:
            return self.gemini_client.get_correction(text)
        except Exception as e:
            logging.error(f"Erro ao chamar get_correction da API Gemini: {e}")
            return text

    def _async_text_correction(self, text: str, service: str, cancel_event: threading.Event) -> None:
        """Corrige o texto de forma assíncrona com timeout e verificação de cancelamento."""
        corrected = text
        self.correction_in_progress = True
        def _call():
            if service == SERVICE_GEMINI:
                return self._correct_text_with_gemini(text)
            if service == SERVICE_OPENROUTER:
                return self._correct_text_with_openrouter(text)
            return text

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call)
                try:
                    corrected = future.result(timeout=30)
                except concurrent.futures.TimeoutError:
                    logging.error("Tempo esgotado na correção de texto.")
                    future.cancel()
                except Exception as exc:
                    logging.error(f"Erro ao corrigir texto: {exc}")
        finally:
            self.correction_in_progress = False
            if not cancel_event.is_set() and self.is_state_transcribing_fn and self.is_state_transcribing_fn():
                if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
                    logging.info(f"Transcrição corrigida: {corrected}")
                self.on_transcription_result_callback(corrected, text)

    def _get_dynamic_batch_size(self) -> int:
        if not torch.cuda.is_available() or self.gpu_index < 0:
            logging.info("GPU não disponível ou não selecionada, usando batch size de CPU (4).")
            return 4

        if self.batch_size_mode == "manual":
            logging.info(
                f"Modo de batch size manual selecionado. Usando valor configurado: {self.manual_batch_size}"
            )
            return self.manual_batch_size

        # Lógica para modo "auto" (dinâmico)
        return select_batch_size(self.gpu_index, fallback=self.batch_size)

    def start_model_loading(self):
        threading.Thread(target=self._initialize_model_and_processor, daemon=True, name="ModelLoadThread").start()

    def cancel_transcription(self):
        """Cancela a transcrição em andamento."""
        self.transcription_cancel_event.set()
        # Notifica o callback se ele existir
        if self.on_transcription_cancelled_callback:
            self.on_transcription_cancelled_callback()

    def is_transcription_running(self) -> bool:
        """Indica se há transcrição em andamento."""
        return self.transcription_in_progress

    def cancel_text_correction(self):
        """Cancela a correção de texto em andamento."""
        self.correction_cancel_event.set()

    def is_text_correction_running(self) -> bool:
        """Indica se há correção de texto em andamento."""
        return self.correction_in_progress

    def _load_model_task(self):
        # Removido: model_loaded_successfully = False
        # Removido: error_message = "Unknown error during model load."
        try:
            # Removido: device_param = "cpu"
            torch_dtype_local = torch.float32

            if torch.cuda.is_available():
                if self.gpu_index == -1: # Auto-seleção de GPU
                    num_gpus = torch.cuda.device_count()
                    if num_gpus > 0:
                        best_gpu_index = 0
                        max_vram = 0
                        for i in range(num_gpus):
                            props = torch.cuda.get_device_properties(i)
                            if props.total_memory > max_vram:
                                max_vram = props.total_memory
                                best_gpu_index = i
                        self.gpu_index = best_gpu_index
                        logging.info(f"Auto-seleção de GPU: {self.gpu_index} ({torch.cuda.get_device_name(self.gpu_index)})")
                    else:
                        logging.info("Nenhuma GPU disponível, usando CPU.")
                        self.gpu_index = -1 # Garante que o índice seja -1 se não houver GPU
                
            model_id = "openai/whisper-large-v3"
            
            logging.info(f"Carregando processador de {model_id}...")
            processor = AutoProcessor.from_pretrained(model_id)

            # Determinar o dispositivo explicitamente antes de carregar o modelo
            device = f"cuda:{self.gpu_index}" if self.gpu_index >= 0 and torch.cuda.is_available() else "cpu"
            logging.info(f"Dispositivo de carregamento do modelo definido explicitamente como: {device}")

            if torch.cuda.is_available() and self.gpu_index >= 0:
                torch_dtype_local = torch.float16
                logging.info("GPU detectada e selecionada, usando torch.float16.")
            else:
                torch_dtype_local = torch.float32
                logging.info("Nenhuma GPU detectada ou selecionada, usando torch.float32 (CPU).")

            logging.info(f"Carregando modelo {model_id}...")
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch_dtype_local,
                low_cpu_mem_usage=True, # Ajuda a reduzir o uso de RAM durante o carregamento
                use_safetensors=True,
                device_map={'': device} # Especifica que todo o modelo vai para o dispositivo alvo
            )
            
            # Retorna o modelo e o processador para que a pipeline seja criada fora desta função
            return model, processor

        except Exception as e:
            error_message = f"Falha ao carregar o modelo: {e}"
            logging.error(error_message, exc_info=True)
            # Removido: self.on_model_error_callback(error_message) # Notifica o erro imediatamente
            return None, None # Retorna None em caso de falha

    def transcribe_audio_segment(self, audio_input: np.ndarray, agent_mode: bool = False):
        with self.transcription_lock:
            if self.transcription_in_progress:
                logging.warning("Transcrição já em andamento, ignorando nova solicitação.")
                return
            self.transcription_in_progress = True
        self.transcription_cancel_event.clear()

        self.transcription_future = self.transcription_executor.submit(self._transcription_task, audio_input, agent_mode)

    def _transcription_task(self, audio_input: np.ndarray, agent_mode: bool) -> None:
        if self.transcription_cancel_event.is_set():
            with self.transcription_lock:
                self.transcription_in_progress = False
            logging.info("Transcrição cancelada antes do início do processamento.")
            return

        text_result = None
        try:
            if self.pipe is None:
                error_message = "Pipeline de transcrição indisponível. Modelo não carregado ou falhou."
                logging.error(error_message)
                self.on_model_error_callback(error_message)
                return

            dynamic_batch_size = self._get_dynamic_batch_size()
            logging.info(f"Iniciando transcrição de segmento com batch_size={dynamic_batch_size}...")

            generate_kwargs = {
                "task": "transcribe",
                "language": None
            }
            result = self.pipe(
                audio_input,
                chunk_length_s=30,
                batch_size=dynamic_batch_size,
                return_timestamps=False,
                generate_kwargs=generate_kwargs
            )

            if result and "text" in result:
                text_result = result["text"].strip()
                if not text_result:
                    text_result = "[No speech detected]"
                else:
                    logging.info("Transcrição de segmento bem-sucedida.")
            else:
                text_result = "[Transcription failed: Bad format]"
                logging.error(f"Formato de resultado inesperado: {result}")

        except Exception as e:
            logging.error(f"Erro durante a transcrição de segmento: {e}", exc_info=True)
            text_result = f"[Transcription Error: {e}]"

        finally:
            with self.transcription_lock:
                self.transcription_in_progress = False

            if self.transcription_cancel_event.is_set():
                logging.info("Transcrição cancelada. Resultado descartado.")
                self.transcription_cancel_event.clear()
                return

            if text_result and self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY):
                logging.info(f"Transcrição bruta: {text_result}")

            if not text_result or text_result == "[No speech detected]" or text_result.strip().startswith("[Transcription Error:"):
                logging.warning(f"Segmento processado sem texto significativo ou com erro: {text_result}")
                if text_result and self.on_segment_transcribed_callback:
                    self.on_segment_transcribed_callback(text_result or "")
                if (
                    not agent_mode
                    and text_result
                    and (
                        not self.is_state_transcribing_fn
                        or self.is_state_transcribing_fn()
                    )
                ):
                    self.on_transcription_result_callback(text_result, text_result)
                elif not agent_mode and text_result:
                    logging.warning(
                        "Estado mudou antes do resultado de transcrição. UI não será atualizada."
                    )
                return

            if self.on_segment_transcribed_callback:
                self.on_segment_transcribed_callback(text_result)

            if agent_mode:
                try:
                    logging.info(f"Enviando texto para o modo agente: '{text_result}'")
                    agent_response = self.gemini_client.get_agent_response(text_result)
                    logging.info(
                        f"Resposta recebida do modo agente: '{agent_response}'"
                    )
                    if not self.is_state_transcribing_fn or self.is_state_transcribing_fn():
                        self.on_agent_result_callback(agent_response)
                    else:
                        logging.warning(
                            "Estado mudou antes do resultado do agente. UI não será atualizada."
                        )
                except Exception as e:
                    logging.error(f"Erro ao processar o comando do agente: {e}", exc_info=True)
                    if not self.is_state_transcribing_fn or self.is_state_transcribing_fn():
                        self.on_agent_result_callback(text_result)  # Falha, retorna o texto original
                    else:
                        logging.warning(
                            "Estado mudou antes do resultado do agente. UI não será atualizada."
                        )
            else:
                service = self._get_text_correction_service()
                self.correction_cancel_event.clear()
                self.correction_thread = threading.Thread(
                    target=self._async_text_correction,
                    args=(text_result, service, self.correction_cancel_event),
                    daemon=True,
                    name="TextCorrectionThread",
                )
                self.correction_thread.start()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug("Cache da GPU limpo após tarefa de transcrição.")

    def shutdown(self) -> None:
        """Encerra o executor de transcrição."""
        try:
            self.transcription_executor.shutdown(wait=False)
        except Exception as e:
            logging.error(f"Erro ao encerrar o executor de transcrição: {e}")
