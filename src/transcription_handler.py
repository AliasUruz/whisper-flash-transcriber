import logging
import threading
import concurrent.futures
import time
import numpy as np
import torch

from .openrouter_api import OpenRouterAPI  # Assumindo que está na raiz ou em path acessível
from .utils import select_batch_size
from .config_manager import (
    BATCH_SIZE_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY,
    MANUAL_BATCH_SIZE_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE,
    SERVICE_OPENROUTER,
    SERVICE_GEMINI,
    OPENROUTER_API_KEY_CONFIG_KEY,
    OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    GEMINI_AGENT_PROMPT_CONFIG_KEY,
    OPENROUTER_AGENT_PROMPT_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    OPENROUTER_PROMPT_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    CHUNK_LENGTH_SEC_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_BACKEND_CONFIG_KEY,
    CLEAR_GPU_CACHE_CONFIG_KEY,
    ASR_COMPUTE_DEVICE_CONFIG_KEY,
    ASR_DTYPE_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
    ASR_CT2_CPU_THREADS_CONFIG_KEY,
    ASR_CACHE_DIR_CONFIG_KEY,
)
from .asr.backends import make_backend

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
    ):
        self.config_manager = config_manager
        self.gemini_client = gemini_api_client # Instância da API Gemini injetada
        self.gemini_api = gemini_api_client
        self.on_model_ready_callback = on_model_ready_callback
        self.on_model_error_callback = on_model_error_callback
        self.on_transcription_result_callback = on_transcription_result_callback # Para resultado final
        self.on_agent_result_callback = on_agent_result_callback # Para resultado do agente
        self.on_segment_transcribed_callback = on_segment_transcribed_callback # Para segmentos em tempo real
        self.is_state_transcribing_fn = is_state_transcribing_fn
        self.core_instance_ref = None  # Referência ao AppCore
        # "state_check_callback" é preservado apenas para retrocompatibilidade;
        # utilize "is_state_transcribing_fn" nas novas implementações.
        self.state_check_callback = is_state_transcribing_fn
        self.correction_in_progress = False
        self.correction_thread = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.pipe = None
        self.backend_resolved = None
        # Futura tarefa de transcrição em andamento
        self.transcription_future = None
        # Executor dedicado para a tarefa de transcrição em background
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        # Evento para sinalizar cancelamento de transcrição em andamento
        self.transcription_cancel_event = threading.Event()

        # Estado do backend de ASR
        self._asr_backend = None
        self._asr_backend_name = None
        self._asr_model_id = None
        self._asr_loaded = False

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
        self.gemini_prompt = self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY)
        self.min_transcription_duration = self.config_manager.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY)
        self.chunk_length_sec = self.config_manager.get(CHUNK_LENGTH_SEC_CONFIG_KEY)
        self.chunk_length_mode = self.config_manager.get("chunk_length_mode", "manual")
        self.enable_torch_compile = bool(self.config_manager.get("enable_torch_compile", False))
        # Configurações de ASR
        self._asr_backend_name = self.config_manager.get(ASR_BACKEND_CONFIG_KEY)
        self._asr_model_id = self.config_manager.get(ASR_MODEL_ID_CONFIG_KEY)
        self.asr_compute_device = self.config_manager.get(ASR_COMPUTE_DEVICE_CONFIG_KEY)
        self.asr_dtype = self.config_manager.get(ASR_DTYPE_CONFIG_KEY)
        self.asr_ct2_compute_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        self.asr_ct2_cpu_threads = self.config_manager.get(ASR_CT2_CPU_THREADS_CONFIG_KEY)
        self.asr_cache_dir = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)

        self.openrouter_client = None
        # self.gemini_client é injetado
        self.device_in_use = None # Nova variável para armazenar o dispositivo em uso

        self._init_api_clients()

    def _init_api_clients(self):
        # Lógica de inicialização de OpenRouterAPI e GeminiAPI
        # (movida de WhisperCore._init_openrouter_client e _init_gemini_client)
        # ...
        self.openrouter_client = None
        self.openrouter_api = None
        if self.text_correction_enabled and self.text_correction_service == SERVICE_OPENROUTER and self.openrouter_api_key and OpenRouterAPI:
            try:
                self.openrouter_client = OpenRouterAPI(api_key=self.openrouter_api_key, model_id=self.openrouter_model)
                self.openrouter_api = self.openrouter_client
                logging.info("OpenRouter API client initialized.")
            except Exception as e:
                logging.error(f"Error initializing OpenRouter API client: {e}")

        # O cliente Gemini agora é injetado, então sua inicialização foi removida daqui.
        # A inicialização do OpenRouter é mantida.

    @property
    def asr_backend(self):
        return self._asr_backend_name

    @asr_backend.setter
    def asr_backend(self, value):
        if value != self._asr_backend_name:
            self._asr_backend_name = value
            self.reload_asr()

    @property
    def asr_model_id(self):
        return self._asr_model_id

    @asr_model_id.setter
    def asr_model_id(self, value):
        if value != self._asr_model_id:
            self._asr_model_id = value
            self.reload_asr()

    def unload(self):
        """Descarta a pipeline atual."""
        self.pipe = None

    def reload_asr(self):
        """Recarrega o backend de ASR e o modelo associado."""
        if hasattr(self, "_asr_backend") and self._asr_backend is not None:
            try:
                self._asr_backend.unload()
                logging.info("Backend ASR anterior descarregado.")
            except Exception as e:
                logging.warning(f"Falha ao descarregar backend ASR: {e}")

        if bool(self.config_manager.get(CLEAR_GPU_CACHE_CONFIG_KEY)) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Inicia o carregamento do novo modelo em uma thread
        self.start_model_loading()

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
        self.gemini_prompt = self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY)
        self.min_transcription_duration = self.config_manager.get(MIN_TRANSCRIPTION_DURATION_CONFIG_KEY)
        self.chunk_length_sec = self.config_manager.get(CHUNK_LENGTH_SEC_CONFIG_KEY)
        self.chunk_length_mode = self.config_manager.get("chunk_length_mode", "manual")
        self.enable_torch_compile = bool(self.config_manager.get("enable_torch_compile", False))
        self.asr_backend = self.config_manager.get(ASR_BACKEND_CONFIG_KEY)
        self.asr_model_id = self.config_manager.get(ASR_MODEL_ID_CONFIG_KEY)
        self.asr_compute_device = self.config_manager.get(ASR_COMPUTE_DEVICE_CONFIG_KEY)
        self.asr_dtype = self.config_manager.get(ASR_DTYPE_CONFIG_KEY)
        self.asr_ct2_compute_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        self.asr_cache_dir = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)
        logging.info("TranscriptionHandler: Configurações atualizadas.")

    def _resolve_asr_settings(self):
        """Determina backend, modelo e parâmetros de ASR conforme hardware."""
        backend = (self.asr_backend or "auto").lower()
        compute_device = (self.asr_compute_device or "auto").lower()
        model_id = self.asr_model_id or "auto"
        dtype = (self.asr_dtype or "auto").lower()

        if backend == "auto":
            backend = "transformers"

        if compute_device == "auto":
            compute_device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype == "auto":
            dtype = "float16" if compute_device.startswith("cuda") else "float32"

        default_gpu_model = "openai/whisper-large-v3-turbo"
        default_cpu_model = "openai/whisper-large-v3"
        if model_id in ("auto", default_gpu_model, default_cpu_model):
            model_id = default_gpu_model if compute_device.startswith("cuda") else default_cpu_model

        return backend, model_id, compute_device, dtype


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

    def _async_text_correction(self, text: str, is_agent_mode: bool, gemini_prompt: str, openrouter_prompt: str, was_transcribing_when_started: bool):
        if not self.text_correction_enabled:
            self.correction_in_progress = False
            self.on_transcription_result_callback(text, text)
            return

        self.correction_in_progress = True
        corrected = text  # Default to original text
        future = None
        try:
            active_provider = self._get_text_correction_service()
            if active_provider == SERVICE_NONE:
                logging.info(
                    "Correção de texto desativada ou provedor indisponível."
                )
                return

            api_key = self.config_manager.get_api_key(active_provider)

            if not api_key:
                logging.warning(
                    f"Nenhuma chave de API encontrada para o provedor {active_provider}. Pulando correção de texto."
                )
                return

            if active_provider == "gemini":
                if not is_agent_mode:
                    prompt = gemini_prompt
                else:
                    logging.info("Modo Agente ativado. Usando prompt do Agente para o Gemini.")
                    prompt = self.config_manager.get(GEMINI_AGENT_PROMPT_CONFIG_KEY)
                future = self.executor.submit(self.gemini_api.correct_text_async, corrected, prompt, api_key)
                corrected = future.result()
            elif active_provider == "openrouter":
                if not is_agent_mode:
                    prompt = openrouter_prompt
                else:
                    logging.info("Modo Agente ativado. Usando prompt do Agente para o OpenRouter.")
                    prompt = self.config_manager.get(OPENROUTER_AGENT_PROMPT_CONFIG_KEY)

                model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
                future = self.executor.submit(self.openrouter_api.correct_text_async, corrected, prompt, api_key, model)
                corrected = future.result()
            else:
                logging.error(f"Provedor de IA desconhecido: {active_provider}")

        except Exception as exc:
            logging.error(f"Erro ao corrigir texto: {exc}")
            if future and not future.done():
                future.cancel()
        finally:
            self.correction_in_progress = False
            # O resultado da correção deve ser sempre retornado, independentemente
            # de uma mudança de estado subsequente, para evitar perda de dados do usuário.
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
        return select_batch_size(
            self.gpu_index,
            fallback=self.batch_size,
            chunk_length_sec=self.chunk_length_sec
        )

    def start_model_loading(self):
        threading.Thread(target=self._load_model_task, daemon=True, name="ModelLoadThread").start()

    def is_transcription_running(self) -> bool:
        """Indica se existe tarefa de transcrição ainda não concluída."""
        return (
            self.transcription_future is not None
            and not self.transcription_future.done()
        )

    def is_text_correction_running(self) -> bool:
        """Indica se há correção de texto em andamento."""
        return self.correction_in_progress

    def stop_transcription(self) -> None:
        """Sinaliza que a transcrição em andamento deve ser cancelada."""
        self.transcription_cancel_event.set()

    def _load_model_task(self):
        try:
            backend_name, model_id, compute_device, dtype = self._resolve_asr_settings()

            logging.info(
                "Carregando backend ASR: %s com modelo %s em %s (%s)",
                backend_name,
                model_id,
                compute_device,
                dtype,
            )

            self._asr_backend = make_backend(backend_name)

            attn_impl = "sdpa"
            try:
                import importlib.util

                if importlib.util.find_spec("flash_attn") is not None:
                    attn_impl = "flash_attn2"
                    logging.info("FlashAttention 2 detectado.")
            except Exception:
                pass

            self._asr_backend.load(
                model_id=model_id,
                device=compute_device,
                dtype=dtype,
                ct2_compute_type=self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
                cache_dir=self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY),
                attn_implementation=attn_impl,
            )

            self._asr_backend.warmup()
            self._asr_loaded = True
            self.on_model_ready_callback()
            logging.info("Backend ASR e modelo carregados com sucesso.")

        except ImportError as e:
            self._asr_loaded = False
            error_message = f"Falha ao importar dependências do backend ASR: {e}"
            logging.error(error_message, exc_info=True)
            self.on_model_error_callback(error_message)
        except Exception as e:
            self._asr_loaded = False
            error_message = f"Falha ao carregar backend ASR: {e}"
            logging.error(error_message, exc_info=True)
            self.on_model_error_callback(error_message)

    def transcribe_audio_segment(self, audio_source: str | np.ndarray, agent_mode: bool = False):
        """Envia o áudio (arquivo ou array) para transcrição assíncrona."""
        self.transcription_future = self.transcription_executor.submit(
            self._transcription_task, audio_source, agent_mode
        )

    def _transcription_task(self, audio_source: str | np.ndarray, agent_mode: bool) -> None:
        """Tarefa executada em uma thread para transcrever um segmento de áudio."""
        self.transcription_cancel_event.clear()
        text_result = ""

        try:
            if not self._asr_loaded or self._asr_backend is None:
                error_message = "Backend ASR indisponível. Modelo não carregado ou falhou."
                logging.error(error_message)
                self.on_model_error_callback(error_message)
                return

            if self.transcription_cancel_event.is_set():
                logging.info("Transcrição interrompida antes do início do processamento.")
                return

            dynamic_batch_size = self._get_dynamic_batch_size()
            logging.info(
                f"Iniciando transcrição com backend: {type(self._asr_backend).__name__}, batch_size={dynamic_batch_size}"
            )

            result = self._asr_backend.transcribe(
                audio_source,
                chunk_length_s=float(self.chunk_length_sec),
                batch_size=dynamic_batch_size,
            )

            text_result = result.get("text", "").strip() or "[No speech detected]"
            logging.info("Transcrição concluída com sucesso.")

        except Exception as e:
            logging.error(f"Erro durante a tarefa de transcrição: {e}", exc_info=True)
            text_result = f"[Transcription Error: {e}]"

        finally:
            if self.transcription_cancel_event.is_set():
                logging.info("Transcrição interrompida durante o processamento. Resultado descartado.")
                return

            if agent_mode:
                if self.on_agent_result_callback:
                    self.on_agent_result_callback(text_result)
            elif self.on_transcription_result_callback:
                # O callback lida com o texto bruto e o corrigido (se habilitado)
                self.on_transcription_result_callback(text_result, text_result)

    def _effective_chunk_length(self) -> float:
        """
        Heurística simples para chunk_length_sec quando em modo 'auto'.
        Baseada na VRAM livre estimada da GPU alvo.
        """
        try:
            if not torch.cuda.is_available() or self.gpu_index < 0:
                return float(self.chunk_length_sec)  # manter o manual atual em CPU
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(f"cuda:{self.gpu_index}"))
            free_gb = free_bytes / (1024 ** 3)
            # Heurística: mais VRAM livre -> chunks maiores
            if free_gb >= 12:
                return 60.0
            elif free_gb >= 8:
                return 45.0
            elif free_gb >= 6:
                return 30.0
            elif free_gb >= 4:
                return 20.0
            else:
                return 15.0
        except Exception:
            # fallback seguro
            try:
                return float(self.chunk_length_sec)
            except Exception:
                return 30.0

    def shutdown(self) -> None:
        """Encerra o executor de transcrição."""
        try:
            # Sinaliza cancelamento para qualquer tarefa em andamento
            self.transcription_cancel_event.set()

            # Aguarda a conclusão da thread de correção, se estiver ativa
            if hasattr(self, "correction_thread") and self.correction_thread:
                if self.correction_thread.is_alive():
                    self.correction_thread.join(timeout=1)

            self.transcription_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logging.error(f"Erro ao encerrar o executor de transcrição: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug(
                    "Cache da GPU liberado no encerramento do aplicativo."
                )
