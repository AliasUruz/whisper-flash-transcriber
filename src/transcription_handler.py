import logging
import threading
import concurrent.futures
import time
import numpy as np
import torch
try:
    from whisper_flash import make_backend
except Exception:  # pragma: no cover - dependency not installed during tests
    make_backend = None  # type: ignore
from .openrouter_api import OpenRouterAPI # Assumindo que está na raiz ou em path acessível
from .audio_handler import AUDIO_SAMPLE_RATE

# Importar constantes de configuração
from .utils import select_batch_size
from .config_manager import (
    BATCH_SIZE_CONFIG_KEY, GPU_INDEX_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY, MANUAL_BATCH_SIZE_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY, TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE, SERVICE_OPENROUTER, SERVICE_GEMINI,
    OPENROUTER_API_KEY_CONFIG_KEY, OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    GEMINI_AGENT_PROMPT_CONFIG_KEY,
    OPENROUTER_AGENT_PROMPT_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    OPENROUTER_PROMPT_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY, DISPLAY_TRANSCRIPTS_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    CHUNK_LENGTH_SEC_CONFIG_KEY,
    CLEAR_GPU_CACHE_CONFIG_KEY,
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

        self._asr_backend = None
        self._asr_loaded = False
        # Futura tarefa de transcrição em andamento
        self.transcription_future = None
        # Executor dedicado para a tarefa de transcrição em background
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        # Evento para sinalizar cancelamento de transcrição em andamento
        self.transcription_cancel_event = threading.Event()

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
        try:
            self._asr_backend.unload()
        except Exception as e:
            logging.warning(f"Falha ao descarregar backend ASR: {e}")

        if bool(self.config_manager.get(CLEAR_GPU_CACHE_CONFIG_KEY)) and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # _initialize_model_and_processor executa _load_model_task internamente
        self._initialize_model_and_processor()

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
        self.asr_backend = self.config_manager.get("asr_backend", self.asr_backend)
        self.asr_model_id = self.config_manager.get("asr_model_id", self.asr_model_id)
        logging.info("TranscriptionHandler: Configurações atualizadas.")

    def _initialize_model_and_processor(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._load_model_task()
        except Exception as e:
            error_message = f"Erro na inicialização do backend ASR: {e}"
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
        threading.Thread(target=self._initialize_model_and_processor, daemon=True, name="ModelLoadThread").start()

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
            if make_backend is None:
                raise RuntimeError("make_backend function is not available")

            asr_model_id = self.config_manager.get("asr_model_id")
            asr_backend = self.config_manager.get("asr_backend")
            asr_compute_device = self.config_manager.get("asr_compute_device")
            asr_dtype = self.config_manager.get("asr_dtype")
            asr_ct2_compute_type = self.config_manager.get("asr_ct2_compute_type")
            asr_cache_dir = self.config_manager.get("asr_cache_dir")

            self._asr_backend = make_backend(asr_backend)

            attn_impl = "sdpa"
            try:
                import importlib.util
                if importlib.util.find_spec("flash_attn") is not None:
                    attn_impl = "flash_attn2"
            except Exception:
                pass

            self._asr_backend.load(
                model_id=asr_model_id,
                compute_device=asr_compute_device,
                dtype=asr_dtype,
                ct2_compute_type=asr_ct2_compute_type,
                cache_dir=asr_cache_dir,
                attn_implementation=attn_impl,
            )
            self._asr_backend.warmup()
            self._asr_loaded = True
            self.on_model_ready_callback()
        except Exception as e:
            error_message = f"Falha ao carregar o modelo: {e}"
            logging.error(error_message, exc_info=True)
            self.on_model_error_callback(error_message)

    def transcribe_audio_segment(self, audio_source: str | np.ndarray, agent_mode: bool = False):
        """Envia o áudio (arquivo ou array) para transcrição assíncrona."""
        self.transcription_future = self.transcription_executor.submit(
            self._transcription_task, audio_source, agent_mode
        )

    def _transcription_task(self, audio_source: str | np.ndarray, agent_mode: bool) -> None:
        self.transcription_cancel_event.clear()

        if self.transcription_cancel_event.is_set():
            logging.info("Transcrição interrompida por stop signal antes do início do processamento.")
            return

        text_result = None
        try:

            if not self._asr_loaded or self._asr_backend is None:
                error_message = "Backend de ASR indisponível. Modelo não carregado ou falhou."
                logging.error(error_message)
                self.on_model_error_callback(error_message)
                return

            dynamic_batch_size = self._get_dynamic_batch_size()
            logging.info(f"Iniciando transcrição de segmento com batch_size={dynamic_batch_size}...")

            if isinstance(audio_source, np.ndarray) and audio_source.ndim > 1:
                audio_source = audio_source.flatten()

            # Revalidar chunk em runtime se modo auto (segurança extra)
            if self.chunk_length_mode == "auto":
                try:
                    self.chunk_length_sec = float(self._effective_chunk_length())
                except Exception:
                    pass

            # Métricas detalhadas de tempos: t_pre, t_infer, t_post e t_total_segment
            # Campos técnicos
            try:
                device = f"cuda:{self.gpu_index}" if torch.cuda.is_available() and self.gpu_index >= 0 else "cpu"
            except Exception:
                device = "cpu"
            dtype = "fp16" if (torch.cuda.is_available() and self.gpu_index >= 0) else "fp32"
            try:
                import importlib.util as _spec_util
                attn_impl = "flash_attn2" if _spec_util.find_spec("flash_attn") is not None else "sdpa"
            except Exception:
                attn_impl = "sdpa"

            t_total_start = time.perf_counter()
            # t_pre: placeholder para etapas leves de pré-processamento (flatten já foi aplicado)
            t_pre_start = time.perf_counter()
            t_pre_end = time.perf_counter()
            t_pre_ms = (t_pre_end - t_pre_start) * 1000.0

            # Inferência (t_infer) sob no_grad
            with torch.no_grad():
                t_infer_start = time.perf_counter()
                result = self._asr_backend.transcribe(
                    audio_source,
                    chunk_length_s=self.chunk_length_sec,
                    batch_size=dynamic_batch_size,
                )
                t_infer_end = time.perf_counter()
            t_infer_ms = (t_infer_end - t_infer_start) * 1000.0

            # Expor último batch dinâmico para UI/tooltip
            try:
                self.last_dynamic_batch_size = int(dynamic_batch_size)
            except Exception:
                self.last_dynamic_batch_size = None

            # t_post: montagem/validação do texto
            t_post_start = time.perf_counter()
            # o processamento de result segue abaixo (text_result, etc.)

            if result and "text" in result:
                text_result = result["text"].strip()
                if not text_result:
                    text_result = "[No speech detected]"
                else:
                    logging.info("Transcrição de segmento bem-sucedida.")
            else:
                text_result = "[Transcription failed: Bad format]"
                logging.error(f"Formato de resultado inesperado: {result}")

            # concluir t_post e t_total
            t_post_end = time.perf_counter()
            t_post_ms = (t_post_end - t_post_start) * 1000.0
            t_total_ms = (t_post_end - t_total_start) * 1000.0

            # Logs [METRIC]
            try:
                logging.info(f"[METRIC] stage=t_pre value_ms={t_pre_ms:.2f} device={device} chunk={self.chunk_length_sec} batch={dynamic_batch_size} dtype={dtype} attn={attn_impl}")
                logging.info(f"[METRIC] stage=t_infer value_ms={t_infer_ms:.2f} device={device} chunk={self.chunk_length_sec} batch={dynamic_batch_size} dtype={dtype} attn={attn_impl}")
                logging.info(f"[METRIC] stage=t_post value_ms={t_post_ms:.2f} device={device} chunk={self.chunk_length_sec} batch={dynamic_batch_size} dtype={dtype} attn={attn_impl}")
                logging.info(f"[METRIC] stage=segment_total value_ms={t_total_ms:.2f} device={device} chunk={self.chunk_length_sec} batch={dynamic_batch_size} dtype={dtype} attn={attn_impl}")
            except Exception:
                pass

            # Fim do bloco de processamento principal do segmento

        except Exception as e:
            # Tratamento de OOM e fallback automático (reduz batch, depois chunk) – não recria a pipeline
            try:
                err_txt = str(e).lower()
                is_oom = any(tok in err_txt for tok in ["out of memory", "cuda oom", "cublas", "cudnn", "hip out of memory", "alloc"])
                if is_oom:
                    try:
                        old_bs = int(dynamic_batch_size)
                    except Exception:
                        old_bs = None
                    did_change = False
                    if old_bs and old_bs > 1:
                        new_bs = max(1, old_bs // 2)
                        try:
                            self.last_dynamic_batch_size = new_bs
                        except Exception:
                            pass
                        did_change = True
                        logging.warning(f"OOM detectado. Reduzindo batch_size de {old_bs} para {new_bs} para próximas submissões.")
                        try:
                            logging.info(f"[METRIC] stage=oom_recovery action=reduce_batch from={old_bs} to={new_bs}")
                        except Exception:
                            pass
                    if not did_change:
                        # Reduz chunk_length_sec moderadamente
                        try:
                            old_chunk = float(self.chunk_length_sec)
                        except Exception:
                            old_chunk = 30.0
                        new_chunk = max(10.0, old_chunk * 0.66)
                        if new_chunk < old_chunk:
                            self.chunk_length_sec = new_chunk
                            logging.warning(f"OOM persistente. Reduzindo chunk_length_sec de {old_chunk:.1f}s para {new_chunk:.1f}s para próximas submissões.")
                            try:
                                logging.info(f"[METRIC] stage=oom_recovery action=reduce_chunk from={old_chunk:.1f} to={new_chunk:.1f}")
                            except Exception:
                                pass
                # Continua fluxo normal de erro
            except Exception as _oom_adj_e:
                logging.debug(f"Falha ao ajustar parâmetros após OOM: {_oom_adj_e}")
            logging.error(f"Erro durante a transcrição de segmento: {e}", exc_info=True)
            text_result = f"[Transcription Error: {e}]"

        finally:
            if self.transcription_cancel_event.is_set():
                logging.info("Transcrição interrompida por stop signal. Resultado descartado.")
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

            # empty_cache opcional após segmentos longos (heurística simples) quando em GPU
            try:
                import torch
                from .config_manager import CLEAR_GPU_CACHE_CONFIG_KEY
                enable_clear = bool(self.config_manager.get(CLEAR_GPU_CACHE_CONFIG_KEY))
                is_gpu = torch.cuda.is_available() and getattr(self, "gpu_index", -1) >= 0
                long_audio = float(getattr(self, "chunk_length_sec", 30.0)) >= 45.0
                if enable_clear and is_gpu and long_audio:
                    t_ec_start = time.perf_counter()
                    before_b = torch.cuda.memory_allocated() if hasattr(torch.cuda, "memory_allocated") else 0
                    torch.cuda.empty_cache()
                    after_b = torch.cuda.memory_allocated() if hasattr(torch.cuda, "memory_allocated") else 0
                    t_ec_ms = (time.perf_counter() - t_ec_start) * 1000.0
                    freed_mb = max(0.0, (before_b - after_b) / (1024 ** 2))
                    logging.info(f"[METRIC] stage=empty_cache value_ms={t_ec_ms:.2f} freed_estimate_mb={freed_mb:.1f}")
            except Exception as _ec_e:
                logging.debug(f"Falha ao executar empty_cache opcional: {_ec_e}")

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
                if self.text_correction_enabled:
                    self._get_text_correction_service()
                    was_transcribing_when_started = (
                        self.is_state_transcribing_fn()
                        if self.is_state_transcribing_fn
                        else False
                    )
                    openrouter_prompt = self.config_manager.get(OPENROUTER_PROMPT_CONFIG_KEY)
                    self.correction_thread = threading.Thread(
                        target=self._async_text_correction,
                        args=(text_result, agent_mode, self.gemini_prompt, openrouter_prompt, was_transcribing_when_started),
                        daemon=True,
                        name="TextCorrectionThread",
                    )
                    self.correction_thread.start()
                else:
                    self.on_transcription_result_callback(text_result, text_result)

            # Mantemos a VRAM em cache para acelerar transcrições consecutivas.
            # A limpeza completa ocorre somente no shutdown.
            if torch.cuda.is_available():
                logging.debug(
                    "Cache da GPU preservado para transcrições consecutivas."
                )

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
