import logging
import threading
import concurrent.futures
import torch
from transformers import pipeline

try:
    from optimum.bettertransformer import BetterTransformer  # noqa: F401
    BETTERTRANSFORMER_AVAILABLE = True
except Exception:
    BETTERTRANSFORMER_AVAILABLE = False
from .openrouter_api import (
    OpenRouterAPI,
)  # Assumindo que está na raiz ou em path acessível
import numpy as np  # Necessário para o audio_input

# Importar constantes de configuração
from utils import select_batch_size
from .config_manager import (
    BATCH_SIZE_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY,
    MANUAL_BATCH_SIZE_CONFIG_KEY,  # Novos
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    SERVICE_NONE,
    SERVICE_OPENROUTER,
    SERVICE_GEMINI,
    OPENROUTER_API_KEY_CONFIG_KEY,
    OPENROUTER_MODEL_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    GEMINI_AGENT_PROMPT_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    OPENROUTER_PROMPT_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
    WHISPER_MODEL_ID_CONFIG_KEY,
    USE_TURBO_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,  # Nova constante
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    USE_FLASH_ATTENTION_2_CONFIG_KEY,
    TEXT_CORRECTION_TIMEOUT_CONFIG_KEY,
)

# Mensagem padronizada para falhas na otimização Turbo/Flash Attention 2
OPTIMIZATION_TURBO_FALLBACK_MSG = (
    "Falha ao aplicar otimização 'Turbo Mode' (Flash Attention 2)."
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
        on_optimization_fallback_callback=None,
    ):
        self.config_manager = config_manager
        # Cliente Gemini injetado
        self.gemini_api = gemini_api_client
        self.on_model_ready_callback = on_model_ready_callback
        self.on_model_error_callback = on_model_error_callback
        self.on_optimization_fallback_callback = on_optimization_fallback_callback
        self.on_transcription_result_callback = (
            on_transcription_result_callback  # Para resultado final
        )
        self.on_agent_result_callback = (
            on_agent_result_callback  # Para resultado do agente
        )
        self.on_segment_transcribed_callback = (
            on_segment_transcribed_callback  # Para segmentos em tempo real
        )
        self.is_state_transcribing_fn = is_state_transcribing_fn
        # "state_check_callback" é preservado apenas para retrocompatibilidade;
        # utilize "is_state_transcribing_fn" nas novas implementações.
        self.state_check_callback = is_state_transcribing_fn
        self.correction_in_progress = False
        self.correction_thread = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.transcription_pipeline = None
        self.is_model_loading = False
        # Futura tarefa de transcrição em andamento
        self.transcription_future = None
        # Indica se uma transcrição está em progresso
        self.is_transcribing = False
        # Evento de sinalização para parar tarefas de transcrição
        self._stop_signal_event = threading.Event()
        # Executor dedicado para a tarefa de transcrição em background
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        # Evento para sinalizar cancelamento de transcrição em andamento
        self.transcription_cancel_event = threading.Event()
        self.model_loaded_event = threading.Event()

        # Configurações de modelo e API (carregadas do config_manager)
        self.batch_size = self.config_manager.get(
            BATCH_SIZE_CONFIG_KEY
        )  # Agora é o batch_size padrão para o modo auto
        self.batch_size_mode = self.config_manager.get(
            BATCH_SIZE_MODE_CONFIG_KEY
        )  # Novo
        self.manual_batch_size = self.config_manager.get(
            MANUAL_BATCH_SIZE_CONFIG_KEY
        )  # Novo
        self.gpu_index = self.config_manager.get(GPU_INDEX_CONFIG_KEY)
        self.use_turbo = self.config_manager.get(USE_TURBO_CONFIG_KEY)
        self.batch_size_specified = self.config_manager.get(
            "batch_size_specified"
        )  # Ainda usado para validação
        self.gpu_index_specified = self.config_manager.get(
            "gpu_index_specified"
        )  # Ainda usado para validação

        self.text_correction_enabled = self.config_manager.get(
            TEXT_CORRECTION_ENABLED_CONFIG_KEY
        )
        self.text_correction_service = self.config_manager.get(
            TEXT_CORRECTION_SERVICE_CONFIG_KEY
        )
        self.openrouter_api_key = self.config_manager.get(OPENROUTER_API_KEY_CONFIG_KEY)
        self.openrouter_model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
        self.gemini_api_key = self.config_manager.get(GEMINI_API_KEY_CONFIG_KEY)
        self.gemini_agent_model = self.config_manager.get("gemini_agent_model")
        self.gemini_prompt = self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY)
        self.text_correction_timeout = self.config_manager.get(
            TEXT_CORRECTION_TIMEOUT_CONFIG_KEY, 30
        )
        self.min_transcription_duration = self.config_manager.get(
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY
        )
        self.use_flash_attention_2 = self.config_manager.get("use_flash_attention_2")

        self.openrouter_client = None
        # self.gemini_api é injetado
        self.device_in_use = None  # Nova variável para armazenar o dispositivo em uso

        self._init_api_clients()
        # Removido: self._initialize_model_and_processor() # Chamada para inicializar o modelo e o processador

    def _init_api_clients(self):
        # Lógica de inicialização de OpenRouterAPI e GeminiAPI
        # (movida de WhisperCore._init_openrouter_client e _init_gemini_client)
        # ...
        self.openrouter_client = None
        self.openrouter_api = None
        if (
            self.text_correction_enabled
            and self.text_correction_service == SERVICE_OPENROUTER
            and self.openrouter_api_key
            and OpenRouterAPI
        ):
            try:
                self.openrouter_client = OpenRouterAPI(
                    api_key=self.openrouter_api_key, model_id=self.openrouter_model
                )
                self.openrouter_api = self.openrouter_client
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
        self.use_turbo = self.config_manager.get(USE_TURBO_CONFIG_KEY)
        self.text_correction_enabled = self.config_manager.get(
            TEXT_CORRECTION_ENABLED_CONFIG_KEY
        )
        self.text_correction_service = self.config_manager.get(
            TEXT_CORRECTION_SERVICE_CONFIG_KEY
        )
        self.openrouter_api_key = self.config_manager.get(OPENROUTER_API_KEY_CONFIG_KEY)
        self.openrouter_model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
        self.gemini_api_key = self.config_manager.get(GEMINI_API_KEY_CONFIG_KEY)
        self.gemini_agent_model = self.config_manager.get("gemini_agent_model")
        self.gemini_prompt = self.config_manager.get(GEMINI_PROMPT_CONFIG_KEY)
        self.text_correction_timeout = self.config_manager.get(
            TEXT_CORRECTION_TIMEOUT_CONFIG_KEY, 30
        )
        self.min_transcription_duration = self.config_manager.get(
            MIN_TRANSCRIPTION_DURATION_CONFIG_KEY
        )
        self.use_flash_attention_2 = self.config_manager.get("use_flash_attention_2")
        logging.info("TranscriptionHandler: Configurações atualizadas.")

        if (
            self.text_correction_enabled
            and self.text_correction_service == SERVICE_OPENROUTER
            and self.openrouter_api_key
            and self.openrouter_client is None
            and OpenRouterAPI
        ):
            try:
                self.openrouter_client = OpenRouterAPI(
                    api_key=self.openrouter_api_key, model_id=self.openrouter_model
                )
                self.openrouter_api = self.openrouter_client
                logging.info(
                    "OpenRouter API client initialized via update_config."
                )
            except Exception as e:
                logging.error(f"Error initializing OpenRouter API client: {e}")

    def _get_device_and_dtype(self):
        """Define o dispositivo e o dtype ideais para o modelo."""
        device = (
            f"cuda:{self.gpu_index}"
            if torch.cuda.is_available() and self.gpu_index >= 0
            else "cpu"
        )
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.device_in_use = device
        return device, torch_dtype

    def _initialize_model_and_processor(self):
        """Realiza o carregamento assíncrono do modelo."""
        self._load_model_task()

    def _get_text_correction_service(self):
        if not self.text_correction_enabled:
            return SERVICE_NONE
        if (
            self.text_correction_service == SERVICE_OPENROUTER
            and self.openrouter_client
        ):
            return SERVICE_OPENROUTER
        # Verifica se o cliente Gemini existe e se a chave é válida
        if (
            self.text_correction_service == SERVICE_GEMINI
            and self.gemini_api
            and self.gemini_api.is_valid
        ):
            return SERVICE_GEMINI
        return SERVICE_NONE

    def _async_text_correction(
        self, text: str, is_agent_mode: bool, gemini_prompt: str, openrouter_prompt: str
    ):
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
                logging.info("Correção de texto desativada ou provedor indisponível.")
                self.correction_in_progress = False
                self.on_transcription_result_callback(text, text)
                return

            api_key = self.config_manager.get_api_key(active_provider)

            if not api_key:
                logging.warning(
                    f"Nenhuma chave de API encontrada para o provedor {active_provider}. Pulando correção de texto."
                )
                self.correction_in_progress = False
                self.on_transcription_result_callback(text, text)
                return

            if active_provider == "gemini":
                if not is_agent_mode:
                    prompt = gemini_prompt
                else:
                    logging.info(
                        "Modo Agente ativado. Usando prompt do Agente para o Gemini."
                    )
                    prompt = self.config_manager.get(GEMINI_AGENT_PROMPT_CONFIG_KEY)
                future = self.executor.submit(
                    self.gemini_api.correct_text_async,
                    corrected,
                    prompt,
                    api_key,
                    self.config_manager.get("gemini_model"),
                )
                corrected = future.result(timeout=self.text_correction_timeout)
            elif active_provider == "openrouter":
                if not is_agent_mode:
                    prompt = openrouter_prompt
                else:
                    logging.info(
                        "Modo Agente ativado. Usando prompt do Agente para o OpenRouter."
                    )
                    prompt = self.config_manager.get(OPENROUTER_PROMPT_CONFIG_KEY)

                model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
                future = self.executor.submit(
                    self.openrouter_api.correct_text_async,
                    corrected,
                    prompt,
                    api_key,
                    model,
                )
                corrected = future.result(timeout=self.text_correction_timeout)
            else:
                logging.error(f"Provedor de IA desconhecido: {active_provider}")

        except concurrent.futures.TimeoutError:
            logging.error(
                "Correção de texto excedeu o tempo limite. Retornando texto original."
            )
            if future and not future.done():
                future.cancel()
            corrected = text
        except Exception as exc:
            logging.error(f"Erro ao corrigir texto: {exc}")
            if future and not future.done():
                future.cancel()
            corrected = text  # Ensure original text is used if correction fails
        finally:
            self.correction_in_progress = False
            # O resultado da correção deve ser sempre retornado, independentemente
            # de uma mudança de estado subsequente, para evitar perda de dados do usuário.
            if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
                logging.info(f"Transcrição corrigida: {corrected}")
            self.on_transcription_result_callback(corrected, text)

    def _get_dynamic_batch_size(self) -> int:
        if not torch.cuda.is_available() or self.gpu_index < 0:
            logging.info(
                "GPU não disponível ou não selecionada, usando batch size de CPU (4)."
            )
            return 4

        if self.batch_size_mode == "manual":
            logging.info(
                f"Modo de batch size manual selecionado. Usando valor configurado: {self.manual_batch_size}"
            )
            return self.manual_batch_size

        # Lógica para modo "auto" (dinâmico)
        return select_batch_size(self.gpu_index, fallback=self.batch_size)

    def start_model_loading(self):
        if self.is_model_loading:
            logging.info(
                "TranscriptionHandler: carregamento do modelo já em andamento."
            )
            return
        self.is_model_loading = True
        threading.Thread(
            target=self._initialize_model_and_processor,
            daemon=True,
            name="ModelLoadThread",
        ).start()

    def is_transcription_running(self) -> bool:
        """Indica se existe tarefa de transcrição ainda não concluída."""
        return self.is_transcribing

    def is_text_correction_running(self) -> bool:
        """Indica se há correção de texto em andamento."""
        return self.correction_in_progress

    def stop_transcription(self) -> None:
        """Sinaliza que a transcrição em andamento deve ser cancelada."""
        self.transcription_cancel_event.set()
        if self.is_transcribing and self.transcription_future:
            self.transcription_future.cancel()

    def _load_model_task(self):
        model_id = self.config_manager.get(
            WHISPER_MODEL_ID_CONFIG_KEY, "openai/whisper-large-v3"
        )
        try:
            device, torch_dtype = self._get_device_and_dtype()
            logging.info(
                f"TranscriptionHandler: carregando pipeline para {model_id} no {device}..."
            )
            self.transcription_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                torch_dtype=torch_dtype,
                device=device,
            )
            flash_enabled = self.use_turbo and self.use_flash_attention_2
            if flash_enabled:
                if device.startswith("cuda"):
                    if not BETTERTRANSFORMER_AVAILABLE:
                        warn_msg = (
                            "Pacote 'optimum[bettertransformer]' nao encontrado. Modo Turbo desativado."
                        )
                        logging.warning(warn_msg)
                        if self.on_optimization_fallback_callback:
                            self.on_optimization_fallback_callback(warn_msg)
                    else:
                        logging.info(
                            "Tentando aplicar Flash Attention 2 via BetterTransformer..."
                        )
                        try:
                            if BETTERTRANSFORMER_AVAILABLE:
                                self.transcription_pipeline.model = (
                                    self.transcription_pipeline.model.to_bettertransformer()
                                )
                                logging.info("Flash Attention 2 aplicada com sucesso.")
                        except Exception as exc:
                            warn_msg = f"{OPTIMIZATION_TURBO_FALLBACK_MSG} Motivo: {exc}"
                            logging.warning(warn_msg)
                            if self.on_optimization_fallback_callback:
                                self.on_optimization_fallback_callback(warn_msg)
                else:
                    warn_msg = (
                        f"{OPTIMIZATION_TURBO_FALLBACK_MSG} Motivo: nenhum GPU foi detectado. Desative ou ajuste as configurações."
                    )
                    logging.warning(warn_msg)
                    if self.on_optimization_fallback_callback:
                        self.on_optimization_fallback_callback(warn_msg)
            elif self.use_flash_attention_2 and not self.use_turbo:
                logging.info(
                    "Turbo Mode desativado; ignorando otimização Flash Attention 2."
                )
            if self.on_model_ready_callback:
                self.on_model_ready_callback()
        except Exception as exc:
            logging.error(f"Erro ao carregar o modelo Whisper: {exc}", exc_info=True)
            if self.on_model_error_callback:
                self.on_model_error_callback(str(exc))
        finally:
            self.model_loaded_event.set()
            self.is_model_loading = False

    def transcribe_audio_segment(
        self, audio_input: np.ndarray, agent_mode: bool = False
    ):
        """Envia segmento para transcrição assíncrona."""
        self._stop_signal_event.clear()

        # Espera o modelo carregar antes de submeter a tarefa
        if not self.model_loaded_event.is_set():
            logging.info("Modelo ainda não carregado. Aguardando...")
            self.model_loaded_event.wait()  # Bloqueia até o modelo estar pronto
            logging.info("Modelo carregado. Prosseguindo com a transcrição.")

        self.is_transcribing = True
        self.transcription_future = self.transcription_executor.submit(
            self._transcribe_audio_chunk, audio_input, agent_mode
        )

    def _transcribe_audio_chunk(
        self, audio_input: np.ndarray, agent_mode: bool
    ) -> None:
        if self.transcription_cancel_event.is_set():
            logging.info(
                "Transcrição interrompida por stop signal antes do início do processamento."
            )
            return

        text_result = None
        try:
            if self.transcription_pipeline is None:
                error_message = "Pipeline de transcrição indisponível. Modelo não carregado ou falhou."
                logging.error(error_message)
                if self.on_model_error_callback and not self.model_loaded_event.is_set():
                    self.on_model_error_callback(error_message)
                return
            audio_data = audio_input
            logging.debug(
                f"Transcrevendo áudio de {len(audio_input)/16000:.2f} segundos."
            )
            result = self.transcription_pipeline(audio_data.copy())
            text_result = result["text"].strip()
            logging.info(f"Transcrição recebida: {text_result}")
            if self.on_transcription_result_callback:
                self.on_transcription_result_callback(text_result)
        except Exception as e:
            logging.error(f"Erro durante a transcrição: {e}", exc_info=True)
        finally:
            self.is_transcribing = False

            if text_result and self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY):
                logging.info(f"Transcrição bruta: {text_result}")

            if (
                not text_result
                or text_result == "[No speech detected]"
                or text_result.strip().startswith("[Transcription Error:")
            ):
                logging.warning(
                    f"Segmento processado sem texto significativo ou com erro: {text_result}"
                )
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
                    agent_response = self.gemini_api.get_agent_response(text_result)
                    logging.info(
                        f"Resposta recebida do modo agente: '{agent_response}'"
                    )
                    if (
                        not self.is_state_transcribing_fn
                        or self.is_state_transcribing_fn()
                    ):
                        self.on_agent_result_callback(agent_response)
                    else:
                        logging.warning(
                            "Estado mudou antes do resultado do agente. UI não será atualizada."
                        )
                except Exception as e:
                    logging.error(
                        f"Erro ao processar o comando do agente: {e}", exc_info=True
                    )
                    if (
                        not self.is_state_transcribing_fn
                        or self.is_state_transcribing_fn()
                    ):
                        self.on_agent_result_callback(
                            text_result
                        )  # Falha, retorna o texto original
                    else:
                        logging.warning(
                            "Estado mudou antes do resultado do agente. UI não será atualizada."
                        )
            else:
                if self.text_correction_enabled:
                    openrouter_prompt = self.config_manager.get(
                        OPENROUTER_PROMPT_CONFIG_KEY
                    )
                    self.correction_thread = threading.Thread(
                        target=self._async_text_correction,
                        args=(
                            text_result,
                            agent_mode,
                            self.gemini_prompt,
                            openrouter_prompt,
                        ),
                        daemon=True,
                        name="TextCorrectionThread",
                    )
                    self.correction_thread.start()
                else:
                    self.on_transcription_result_callback(text_result, text_result)

            if torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
                logging.debug("Cache da GPU limpo após tarefa de transcrição.")

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
