import logging
import threading
import concurrent.futures
import time
import numpy as np
import torch

try:  # pragma: no cover - biblioteca opcional
    from whisper_flash import make_backend  # type: ignore
except Exception:  # pragma: no cover
    from .asr.backends import make_backend  # type: ignore

try:  # pragma: no cover - transformers podem não estar instalados em testes
    from transformers import (
        pipeline,
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        BitsAndBytesConfig,
    )
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModelForSpeechSeq2Seq = None  # type: ignore

    class BitsAndBytesConfig:  # type: ignore[py-class-var]
        def __init__(self, *_, **__):
            pass

from .openrouter_api import OpenRouterAPI # Assumindo que está na raiz ou em path acessível
from .audio_handler import AUDIO_SAMPLE_RATE
from pathlib import Path

# Importar constantes de configuração
from .utils import select_batch_size
from . import state_manager as sm
from .config_manager import (
    ASR_BACKEND_CONFIG_KEY,
    ASR_MODEL_ID_CONFIG_KEY,
    ASR_COMPUTE_DEVICE_CONFIG_KEY,
    ASR_DTYPE_CONFIG_KEY,
    ASR_CT2_COMPUTE_TYPE_CONFIG_KEY,
    ASR_CACHE_DIR_CONFIG_KEY,
    MIN_TRANSCRIPTION_DURATION_CONFIG_KEY,
    TEXT_CORRECTION_ENABLED_CONFIG_KEY,
    TEXT_CORRECTION_SERVICE_CONFIG_KEY,
    OPENROUTER_API_KEY_CONFIG_KEY,
    OPENROUTER_MODEL_CONFIG_KEY,
    OPENROUTER_TIMEOUT_CONFIG_KEY,
    GEMINI_API_KEY_CONFIG_KEY,
    GEMINI_PROMPT_CONFIG_KEY,
    SERVICE_NONE,
    SERVICE_OPENROUTER,
    SERVICE_GEMINI,
    BATCH_SIZE_CONFIG_KEY,
    BATCH_SIZE_MODE_CONFIG_KEY,
    MANUAL_BATCH_SIZE_CONFIG_KEY,
    GPU_INDEX_CONFIG_KEY,
    CHUNK_LENGTH_SEC_CONFIG_KEY,
    ASR_CT2_CPU_THREADS_CONFIG_KEY,
    CLEAR_GPU_CACHE_CONFIG_KEY,
    OPENROUTER_PROMPT_CONFIG_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    DISPLAY_TRANSCRIPTS_KEY,
)

LOGGER = logging.getLogger('whisper_flash_transcriber.transcription')


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
        # Contexto reutilizado para logs padronizados sobre o backend/modelo
        self._model_log_context: dict[str, object] = {}
        self._model_load_started_at: float | None = None

        # Configurações de modelo e API (carregadas do config_manager)
        self.batch_size = self.config_manager.get(BATCH_SIZE_CONFIG_KEY) # Agora é o batch_size padrão para o modo auto
        self.batch_size_mode = self.config_manager.get(BATCH_SIZE_MODE_CONFIG_KEY) # Novo
        self.manual_batch_size = self.config_manager.get(MANUAL_BATCH_SIZE_CONFIG_KEY) # Novo
        self.gpu_index = self.config_manager.get(GPU_INDEX_CONFIG_KEY)
        self.gpu_index_requested = self.gpu_index
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
        # Inicializar atributos internos sem acionar recarga imediata do backend
        self._asr_backend_name = self.config_manager.get(ASR_BACKEND_CONFIG_KEY)
        self._asr_model_id = self.config_manager.get(ASR_MODEL_ID_CONFIG_KEY)
        self._asr_backend = None

        self.asr_compute_device = self.config_manager.get(ASR_COMPUTE_DEVICE_CONFIG_KEY)
        self.asr_dtype = self.config_manager.get(ASR_DTYPE_CONFIG_KEY)
        self.asr_ct2_compute_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        self.asr_ct2_cpu_threads = self.config_manager.get(ASR_CT2_CPU_THREADS_CONFIG_KEY)
        self.asr_cache_dir = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)

        self.openrouter_client = None
        # self.gemini_client é injetado
        self.device_in_use = None # Nova variável para armazenar o dispositivo em uso
        self.last_dynamic_batch_size = None

        self._init_api_clients()

    def _report_adjustment(self, message: str, *, level: int = logging.WARNING) -> None:
        """Loga um aviso e envia a mensagem para a UI quando disponível."""
        if not message:
            return
        core = getattr(self, "core_instance_ref", None)
        if core and hasattr(core, "report_runtime_notice"):
            core.report_runtime_notice(message, level=level)
        else:
            logging.log(level, message)

    def _build_backend_load_kwargs(
        self,
        backend_name: str,
        *,
        asr_dtype,
        asr_ct2_compute_type,
        asr_cache_dir,
        transformers_device,
    ) -> dict:
        """Constroi os parâmetros de ``load`` para o backend escolhido."""
        if backend_name in {"transformers", "whisper"}:
            attn_impl = "sdpa"
            try:
                import importlib.util

                if importlib.util.find_spec("flash_attn") is not None:
                    attn_impl = "flash_attention_2"
            except Exception:
                pass
            return {
                "device": transformers_device,
                "dtype": asr_dtype,
                "cache_dir": asr_cache_dir,
                "attn_implementation": attn_impl,
            }
        if backend_name in {"ct2", "faster-whisper", "ctranslate2"}:
            return {
                "ct2_compute_type": asr_ct2_compute_type,
                "cache_dir": asr_cache_dir,
            }
        return {"cache_dir": asr_cache_dir}

    def _init_api_clients(self):
        # Lógica de inicialização de OpenRouterAPI e GeminiAPI
        # (movida de WhisperCore._init_openrouter_client e _init_gemini_client)
        # ...
        self.openrouter_client = None
        self.openrouter_api = None
        if self.text_correction_enabled and self.text_correction_service == SERVICE_OPENROUTER and self.openrouter_api_key and OpenRouterAPI:
            try:
                openrouter_timeout = self.config_manager.get_timeout(
                    OPENROUTER_TIMEOUT_CONFIG_KEY,
                    OpenRouterAPI.DEFAULT_TIMEOUT,
                )
                self.openrouter_client = OpenRouterAPI(
                    api_key=self.openrouter_api_key,
                    model_id=self.openrouter_model,
                    request_timeout=openrouter_timeout,
                )
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

    def reload_asr(self) -> bool:
        """Recarrega o backend de ASR e o modelo associado."""
        core_ref = getattr(self, "core_instance_ref", None)
        state_mgr = getattr(core_ref, "state_manager", None) if core_ref is not None else None

        logging.info(
            "Iniciando recarga do backend ASR (backend=%s, model=%s, device=%s).",
            self._asr_backend_name,
            self._asr_model_id,
            self.asr_compute_device,
        )

        def _signal_loading_state() -> None:
            if core_ref is not None and hasattr(core_ref, "notify_model_loading_started"):
                try:
                    core_ref.notify_model_loading_started()
                    return
                except Exception as state_error:
                    logging.debug(
                        "Falha ao acionar notify_model_loading_started durante reload: %s",
                        state_error,
                        exc_info=True,
                    )
            if state_mgr is not None:
                try:
                    state_mgr.set_state(
                        sm.STATE_LOADING_MODEL,
                        source="transcription_handler",
                        details="Reloading ASR backend",
                    )
                except Exception as state_error:
                    logging.debug(
                        "Falha ao sinalizar estado LOADING_MODEL durante reload: %s",
                        state_error,
                        exc_info=True,
                    )

        _signal_loading_state()

        if self._asr_backend is not None:
            try:
                self._asr_backend.unload()
            except Exception as unload_error:
                logging.warning(
                    "Falha ao descarregar backend ASR atual antes do reload: %s",
                    unload_error,
                )
            finally:
                self._asr_backend = None

        self.pipe = None

        if bool(self.config_manager.get(CLEAR_GPU_CACHE_CONFIG_KEY)) and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as cache_error:
                logging.debug(
                    "Falha ao limpar cache CUDA antes do reload: %s",
                    cache_error,
                    exc_info=True,
                )

        try:
            model, processor = self._initialize_model_and_processor()
        except Exception as exc:  # salvaguarda contra erros inesperados
            logging.error(
                "Erro inesperado durante o reload do backend ASR: %s",
                exc,
                exc_info=True,
            )
            if state_mgr is not None:
                try:
                    state_mgr.set_state(
                        sm.StateEvent.MODEL_LOADING_FAILED,
                        source="transcription_handler",
                        details=str(exc),
                    )
                except Exception as state_error:
                    logging.debug(
                        "Falha ao sinalizar erro inesperado de recarga: %s",
                        state_error,
                        exc_info=True,
                    )
            if self.on_model_error_callback:
                try:
                    self.on_model_error_callback(str(exc))
                except Exception as callback_error:
                    logging.debug(
                        "Falha ao notificar erro inesperado de recarga: %s",
                        callback_error,
                        exc_info=True,
                    )
            return False

        if model is None and processor is None:
            logging.error(
                "Recarga do backend ASR não produziu artefatos; verifique configurações e logs anteriores.",
            )
            if state_mgr is not None:
                try:
                    state_mgr.set_state(
                        sm.StateEvent.MODEL_LOADING_FAILED,
                        source="transcription_handler",
                        details="No artifacts returned during reload",
                    )
                except Exception as state_error:
                    logging.debug(
                        "Falha ao sinalizar ausência de artefatos no reload: %s",
                        state_error,
                        exc_info=True,
                    )
            return False

        logging.info(
            "Backend ASR recarregado com sucesso (backend=%s, device=%s).",
            self.backend_resolved or self._asr_backend_name,
            self.device_in_use or "unknown",
        )
        return True

    def update_config(self, *, trigger_reload: bool = True) -> bool:
        """Atualiza as configurações do handler a partir do ``config_manager``.

        Parameters
        ----------
        trigger_reload:
            Quando ``True`` (padrão), dispara o recarregamento imediato do
            backend caso algum parâmetro crítico tenha sido alterado.

        Returns
        -------
        bool
            Indica se parâmetros que exigem recarregamento foram modificados.
        """
        previous_text_correction_enabled = getattr(self, "text_correction_enabled", False)
        previous_text_correction_service = getattr(self, "text_correction_service", SERVICE_NONE)
        previous_openrouter_key = getattr(self, "openrouter_api_key", "")
        previous_openrouter_model = getattr(self, "openrouter_model", "")

        self.batch_size = self.config_manager.get(BATCH_SIZE_CONFIG_KEY)
        self.batch_size_mode = self.config_manager.get(BATCH_SIZE_MODE_CONFIG_KEY)
        self.manual_batch_size = self.config_manager.get(MANUAL_BATCH_SIZE_CONFIG_KEY)
        self.gpu_index = self.config_manager.get(GPU_INDEX_CONFIG_KEY)
        self.gpu_index_requested = self.gpu_index
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

        previous_backend = self._asr_backend_name
        previous_model_id = self._asr_model_id
        previous_device = getattr(self, "asr_compute_device", None)
        previous_dtype = getattr(self, "asr_dtype", None)
        previous_ct2_type = getattr(self, "asr_ct2_compute_type", None)
        previous_ct2_threads = getattr(self, "asr_ct2_cpu_threads", None)
        previous_cache_dir = getattr(self, "asr_cache_dir", None)

        backend_value = self.config_manager.get(ASR_BACKEND_CONFIG_KEY)
        model_value = self.config_manager.get(ASR_MODEL_ID_CONFIG_KEY)
        device_value = self.config_manager.get(ASR_COMPUTE_DEVICE_CONFIG_KEY)
        dtype_value = self.config_manager.get(ASR_DTYPE_CONFIG_KEY)
        ct2_type_value = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        ct2_threads_value = self.config_manager.get(ASR_CT2_CPU_THREADS_CONFIG_KEY)
        cache_dir_value = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)

        backend_changed = backend_value != previous_backend
        model_changed = model_value != previous_model_id
        device_changed = device_value != previous_device
        dtype_changed = dtype_value != previous_dtype
        ct2_type_changed = ct2_type_value != previous_ct2_type
        ct2_threads_changed = ct2_threads_value != previous_ct2_threads
        cache_dir_changed = cache_dir_value != previous_cache_dir

        reload_needed = (
            backend_changed
            or model_changed
            or device_changed
            or dtype_changed
            or ct2_type_changed
            or ct2_threads_changed
            or cache_dir_changed
        )

        correction_changed = (
            previous_text_correction_enabled != self.text_correction_enabled
            or previous_text_correction_service != self.text_correction_service
            or previous_openrouter_key != self.openrouter_api_key
            or previous_openrouter_model != self.openrouter_model
        )

        # Atualiza internamente sem acionar recarga automática; o caller decide
        # quando reconstruir o backend.
        self._asr_backend_name = backend_value
        self._asr_model_id = model_value
        self.asr_compute_device = device_value
        self.asr_dtype = dtype_value
        self.asr_ct2_compute_type = ct2_type_value
        self.asr_ct2_cpu_threads = ct2_threads_value
        self.asr_cache_dir = cache_dir_value

        if reload_needed and trigger_reload:
            logging.info(
                "TranscriptionHandler: parâmetros críticos alterados; recarregando backend ASR.",
            )
            self.reload_asr()

        if correction_changed:
            self._init_api_clients()

        logging.info("TranscriptionHandler: Configurações atualizadas.")
        return reload_needed

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
        default_cpu_model = "openai/whisper-large-v3-turbo"
        if model_id in ("auto", default_gpu_model, default_cpu_model):
            model_id = default_gpu_model if compute_device.startswith("cuda") else default_cpu_model

        return backend, model_id, compute_device, dtype

    def _update_model_log_context(self, **entries: object) -> None:
        filtered = {k: v for k, v in entries.items() if v not in (None, "")}
        if filtered:
            self._model_log_context.update(filtered)

    @staticmethod
    def _format_log_value(value: object) -> str:
        if isinstance(value, float):
            magnitude = abs(value)
            if magnitude == 0:
                return "0.00"
            return f"{value:.2f}" if magnitude >= 0.01 else f"{value:.4f}"
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _log_model_event(self, event: str, *, level: int = logging.INFO, **fields: object) -> None:
        context: dict[str, object] = {}
        context.update(self._model_log_context)
        context.update({k: v for k, v in fields.items() if v not in (None, "")})

        backend = context.get("backend") or self.backend_resolved or self.asr_backend
        if backend:
            context["backend"] = backend
        model_id = (
            context.get("model")
            or context.get("model_id")
            or getattr(self._asr_backend, "model_id", None)
            or self.asr_model_id
        )
        if model_id:
            context["model"] = model_id
        context.pop("model_id", None)

        message_order = [
            "event",
            "backend",
            "model",
            "device",
            "dtype",
            "compute_type",
            "chunk_length_s",
            "batch_size",
            "size",
            "duration_ms",
            "agent_mode",
            "status",
            "error",
        ]

        parts: list[str] = []
        used_keys: set[str] = set()
        context["event"] = event
        for key in message_order:
            if key in context:
                parts.append(f"{key}={self._format_log_value(context[key])}")
                used_keys.add(key)
        for key in sorted(context.keys()):
            if key not in used_keys:
                parts.append(f"{key}={self._format_log_value(context[key])}")

        logging.log(level, "[ASR] " + " ".join(parts))

    def _format_audio_source(self, audio_source: str | np.ndarray | bytes | bytearray | list[float] | None) -> str:
        if audio_source is None:
            return "none"
        try:
            if isinstance(audio_source, np.ndarray):
                samples = int(audio_source.size)
                if samples and AUDIO_SAMPLE_RATE:
                    duration = samples / float(AUDIO_SAMPLE_RATE)
                    return f"array_samples={samples} duration_s={duration:.2f}"
                return f"array_samples={samples}"
            if isinstance(audio_source, (bytes, bytearray)):
                return f"bytes={len(audio_source)}"
            if isinstance(audio_source, list):
                return f"list_samples={len(audio_source)}"
            if isinstance(audio_source, str):
                path = Path(audio_source)
                if path.exists():
                    return f"file_bytes={path.stat().st_size}"
                return "file"
        except Exception:
            pass
        return "unknown"

    def _initialize_model_and_processor(self):
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    logging.debug(
                        "Falha ao limpar cache CUDA antes do carregamento do modelo.",
                        exc_info=True,
                    )
            model, processor = self._load_model_task()
        except Exception as exc:
            error_message = f"Erro na inicialização da pipeline: {exc}"
            logging.error(error_message, exc_info=True)
            if self.on_model_error_callback:
                try:
                    self.on_model_error_callback(error_message)
                except Exception:
                    logging.debug(
                        "Falha ao notificar erro na inicialização do modelo.",
                        exc_info=True,
                    )
            return None, None

        if model is None and processor is None:
            logging.debug(
                "Model loader returned no artifacts; pipeline initialization skipped."
            )
            return None, None

        try:
            if processor is None:
                self.pipe = getattr(model, "pipe", None) or model
                self._asr_loaded = True
                logging.info(
                    "ASR backend '%s' inicializado (device=%s).",
                    self.backend_resolved or self._asr_backend_name,
                    self.device_in_use or "unknown",
                )
                if self.on_model_ready_callback:
                    self.on_model_ready_callback()
                return model, processor

            device = (
                f"cuda:{self.gpu_index}"
                if self.gpu_index is not None
                and self.gpu_index >= 0
                and torch.cuda.is_available()
                else "cpu"
            )

            try:
                if hasattr(model, "eval"):
                    model.eval()
                    logging.info("[METRIC] stage=model_eval_applied value_ms=0")
                    training_flag = getattr(model, "training", None)
                    if training_flag is not None:
                        logging.debug("Model.training=%s (esperado False)", training_flag)
            except Exception as eval_error:
                logging.warning("Falha ao aplicar model.eval(): %s", eval_error)

            if (
                self.enable_torch_compile
                and hasattr(torch, "compile")
                and device.startswith("cuda")
            ):
                try:
                    model = torch.compile(model)  # type: ignore[attr-defined]
                    logging.info("torch.compile aplicado ao modelo (experimental).")
                except Exception as compile_error:
                    logging.warning(
                        "Falha ao aplicar torch.compile: %s. Seguindo sem compile.",
                        compile_error,
                        exc_info=True,
                    )
            else:
                logging.info(
                    "torch.compile desativado ou indisponível; seguindo sem compile."
                )

            if self.chunk_length_mode == "auto":
                try:
                    effective_chunk = float(self._effective_chunk_length())
                    if effective_chunk != self.chunk_length_sec:
                        logging.info(
                            "Chunk length ajustado automaticamente: %.1fs -> %.1fs",
                            self.chunk_length_sec,
                            effective_chunk,
                        )
                    self.chunk_length_sec = effective_chunk
                except Exception as chunk_error:
                    logging.warning(
                        "Falha ao calcular chunk_length auto: %s. Mantendo valor atual %.1fs.",
                        chunk_error,
                        self.chunk_length_sec,
                    )

            generate_kwargs_init = {"task": "transcribe", "language": None}
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                chunk_length_s=self.chunk_length_sec,
                batch_size=self.batch_size,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                generate_kwargs=generate_kwargs_init,
            )
            logging.info(
                "Pipeline de transcrição inicializada com sucesso (device=%s).",
                device,
            )

            try:
                import numpy as _np

                warmup_dur = max(0.1, min(0.25, float(self.chunk_length_sec) * 0.01))
                sr = AUDIO_SAMPLE_RATE
                n = int(sr * warmup_dur)
                t = _np.linspace(0, warmup_dur, n, False, dtype=_np.float32)
                tone = (_np.sin(2 * _np.pi * 440.0 * t)).astype(_np.float32)
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = self.pipe(
                        tone,
                        chunk_length_s=self.chunk_length_sec,
                        batch_size=max(1, int(self.batch_size)),
                        return_timestamps=False,
                        generate_kwargs={"task": "transcribe", "language": None},
                    )
                t1 = time.perf_counter()
                logging.info(
                    "[METRIC] stage=warmup_infer value_ms=%.2f device=%s chunk=%.2f batch=%s",
                    (t1 - t0) * 1000,
                    device,
                    self.chunk_length_sec,
                    self.batch_size,
                )
            except Exception as warmup_error:
                logging.warning("Warmup da pipeline falhou: %s", warmup_error)

            self._asr_loaded = True
            if self.on_model_ready_callback:
                self.on_model_ready_callback()
            return model, processor
        except Exception as exc:
            error_message = f"Erro na inicialização da pipeline: {exc}"
            logging.error(error_message, exc_info=True)
            if self.on_model_error_callback:
                try:
                    self.on_model_error_callback(error_message)
                except Exception:
                    logging.debug(
                        "Falha ao notificar erro de pipeline.",
                        exc_info=True,
                    )
            return None, None

    def _get_text_correction_service(self):
        if not self.text_correction_enabled:
            return SERVICE_NONE
        if self.text_correction_service == SERVICE_OPENROUTER and self.openrouter_client:
            return SERVICE_OPENROUTER
        # Verifica se o cliente Gemini existe E se a chave é válida
        if self.text_correction_service == SERVICE_GEMINI and self.gemini_client and self.gemini_client.is_valid:
            return SERVICE_GEMINI
        return SERVICE_NONE

    def _correct_text_with_openrouter(self, text):
        if not self.openrouter_client or not text:
            return text
        try:
            return self.openrouter_client.correct_text(text)
        except Exception as e:
            logging.error(f"Error correcting text with OpenRouter API: {e}")
            return text

    def _correct_text_with_gemini(self, text: str) -> str:
        """Chama o novo método de correção da API Gemini."""
        if not self.gemini_client or not text or not self.gemini_client.is_valid:
            return text
        try:
            return self.gemini_client.get_correction(text)
        except Exception as e:
            logging.error(f"Erro ao chamar get_correction da API Gemini: {e}")
            return text

    def _process_ai_pipeline(self, transcribed_text: str, is_agent_mode: bool) -> str:
        """Centraliza o fluxo de pós-processamento baseado em IA."""
        if not transcribed_text:
            return transcribed_text

        if is_agent_mode:
            if not self.gemini_api or not getattr(self.gemini_api, "is_valid", False):
                logging.warning("Modo agente ativado, mas o cliente Gemini está indisponível.")
                return transcribed_text
            try:
                agent_response = self.gemini_api.get_agent_response(transcribed_text)
                return agent_response or transcribed_text
            except Exception as exc:
                logging.error(
                    "Erro ao obter resposta do agente Gemini: %s",
                    exc,
                    exc_info=True,
                )
                return transcribed_text

        if not self.text_correction_enabled:
            return transcribed_text

        active_provider = self._get_text_correction_service()
        if active_provider == SERVICE_NONE:
            logging.info("Correção de texto desativada ou provedor indisponível.")
            return transcribed_text

        if active_provider == SERVICE_GEMINI and (
            not self.gemini_api or not getattr(self.gemini_api, "is_valid", False)
        ):
            logging.warning("Cliente Gemini indisponível para correção de texto.")
            return transcribed_text

        if active_provider == SERVICE_OPENROUTER and not self.openrouter_api:
            logging.warning("Cliente OpenRouter indisponível para correção de texto.")
            return transcribed_text

        processed_text = transcribed_text
        self.correction_in_progress = True
        try:
            if active_provider == SERVICE_GEMINI:
                processed_text = self.gemini_api.get_correction(transcribed_text) or transcribed_text
            elif active_provider == SERVICE_OPENROUTER:
                api_key = self.config_manager.get_api_key(SERVICE_OPENROUTER)
                if not api_key:
                    logging.warning(
                        "Nenhuma chave de API encontrada para o provedor OpenRouter. Pulando correção de texto."
                    )
                    return transcribed_text

                model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
                prompt = self.config_manager.get(OPENROUTER_PROMPT_CONFIG_KEY)
                try:
                    self.openrouter_api.reinitialize_client(api_key=api_key, model_id=model)
                except Exception as exc:
                    logging.error(
                        "Erro ao reconfigurar o cliente OpenRouter: %s",
                        exc,
                        exc_info=True,
                    )
                if prompt:
                    processed_text = self.openrouter_api.correct_text_async(
                        transcribed_text,
                        prompt,
                        api_key,
                        model,
                    )
                else:
                    processed_text = self.openrouter_api.correct_text(transcribed_text)
            else:
                logging.error(f"Provedor de IA desconhecido: {active_provider}")
                return transcribed_text
        except Exception as exc:
            logging.error(
                "Erro ao processar texto com o provedor %s: %s",
                active_provider,
                exc,
                exc_info=True,
            )
            processed_text = transcribed_text
        finally:
            self.correction_in_progress = False

        if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
            logging.info(f"Transcrição corrigida: {processed_text}")

        return processed_text or transcribed_text

    def _get_dynamic_batch_size(self) -> int:
        device_in_use = (str(self.device_in_use or "").lower())
        if not (torch.cuda.is_available() and device_in_use.startswith("cuda")):
            logging.info("GPU não disponível ou não selecionada, usando batch size de CPU (4).")
            self.last_dynamic_batch_size = 4
            return 4

        if self.batch_size_mode == "manual":
            logging.info(
                f"Modo de batch size manual selecionado. Usando valor configurado: {self.manual_batch_size}"
            )
            self.last_dynamic_batch_size = self.manual_batch_size
            return self.manual_batch_size

        # Lógica para modo "auto" (dinâmico)
        resolved_index = self.gpu_index if isinstance(self.gpu_index, int) else 0
        value = select_batch_size(
            resolved_index,
            fallback=self.batch_size,
            chunk_length_sec=self.chunk_length_sec
        )
        self.last_dynamic_batch_size = value
        return value

    def _emit_device_warning(self, preferred: str, actual: str, reason: str, *, level: str = "warning") -> None:
        """Registra e propaga avisos de fallback de dispositivo."""
        assert reason, "Fallback de dispositivo sem motivo documentado."
        message = (
            f"Configuração de dispositivo '{preferred}' não aplicada; utilizando '{actual}'. Motivo: {reason}."
        )
        log_level = logging.WARNING if level == "warning" else logging.INFO
        logging.log(log_level, message)
        core = getattr(self, "core_instance_ref", None)
        if core and hasattr(core, "emit_state_warning"):
            try:
                core.emit_state_warning(
                    "DEVICE_FALLBACK",
                    {"preferred": preferred, "actual": actual, "reason": reason},
                    message=message,
                    level=level,
                )
            except Exception as callback_error:
                logging.error(
                    "Falha ao propagar aviso de fallback de dispositivo: %s",
                    callback_error,
                    exc_info=True,
                )

    def _resolve_effective_dtype(self, configured_dtype: str | None) -> str | None:
        """Determina o dtype efetivo considerando o dispositivo em uso."""
        dtype = (configured_dtype or "auto")
        dtype_lower = dtype.lower() if isinstance(dtype, str) else "auto"
        device_lower = str(self.device_in_use or "").lower()

        if device_lower.startswith("cuda"):
            if dtype_lower == "fp16":
                return "float16"
            return dtype_lower

        # CPU ou dispositivo desconhecido: garantir float32 para evitar falhas.
        if dtype_lower not in {"float32", "auto"}:
            logging.info(
                "Ajustando dtype para float32 porque o dispositivo ativo é %s (dtype configurado: %s).",
                device_lower or "cpu",
                dtype_lower,
            )
        return "float32"

    def start_model_loading(self):
        core = getattr(self, "core_instance_ref", None)
        state_mgr = getattr(core, "state_manager", None) if core is not None else None
        if core is not None and hasattr(core, "notify_model_loading_started"):
            try:
                core.notify_model_loading_started()
            except Exception as exc:
                logging.debug(
                    "Falha ao notificar início de carregamento via AppCore: %s",
                    exc,
                    exc_info=True,
                )
                if state_mgr is not None:
                    try:
                        state_mgr.set_state(
                            sm.STATE_LOADING_MODEL,
                            source="transcription_handler",
                            details="Model loading initiated",
                        )
                    except Exception:
                        logging.debug(
                            "Falha ao atualizar estado para LOADING_MODEL diretamente.",
                            exc_info=True,
                        )
        elif state_mgr is not None:
            try:
                state_mgr.set_state(
                    sm.STATE_LOADING_MODEL,
                    source="transcription_handler",
                    details="Model loading initiated",
                )
            except Exception:
                logging.debug(
                    "Falha ao atualizar estado para LOADING_MODEL diretamente.",
                    exc_info=True,
                )
        threading.Thread(
            target=self._initialize_model_and_processor,
            daemon=True,
            name="ModelLoadThread",
        ).start()

    def is_transcription_running(self) -> bool:
        """Indica se existe tarefa de transcrição ainda não concluída."""
        return (
            self.transcription_future is not None
            and not self.transcription_future.done()
        )

    def is_text_correction_running(self) -> bool:
        """Indica se há correção de texto em andamento."""
        return self.correction_in_progress

    def is_model_ready(self) -> bool:
        """Indica se há um backend de ASR pronto para receber áudio."""
        if self._asr_backend is not None:
            backend_model = getattr(self._asr_backend, "model", None)
            if backend_model is None:
                return False
            return True
        return self.pipe is not None

    def stop_transcription(self) -> None:
        """Sinaliza que a transcrição em andamento deve ser cancelada."""
        self.transcription_cancel_event.set()

    def _load_model_task(self):
        try:
            self.device_in_use = "cpu"
            if make_backend is not None:
                backend_preference = self.config_manager.get("asr_backend") or "transformers"
                requested_backend_display = (
                    backend_preference.strip()
                    if isinstance(backend_preference, str)
                    else "transformers"
                )
                backend_candidate = (
                    requested_backend_display.lower()
                    if requested_backend_display
                    else "transformers"
                )
                if not backend_candidate:
                    backend_candidate = "transformers"

                backend_name = backend_candidate
                try:
                    self._asr_backend = make_backend(backend_name)
                except Exception as backend_error:
                    if backend_name != "transformers":
                        logging.warning(
                            "Falha ao instanciar backend '%s': %s. Aplicando fallback 'transformers'.",
                            backend_name,
                            backend_error,
                        )
                        backend_name = "transformers"
                        self._asr_backend = make_backend(backend_name)
                    else:
                        raise
                self.backend_resolved = backend_name

                req_backend, req_model_id, req_device, req_dtype = self._resolve_asr_settings()
                logging.info(
                    "Resolved ASR settings: backend=%s, model=%s, device=%s, dtype=%s",
                    req_backend,
                    req_model_id,
                    req_device,
                    req_dtype,
                )

                available_cuda = torch.cuda.is_available()
                gpu_count = torch.cuda.device_count() if available_cuda else 0

                effective_device = "cpu"
                transformers_device: int | str = -1
                selected_gpu_index: int | None = None

                if req_device == "cpu":
                    effective_device = "cpu"
                    logging.info("ASR device explicitly set to CPU.")
                elif isinstance(req_device, str) and req_device.startswith("cuda"):
                    if not available_cuda:
                        self._emit_device_warning(req_device, "cpu", "CUDA not available.")
                        effective_device = "cpu"
                    elif gpu_count == 0:
                        self._emit_device_warning(req_device, "cpu", "No GPUs detected.")
                        effective_device = "cpu"
                    else:
                        config_gpu_idx = self.config_manager.get(GPU_INDEX_CONFIG_KEY, -1)
                        if 0 <= config_gpu_idx < gpu_count:
                            target_idx = config_gpu_idx
                        else:
                            target_idx = 0
                            if config_gpu_idx != -1:
                                logging.warning(
                                    "Invalid GPU index %s, falling back to GPU 0.",
                                    config_gpu_idx,
                                )
                        effective_device = f"cuda:{target_idx}"
                        transformers_device = f"cuda:{target_idx}"
                        selected_gpu_index = target_idx

                self.device_in_use = effective_device
                self.gpu_index = selected_gpu_index if selected_gpu_index is not None else -1

                logging.info(
                    "Effective ASR device: %s (transformers_device=%s, gpu_index=%s)",
                    self.device_in_use,
                    transformers_device,
                    self.gpu_index,
                )

                effective_dtype = self._resolve_effective_dtype(req_dtype)
                load_kwargs = self._build_backend_load_kwargs(
                    backend_name=backend_name,
                    asr_dtype=effective_dtype,
                    asr_ct2_compute_type=self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
                    asr_cache_dir=self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY),
                    transformers_device=transformers_device,
                )
                load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
                if "cache_dir" in load_kwargs and load_kwargs["cache_dir"]:
                    load_kwargs["cache_dir"] = str(load_kwargs["cache_dir"])

                self._update_model_log_context(
                    backend=backend_name,
                    model=req_model_id,
                    device=effective_device,
                    dtype=load_kwargs.get("dtype", req_dtype),
                    compute_type=load_kwargs.get(
                        "ct2_compute_type",
                        self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
                    ),
                    chunk_length_s=float(self.chunk_length_sec),
                    batch_size=self.batch_size,
                )
                load_started_at = time.perf_counter()
                self._model_load_started_at = load_started_at
                self._log_model_event(
                    "load_start",
                    backend=backend_name,
                    model_id=req_model_id,
                    device=effective_device,
                    dtype=load_kwargs.get("dtype", req_dtype),
                    compute_type=load_kwargs.get(
                        "ct2_compute_type",
                        self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
                    ),
                    chunk_length_s=float(self.chunk_length_sec),
                    batch_size=self.batch_size,
                )
                load_kwargs["model_id"] = req_model_id
                try:
                    self._asr_backend.load(**load_kwargs)
                except Exception as load_error:
                    duration_ms = (time.perf_counter() - load_started_at) * 1000.0
                    self._log_model_event(
                        "load_failure",
                        level=logging.ERROR,
                        backend=backend_name,
                        model_id=req_model_id,
                        device=effective_device,
                        dtype=load_kwargs.get("dtype", req_dtype),
                        compute_type=load_kwargs.get(
                            "ct2_compute_type",
                            self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
                        ),
                        chunk_length_s=float(self.chunk_length_sec),
                        batch_size=self.batch_size,
                        duration_ms=duration_ms,
                        error=str(load_error),
                    )
                    self._model_load_started_at = None
                    raise

                warmup_failed = None
                try:
                    self._asr_backend.warmup()
                except Exception as warmup_error:
                    warmup_failed = warmup_error
                    logging.debug("Falha no warmup do backend ASR: %s", warmup_error)

                duration_ms = (time.perf_counter() - load_started_at) * 1000.0
                resolved_device = getattr(self._asr_backend, "device", effective_device)
                resolved_model = getattr(self._asr_backend, "model_id", req_model_id)
                resolved_dtype = load_kwargs.get("dtype", req_dtype)
                resolved_compute = load_kwargs.get(
                    "ct2_compute_type",
                    self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
                )
                self._update_model_log_context(
                    backend=backend_name,
                    model=resolved_model,
                    device=resolved_device,
                    dtype=resolved_dtype,
                    compute_type=resolved_compute,
                    chunk_length_s=float(self.chunk_length_sec),
                    batch_size=self.batch_size,
                )
                self._log_model_event(
                    "load_success",
                    backend=backend_name,
                    device=resolved_device,
                    dtype=resolved_dtype,
                    compute_type=resolved_compute,
                    duration_ms=duration_ms,
                    status="warmup_failed" if warmup_failed else "ready",
                )
                self._model_load_started_at = None
                logging.info(
                    "Backend '%s' inicializado no dispositivo %s.",
                    self.backend_resolved,
                    self.device_in_use,
                )
                return self._asr_backend, None

            self._asr_backend = None
            model_id = self.asr_model_id
            backend = self.asr_backend
            compute_device = self.asr_compute_device
            model_path = Path(self.asr_cache_dir) / backend / model_id
            if not (model_path.is_dir() and any(model_path.iterdir())):
                raise FileNotFoundError(
                    f"Modelo '{model_id}' não encontrado. Instale-o nas configurações."
                )
            self._update_model_log_context(
                backend=backend,
                model=model_id,
                device=compute_device,
                dtype=self.asr_dtype,
                compute_type=self.asr_ct2_compute_type,
                chunk_length_s=float(self.chunk_length_sec),
                batch_size=self.batch_size,
            )
            load_started_at = time.perf_counter()
            self._model_load_started_at = load_started_at
            self._log_model_event(
                "load_start",
                backend=backend,
                model_id=model_id,
                device=compute_device,
                dtype=self.asr_dtype,
                compute_type=self.asr_ct2_compute_type,
                chunk_length_s=float(self.chunk_length_sec),
                batch_size=self.batch_size,
            )

            if AutoProcessor is None or AutoModelForSpeechSeq2Seq is None:
                raise RuntimeError("Transformers não estão disponíveis neste ambiente.")

            logging.info("Carregando processador de %s...", model_id)
            processor = AutoProcessor.from_pretrained(str(model_path))

            if torch.cuda.is_available():
                if self.gpu_index == -1:
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
                        logging.info(
                            "Auto-seleção de GPU (maior VRAM total): %s (%s)",
                            self.gpu_index,
                            torch.cuda.get_device_name(self.gpu_index),
                        )
                    else:
                        logging.info("Nenhuma GPU disponível, usando CPU.")
                        self.gpu_index = -1
                        level = (
                            "warning"
                            if str(compute_device or "auto").lower() == "cuda"
                            else "info"
                        )
                        self._emit_device_warning(
                            str(compute_device or "auto"),
                            "cpu",
                            "Nenhuma GPU disponível para carregamento.",
                            level=level,
                        )

            device = (
                f"cuda:{self.gpu_index}"
                if self.gpu_index >= 0 and torch.cuda.is_available()
                else "cpu"
            )
            self.device_in_use = device
            torch_dtype_local = torch.float16 if device.startswith("cuda") else torch.float32
            resolved_device = device
            resolved_dtype_label = str(torch_dtype_local).replace("torch.", "")
            self._update_model_log_context(device=resolved_device, dtype=resolved_dtype_label)

            logging.info("Dispositivo de carregamento do modelo definido explicitamente como: %s", device)

            try:
                if (
                    compute_device == "cuda"
                    and torch.cuda.is_available()
                    and self.gpu_index == -1
                ):
                    best_idx = None
                    best_free = -1
                    for i in range(torch.cuda.device_count()):
                        try:
                            free_b, _ = torch.cuda.mem_get_info(torch.device(f"cuda:{i}"))
                            if free_b > best_free:
                                best_free = free_b
                                best_idx = i
                        except Exception as _e:
                            logging.debug(
                                "Falha ao consultar mem_get_info para GPU %s: %s",
                                i,
                                _e,
                            )
                    if best_idx is not None:
                        self.gpu_index = best_idx
                        free_gb = best_free / (1024 ** 3) if best_free > 0 else 0.0
                        total_gb = torch.cuda.get_device_properties(self.gpu_index).total_memory / (1024 ** 3)
                        logging.info(
                            "[METRIC] stage=gpu_autoselect gpu=%s free_gb=%.2f total_gb=%.2f",
                            self.gpu_index,
                            free_gb,
                            total_gb,
                        )
            except Exception as _gpu_sel_e:
                logging.warning("Falha ao escolher GPU por memória livre: %s", _gpu_sel_e)

            quant_config = None
            if compute_device == "cuda" and torch.cuda.is_available() and self.gpu_index >= 0:
                quant_config = BitsAndBytesConfig(load_in_8bit=True)

            attn_impl = "sdpa"
            try:
                import importlib.util

                if importlib.util.find_spec("flash_attn") is not None:
                    attn_impl = "flash_attention_2"
            except Exception:
                pass

            model_kwargs = {
                "torch_dtype": torch_dtype_local,
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
                "device_map": {"": device},
                "attn_implementation": attn_impl,
            }
            if quant_config is not None:
                model_kwargs["quantization_config"] = quant_config

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                cache_dir=str(model_path),
                **model_kwargs,
            )

            duration_ms = (time.perf_counter() - load_started_at) * 1000.0
            self._update_model_log_context(device=resolved_device, dtype=resolved_dtype_label)
            self._log_model_event(
                "load_success",
                backend=backend,
                device=resolved_device,
                dtype=resolved_dtype_label,
                compute_type=self.asr_ct2_compute_type,
                duration_ms=duration_ms,
                status="legacy_loaded",
            )
            self._model_load_started_at = None
            return model, processor
        except OSError:
            error_message = "Diretório de cache inválido. Verifique as configurações."
            if self._model_load_started_at is not None:
                duration_ms = (time.perf_counter() - self._model_load_started_at) * 1000.0
                self._log_model_event(
                    "load_failure",
                    level=logging.ERROR,
                    duration_ms=duration_ms,
                    error=error_message,
                )
                self._model_load_started_at = None
            logging.error(error_message, exc_info=True)
            try:
                if self.on_model_error_callback:
                    self.on_model_error_callback(error_message)
            except Exception:
                logging.debug(
                    "Falha ao notificar erro de diretório inválido.",
                    exc_info=True,
                )
            return None, None
        except Exception as e:
            error_message = f"Falha ao carregar o modelo: {e}"
            if self._model_load_started_at is not None:
                duration_ms = (time.perf_counter() - self._model_load_started_at) * 1000.0
                self._log_model_event(
                    "load_failure",
                    level=logging.ERROR,
                    duration_ms=duration_ms,
                    error=str(e),
                )
                self._model_load_started_at = None
            logging.error(error_message, exc_info=True)
            try:
                if self.on_model_error_callback:
                    self.on_model_error_callback(error_message)
            except Exception:
                logging.debug("Falha ao notificar erro de carregamento.", exc_info=True)
            return None, None

    def transcribe_audio_segment(self, audio_source: str | np.ndarray, agent_mode: bool = False):
        """Envia o áudio (arquivo ou array) para transcrição assíncrona."""
        if not self.is_model_ready():
            logging.error("Transcription pipeline is not available. Model not loaded or failed to load.")
            self.on_model_error_callback("Pipeline de transcrição indisponível. O modelo não foi carregado ou falhou.")
            return
        self.transcription_future = self.transcription_executor.submit(
            self._transcription_task, audio_source, agent_mode
        )

    def _transcription_task(self, audio_source: str | np.ndarray, agent_mode: bool) -> None:
        self.transcription_cancel_event.clear()
        text_result: str | None = None

        if self.transcription_cancel_event.is_set():
            self._log_model_event(
                "transcription_cancelled",
                status="pre_start",
                agent_mode=agent_mode,
            )
            logging.info("Transcrição interrompida por stop signal antes do início do processamento.")
            return

        dynamic_batch_size = None
        size_descriptor = self._format_audio_source(audio_source)
        if self._asr_backend is not None:
            try:
                dynamic_batch_size = self._get_dynamic_batch_size()
            except Exception as batch_error:
                self._log_model_event(
                    "transcription_failure",
                    level=logging.ERROR,
                    batch_size=dynamic_batch_size,
                    chunk_length_s=float(self.chunk_length_sec),
                    size=size_descriptor,
                    agent_mode=agent_mode,
                    error=f"dynamic_batch:{batch_error}",
                )
                raise

            self._update_model_log_context(
                chunk_length_s=float(self.chunk_length_sec),
                batch_size=dynamic_batch_size,
            )
            start_ts = time.perf_counter()
            self._log_model_event(
                "transcription_start",
                batch_size=dynamic_batch_size,
                chunk_length_s=float(self.chunk_length_sec),
                size=size_descriptor,
                agent_mode=agent_mode,
            )
            try:
                result = self._asr_backend.transcribe(
                    audio_source,
                    chunk_length_s=float(self.chunk_length_sec),
                    batch_size=dynamic_batch_size,
                )
            except Exception as transcribe_error:
                duration_ms = (time.perf_counter() - start_ts) * 1000.0
                self._log_model_event(
                    "transcription_failure",
                    level=logging.ERROR,
                    batch_size=dynamic_batch_size,
                    chunk_length_s=float(self.chunk_length_sec),
                    size=size_descriptor,
                    agent_mode=agent_mode,
                    duration_ms=duration_ms,
                    error=str(transcribe_error),
                )
                logging.error(f"Erro durante a transcrição via backend unificado: {transcribe_error}", exc_info=True)
                return
            else:
                duration_ms = (time.perf_counter() - start_ts) * 1000.0
                text_result = result.get("text", "").strip() or "[No speech detected]"
                self._log_model_event(
                    "transcription_success",
                    batch_size=dynamic_batch_size,
                    chunk_length_s=float(self.chunk_length_sec),
                    size=size_descriptor,
                    agent_mode=agent_mode,
                    duration_ms=duration_ms,
                )
        else:
            # Legacy pipeline
            # Garantir que dynamic_batch_size esteja definido mesmo quando o backend
            # for CTranslate2, evitando UnboundLocalError no bloco de exceção.
            dynamic_batch_size = None
            legacy_started_at: float | None = None
            logged_failure = False
            try:
                if self.backend_resolved == "ctranslate2":
                    self._update_model_log_context(
                        chunk_length_s=float(self.chunk_length_sec),
                    )
                    legacy_started_at = time.perf_counter()
                    self._log_model_event(
                        "transcription_start",
                        chunk_length_s=float(self.chunk_length_sec),
                        size=size_descriptor,
                        agent_mode=agent_mode,
                    )
                    segments, _ = self.pipe.transcribe(audio_source, language=None)
                    text_result = "".join(seg.text for seg in segments).strip()
                    if not text_result:
                        text_result = "[No speech detected]"
                    duration_ms = (time.perf_counter() - legacy_started_at) * 1000.0
                    self._log_model_event(
                        "transcription_success",
                        chunk_length_s=float(self.chunk_length_sec),
                        size=size_descriptor,
                        agent_mode=agent_mode,
                        duration_ms=duration_ms,
                    )
                    logging.info("Transcrição via CTranslate2 concluída.")
                else:
                    if self.pipe is None:
                        error_message = "Pipeline de transcrição indisponível. Modelo não carregado ou falhou."
                        self._log_model_event(
                            "transcription_failure",
                            level=logging.ERROR,
                            chunk_length_s=float(self.chunk_length_sec),
                            size=size_descriptor,
                            agent_mode=agent_mode,
                            error="pipeline_unavailable",
                        )
                        logging.error(error_message)
                        self.on_model_error_callback(error_message)
                        return

                    try:
                        dynamic_batch_size = self._get_dynamic_batch_size()
                    except Exception as batch_error:
                        logged_failure = True
                        self._log_model_event(
                            "transcription_failure",
                            level=logging.ERROR,
                            chunk_length_s=float(self.chunk_length_sec),
                            size=size_descriptor,
                            agent_mode=agent_mode,
                            error=f"dynamic_batch:{batch_error}",
                        )
                        raise

                    self._update_model_log_context(
                        chunk_length_s=float(self.chunk_length_sec),
                        batch_size=dynamic_batch_size,
                    )
                    legacy_started_at = time.perf_counter()
                    self._log_model_event(
                        "transcription_start",
                        batch_size=dynamic_batch_size,
                        chunk_length_s=float(self.chunk_length_sec),
                        size=size_descriptor,
                        agent_mode=agent_mode,
                    )
                    logging.info(f"Iniciando transcrição de segmento com batch_size={dynamic_batch_size}...")

                    generate_kwargs = {
                        "task": "transcribe",
                        "language": None
                    }
                    if isinstance(audio_source, np.ndarray) and audio_source.ndim > 1:
                        audio_source = audio_source.flatten()

                    if self.chunk_length_mode == "auto":
                        try:
                            self.chunk_length_sec = float(self._effective_chunk_length())
                        except Exception:
                            pass

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

                    t_total_start = legacy_started_at if legacy_started_at is not None else time.perf_counter()
                    t_pre_start = time.perf_counter()
                    t_pre_end = time.perf_counter()
                    t_pre_ms = (t_pre_end - t_pre_start) * 1000.0

                    with torch.no_grad():
                        t_infer_start = time.perf_counter()
                        result = self.pipe(
                            audio_source,
                            chunk_length_s=self.chunk_length_sec,
                            batch_size=dynamic_batch_size,
                            return_timestamps=False,
                            generate_kwargs=generate_kwargs
                        )
                        t_infer_end = time.perf_counter()
                    t_infer_ms = (t_infer_end - t_infer_start) * 1000.0

                    try:
                        self.last_dynamic_batch_size = int(dynamic_batch_size)
                    except Exception:
                        self.last_dynamic_batch_size = None

                    t_post_start = time.perf_counter()

                    if result and "text" in result:
                        text_result = result["text"].strip()
                        if not text_result:
                            text_result = "[No speech detected]"
                        else:
                            logging.info("Transcrição de segmento bem-sucedida.")
                    else:
                        text_result = "[Transcription failed: Bad format]"
                        logging.error(f"Formato de resultado inesperado: {result}")

                    t_post_end = time.perf_counter()
                    t_post_ms = (t_post_end - t_post_start) * 1000.0
                    t_total_ms = (t_post_end - t_total_start) * 1000.0

                    self._log_model_event(
                        "transcription_success",
                        batch_size=dynamic_batch_size,
                        chunk_length_s=float(self.chunk_length_sec),
                        size=size_descriptor,
                        agent_mode=agent_mode,
                        duration_ms=t_total_ms,
                    )

                    try:
                        logging.info(
                            f"[METRIC] stage=t_pre value_ms={t_pre_ms:.2f} device={device} "
                            f"chunk={self.chunk_length_sec} batch={dynamic_batch_size} "
                            f"dtype={dtype} attn={attn_impl}"
                        )
                        logging.info(
                            f"[METRIC] stage=t_infer value_ms={t_infer_ms:.2f} device={device} "
                            f"chunk={self.chunk_length_sec} batch={dynamic_batch_size} "
                            f"dtype={dtype} attn={attn_impl}"
                        )
                        logging.info(
                            f"[METRIC] stage=t_post value_ms={t_post_ms:.2f} device={device} "
                            f"chunk={self.chunk_length_sec} batch={dynamic_batch_size} "
                            f"dtype={dtype} attn={attn_impl}"
                        )
                        logging.info(
                            f"[METRIC] stage=segment_total value_ms={t_total_ms:.2f} device={device} "
                            f"chunk={self.chunk_length_sec} batch={dynamic_batch_size} "
                            f"dtype={dtype} attn={attn_impl}"
                        )
                    except Exception:
                        pass

                    # Fim do bloco de processamento principal do segmento

            except Exception as e:
                # Tratamento de OOM e fallback automático (reduz batch, depois chunk) – não recria a pipeline
                try:
                    err_txt = str(e).lower()
                    is_oom = any(
                        tok in err_txt
                        for tok in [
                            "out of memory",
                            "cuda oom",
                            "cublas",
                            "cudnn",
                            "hip out of memory",
                            "alloc",
                        ]
                    )
                    if is_oom:
                        logging.warning(
                            "Erro de OOM detectado durante a transcrição. Iniciando rotina de recuperação automática."
                        )
                        self._apply_oom_recovery(dynamic_batch_size)
                    # Continua fluxo normal de erro
                except Exception as _oom_adj_e:
                    logging.debug(f"Falha ao ajustar parâmetros após OOM: {_oom_adj_e}")

                if not logged_failure:
                    if legacy_started_at is not None:
                        duration_ms = (time.perf_counter() - legacy_started_at) * 1000.0
                        self._log_model_event(
                            "transcription_failure",
                            level=logging.ERROR,
                            batch_size=dynamic_batch_size,
                            chunk_length_s=float(self.chunk_length_sec),
                            size=size_descriptor,
                            agent_mode=agent_mode,
                            duration_ms=duration_ms,
                            error=str(e),
                        )
                    else:
                        self._log_model_event(
                            "transcription_failure",
                            level=logging.ERROR,
                            batch_size=dynamic_batch_size,
                            chunk_length_s=float(self.chunk_length_sec),
                            size=size_descriptor,
                            agent_mode=agent_mode,
                            error=str(e),
                        )

                logging.error(f"Erro durante a transcrição de segmento: {e}", exc_info=True)
                text_result = f"[Transcription Error: {e}]"

        if self.transcription_cancel_event.is_set():
            self._log_model_event(
                "transcription_cancelled",
                status="inference_cancelled",
                agent_mode=agent_mode,
            )
            logging.info("Transcrição interrompida por stop signal. Resultado descartado.")
            self.transcription_cancel_event.clear()
            return

        if text_result and self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY):
            logging.info(f"Transcrição bruta: {text_result}")

        if (
            not text_result
            or text_result == "[No speech detected]"
            or text_result.strip().startswith("[Transcription Error:")
        ):
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
                if self.on_transcription_result_callback:
                    self.on_transcription_result_callback(text_result, text_result)
            elif not agent_mode and text_result:
                logging.warning(
                    "Estado mudou antes do resultado de transcrição. UI não será atualizada."
                )
            return

        try:
            enable_clear = bool(self.config_manager.get(CLEAR_GPU_CACHE_CONFIG_KEY))
            is_gpu = torch.cuda.is_available() and getattr(self, "gpu_index", -1) >= 0
            long_audio = float(getattr(self, "chunk_length_sec", 30.0)) >= 45.0
            if enable_clear and is_gpu and long_audio:
                t_ec_start = time.perf_counter()
                before_b = (
                    torch.cuda.memory_allocated()
                    if hasattr(torch.cuda, "memory_allocated")
                    else 0
                )
                torch.cuda.empty_cache()
                after_b = (
                    torch.cuda.memory_allocated()
                    if hasattr(torch.cuda, "memory_allocated")
                    else 0
                )
                t_ec_ms = (time.perf_counter() - t_ec_start) * 1000.0
                freed_mb = max(0.0, (before_b - after_b) / (1024 ** 2))
                logging.info(
                    f"[METRIC] stage=empty_cache value_ms={t_ec_ms:.2f} freed_estimate_mb={freed_mb:.1f}"
                )
        except Exception as _ec_e:
            logging.debug(f"Falha ao executar empty_cache opcional: {_ec_e}")

        final_text = self._process_ai_pipeline(text_result, agent_mode)

        if agent_mode:
            if not self.on_agent_result_callback:
                logging.debug("Callback de resultado do agente não configurado.")
                return
            if not self.is_state_transcribing_fn or self.is_state_transcribing_fn():
                self.on_agent_result_callback(final_text)
            else:
                logging.warning(
                    "Estado mudou antes do resultado do agente. UI não será atualizada."
                )
        else:
            if not self.on_transcription_result_callback:
                logging.debug("Callback de resultado de transcrição não configurado.")
            elif not self.is_state_transcribing_fn or self.is_state_transcribing_fn():
                self.on_transcription_result_callback(final_text, text_result)
            else:
                logging.warning(
                    "Estado mudou antes do resultado de transcrição. UI não será atualizada."
                )

        if torch.cuda.is_available():
            logging.debug(
                "Cache da GPU preservado para transcrições consecutivas."
            )

    def _apply_oom_recovery(self, current_batch_size: int | None) -> bool:
        """Ajusta parâmetros internos após detectar falta de memória (OOM).

        Args:
            current_batch_size: Valor positivo que representa o batch utilizado
                no momento da falha. Informe ``None`` para reaproveitar a última
                configuração bem-sucedida.

        Returns:
            ``True`` quando algum ajuste temporário foi realizado para manter a
            execução atual; ``False`` caso nenhum ajuste adicional esteja
            disponível.
        """

        def _report(message: str) -> None:
            core = getattr(self, "core_instance_ref", None)
            if core and hasattr(core, "report_runtime_notice"):
                try:
                    core.report_runtime_notice(message, level=logging.WARNING)
                    return
                except Exception as exc:
                    logging.debug("Falha ao notificar ajuste de OOM na UI: %s", exc)
            logging.warning(message)

        old_batch_size: int | None = None
        try:
            if current_batch_size is not None:
                old_batch_size = int(current_batch_size)
        except Exception:
            old_batch_size = None

        if old_batch_size is None:
            try:
                if self.last_dynamic_batch_size is not None:
                    old_batch_size = int(self.last_dynamic_batch_size)
            except Exception:
                old_batch_size = None

        if old_batch_size is not None and old_batch_size > 1:
            new_batch_size = max(1, old_batch_size // 2)
            if new_batch_size < old_batch_size:
                if self.batch_size_mode == "manual":
                    self.manual_batch_size = new_batch_size
                else:
                    self.batch_size = new_batch_size
                self.last_dynamic_batch_size = new_batch_size
                message = (
                    "OOM detectado. Reduzindo batch_size de "
                    f"{old_batch_size} para {new_batch_size} apenas na sessão atual."
                )
                _report(message)
                try:
                    logging.info(
                        "[METRIC] stage=oom_recovery action=reduce_batch mode=%s from=%s to=%s",
                        self.batch_size_mode,
                        old_batch_size,
                        new_batch_size,
                    )
                except Exception:
                    pass
                return True

        # Se não conseguimos reduzir batch_size, ajusta chunk_length_sec
        try:
            old_chunk = float(self.chunk_length_sec)
        except Exception:
            old_chunk = 30.0

        new_chunk = max(10.0, old_chunk * 0.66)
        if new_chunk < old_chunk:
            self.chunk_length_sec = new_chunk
            message = (
                "OOM persistente. Reduzindo chunk_length_sec de "
                f"{old_chunk:.1f}s para {new_chunk:.1f}s apenas na sessão atual."
            )
            _report(message)
            try:
                logging.info(
                    "[METRIC] stage=oom_recovery action=reduce_chunk from=%.1f to=%.1f",
                    old_chunk,
                    new_chunk,
                )
            except Exception:
                pass
            return True

        _report(
            "OOM persistente. Batch_size e chunk_length_sec já estão nos limites mínimos para esta sessão."
        )
        return False

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

            self.transcription_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            logging.error(f"Erro ao encerrar o executor de transcrição: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.debug(
                    "Cache da GPU liberado no encerramento do aplicativo."
                )
