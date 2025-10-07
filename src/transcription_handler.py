import concurrent.futures
import importlib
import importlib.util
import logging
import threading
import time
import os
from typing import TYPE_CHECKING, Any

import numpy as np

try:  # pragma: no cover - biblioteca opcional
    from whisper_flash import make_backend  # type: ignore
except Exception:  # pragma: no cover
    from .asr.backends import make_backend  # type: ignore

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
from . import model_manager as model_manager_module
from .logging_utils import current_correlation_id, get_logger, scoped_correlation_id

if TYPE_CHECKING:  # pragma: no cover - hints only
    import torch as torch_type

torch: Any | None
_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is not None:  # pragma: no cover - import side effect
    torch = importlib.import_module("torch")
else:  # pragma: no cover - optional dependency missing
    torch = None


def _torch_cuda_available() -> bool:
    if torch is None:
        return False
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _torch_cuda_device_count() -> int:
    if not _torch_cuda_available():
        return 0
    device_count = getattr(torch.cuda, "device_count", None)
    if device_count is None:
        return 0
    try:
        return int(device_count())
    except Exception:
        return 0


LOGGER = get_logger('whisper_flash_transcriber.transcription', component='TranscriptionHandler')


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
        # Configurações de ASR
        # Inicializar atributos internos sem acionar recarga imediata do backend
        self._asr_backend_name = self.config_manager.get(ASR_BACKEND_CONFIG_KEY)
        self._asr_model_id = self.config_manager.get(ASR_MODEL_ID_CONFIG_KEY)
        self._asr_backend = None

        self.asr_compute_device = self.config_manager.get(ASR_COMPUTE_DEVICE_CONFIG_KEY)
        self.asr_ct2_compute_type = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        self.asr_ct2_cpu_threads = self.config_manager.get(ASR_CT2_CPU_THREADS_CONFIG_KEY)
        self.asr_cache_dir = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)
        self.compute_type_in_use = self.asr_ct2_compute_type
        self.deps_install_dir = self.config_manager.get_deps_install_dir()
        self.hf_home_dir = self.config_manager.get_hf_home_dir()

        self._apply_environment_overrides()

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

    def _apply_environment_overrides(self) -> None:
        overrides = self.config_manager.get_environment_overrides()
        if not overrides:
            return
        for key, value in overrides.items():
            if not value:
                continue
            os.environ[key] = str(value)
        logging.debug(
            "TranscriptionHandler applied environment overrides: %s",
            ", ".join(f"{k}={v}" for k, v in overrides.items()),
        )

    def _build_backend_load_kwargs(
        self,
        backend_name: str,
        *,
        asr_ct2_compute_type,
        asr_cache_dir,
        backend_device,
        backend_device_index,
        asr_ct2_cpu_threads,
    ) -> dict:
        """Constroi os parâmetros de ``load`` para o backend escolhido."""
        if backend_name in {"ct2", "faster-whisper", "ctranslate2"}:
            payload: dict[str, object] = {
                "ct2_compute_type": asr_ct2_compute_type,
                "cache_dir": asr_cache_dir,
            }
            if backend_device:
                payload["device"] = backend_device
            if backend_device_index is not None:
                payload["device_index"] = backend_device_index
            try:
                if asr_ct2_cpu_threads is not None:
                    threads = int(asr_ct2_cpu_threads)
                    if threads > 0:
                        payload["cpu_threads"] = threads
            except (TypeError, ValueError):
                pass
            return payload
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
            "Starting ASR backend reload.",
            extra={
                "event": "asr_reload",
                "details": (
                    f"backend={self._asr_backend_name} model={self._asr_model_id} device={self.asr_compute_device}"
                ),
            },
        )

        def _signal_loading_state() -> None:
            if core_ref is not None and hasattr(core_ref, "notify_model_loading_started"):
                try:
                    core_ref.notify_model_loading_started()
                    return
                except Exception as state_error:
                    logging.debug(
                        "Failed to call notify_model_loading_started during reload: %s",
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
                        "Failed to emit LOADING_MODEL state during reload: %s",
                        state_error,
                        exc_info=True,
                    )

        _signal_loading_state()

        if self._asr_backend is not None:
            try:
                self._asr_backend.unload()
            except Exception as unload_error:
                logging.warning(
                    "Unable to unload existing ASR backend before reload: %s",
                    unload_error,
                )
            finally:
                self._asr_backend = None

        self.pipe = None

        if bool(self.config_manager.get(CLEAR_GPU_CACHE_CONFIG_KEY)) and _torch_cuda_available():
            try:
                assert torch is not None  # narrow type for static checkers
                torch.cuda.empty_cache()
            except Exception as cache_error:
                logging.debug(
                    "Failed to clear CUDA cache before reload: %s",
                    cache_error,
                    exc_info=True,
                )

        try:
            model, processor = self._initialize_model_and_processor()
        except Exception as exc:  # salvaguarda contra erros inesperados
            logging.error(
                "Unexpected error while reloading the ASR backend: %s",
                exc,
                exc_info=True,
                extra={"event": "asr_reload", "status": "error"},
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
                        "Failed to report reload error to state manager: %s",
                        state_error,
                        exc_info=True,
                    )
            if self.on_model_error_callback:
                try:
                    self.on_model_error_callback(str(exc))
                except Exception as callback_error:
                    logging.debug(
                        "Failed to notify reload error callback: %s",
                        callback_error,
                        exc_info=True,
                    )
            return False

        if model is None and processor is None:
            logging.error(
                "ASR backend reload produced no artifacts; check configuration and earlier logs.",
                extra={"event": "asr_reload", "status": "no_artifacts"},
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
                        "Failed to report missing artifacts after reload: %s",
                        state_error,
                        exc_info=True,
                    )
            return False

        logging.info(
            "ASR backend reloaded successfully.",
            extra={
                "event": "asr_reload",
                "status": "complete",
                "details": (
                    f"backend={self.backend_resolved or self._asr_backend_name} device={self.device_in_use or 'unknown'}"
                ),
            },
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

        previous_backend = self._asr_backend_name
        previous_model_id = self._asr_model_id
        previous_device = getattr(self, "asr_compute_device", None)
        previous_ct2_type = getattr(self, "asr_ct2_compute_type", None)
        previous_ct2_threads = getattr(self, "asr_ct2_cpu_threads", None)
        previous_cache_dir = getattr(self, "asr_cache_dir", None)

        backend_value = self.config_manager.get(ASR_BACKEND_CONFIG_KEY)
        model_value = self.config_manager.get(ASR_MODEL_ID_CONFIG_KEY)
        device_value = self.config_manager.get(ASR_COMPUTE_DEVICE_CONFIG_KEY)
        ct2_type_value = self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY)
        ct2_threads_value = self.config_manager.get(ASR_CT2_CPU_THREADS_CONFIG_KEY)
        cache_dir_value = self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY)

        backend_changed = backend_value != previous_backend
        model_changed = model_value != previous_model_id
        device_changed = device_value != previous_device
        ct2_type_changed = ct2_type_value != previous_ct2_type
        ct2_threads_changed = ct2_threads_value != previous_ct2_threads
        cache_dir_changed = cache_dir_value != previous_cache_dir

        new_deps_dir = self.config_manager.get_deps_install_dir()
        new_hf_home = self.config_manager.get_hf_home_dir()

        deps_changed = new_deps_dir != getattr(self, "deps_install_dir", None)
        hf_home_changed = new_hf_home != getattr(self, "hf_home_dir", None)

        reload_needed = (
            backend_changed
            or model_changed
            or device_changed
            or ct2_type_changed
            or ct2_threads_changed
            or cache_dir_changed
            or deps_changed
            or hf_home_changed
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
        self.asr_ct2_compute_type = ct2_type_value
        self.asr_ct2_cpu_threads = ct2_threads_value
        self.asr_cache_dir = cache_dir_value
        self.compute_type_in_use = ct2_type_value
        self.deps_install_dir = new_deps_dir
        self.hf_home_dir = new_hf_home

        if deps_changed or hf_home_changed:
            self._apply_environment_overrides()

        if reload_needed and trigger_reload:
            logging.info(
                "Transcription handler detected critical changes; reloading ASR backend.",
            )
            self.reload_asr()

        if correction_changed:
            self._init_api_clients()

        logging.info(
            "Transcription handler configuration refreshed.",
            extra={"event": "transcription_config_update"},
        )
        return reload_needed

    def _resolve_asr_settings(self):
        """Determina backend, modelo e parâmetros específicos do runtime CT2."""
        backend = model_manager_module.normalize_backend_label(self.asr_backend) or "ctranslate2"
        compute_device = (self.asr_compute_device or "auto").lower()
        model_id = self.asr_model_id or "auto"
        compute_type_raw = (self.asr_ct2_compute_type or "default").lower()

        if compute_device == "auto":
            compute_device = "cuda" if _torch_cuda_available() else "cpu"

        if compute_type_raw in {"auto", "default"}:
            compute_type = "int8_float16" if compute_device.startswith("cuda") else "int8"
        else:
            compute_type = compute_type_raw

        default_gpu_model = "openai/whisper-large-v3-turbo"
        default_cpu_model = "openai/whisper-large-v3-turbo"
        if model_id in ("auto", default_gpu_model, default_cpu_model):
            model_id = default_gpu_model if compute_device.startswith("cuda") else default_cpu_model

        return backend, model_id, compute_device, compute_type

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

    def _emit_transcription_metrics(self, payload: dict[str, object] | None) -> None:
        """Forward transcription metrics to optional collectors in the core instance."""

        if not payload:
            return

        core = getattr(self, "core_instance_ref", None)
        if core is None:
            return

        potential_attributes = (
            "transcription_metrics_callback",
            "transcription_metrics_collector",
            "transcription_metrics_collectors",
            "diagnostic_callbacks",
            "diagnostics_callbacks",
            "telemetry_collectors",
            "transcription_diagnostic_callbacks",
        )
        potential_methods = (
            "dispatch_transcription_metrics",
            "handle_transcription_metrics",
            "emit_transcription_metrics",
            "publish_transcription_metrics",
            "collect_transcription_metrics",
        )

        callbacks: list[object] = []

        for attr in potential_attributes:
            target = getattr(core, attr, None)
            if callable(target):
                callbacks.append(target)
            elif isinstance(target, (list, tuple, set)):
                callbacks.extend(cb for cb in target if callable(cb))

        for method_name in potential_methods:
            method = getattr(core, method_name, None)
            if callable(method):
                callbacks.append(method)

        if not callbacks:
            return

        seen: set[object] = set()
        for callback in callbacks:
            identifier = getattr(callback, "__qualname__", None) or getattr(callback, "__name__", None) or id(callback)
            if identifier in seen:
                continue
            seen.add(identifier)
            try:
                callback(payload)
            except Exception:
                logging.debug(
                    "Failed to propagate transcription metrics via %s.",
                    getattr(callback, "__qualname__", getattr(callback, "__name__", repr(callback))),
                    exc_info=True,
                )

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
            if _torch_cuda_available():
                try:
                    assert torch is not None
                    torch.cuda.empty_cache()
                except Exception:
                    logging.debug(
                        "Failed to clear CUDA cache before model load.",
                        exc_info=True,
                    )
            model, processor = self._load_model_task()
        except Exception as exc:
            error_message = f"Pipeline initialization error: {exc}"
            logging.error(error_message, exc_info=True)
            if self.on_model_error_callback:
                try:
                    self.on_model_error_callback(error_message)
                except Exception:
                    logging.debug(
                        "Failed to notify model initialization error.",
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
                    "ASR backend initialized.",
                    extra={
                        "event": "asr_init",
                        "details": (
                            f"backend={self.backend_resolved or self._asr_backend_name} device={self.device_in_use or 'unknown'}"
                        ),
                    },
                )
                if self.on_model_ready_callback:
                    self.on_model_ready_callback()
                return model, processor

            self._asr_loaded = True
            if self.on_model_ready_callback:
                self.on_model_ready_callback()
            return model, processor
        except Exception as exc:
            error_message = f"Pipeline initialization error: {exc}"
            logging.error(error_message, exc_info=True)
            if self.on_model_error_callback:
                try:
                    self.on_model_error_callback(error_message)
                except Exception:
                    logging.debug(
                        "Failed to propagate pipeline initialization error.",
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
            logging.error(f"Failed to call Gemini API get_correction: {e}")
            return text

    def _process_ai_pipeline(self, transcribed_text: str, is_agent_mode: bool) -> str:
        """Centraliza o fluxo de pós-processamento baseado em IA."""
        if not transcribed_text:
            return transcribed_text

        if is_agent_mode:
            if not self.gemini_api or not getattr(self.gemini_api, "is_valid", False):
                logging.warning(
                    "Agent mode requested but the Gemini client is unavailable.",
                    extra={"event": "agent_mode_correction", "status": "unavailable"},
                )
                return transcribed_text
            try:
                agent_response = self.gemini_api.get_agent_response(transcribed_text)
                return agent_response or transcribed_text
            except Exception as exc:
                logging.error(
                    "Failed to fetch response from Gemini agent: %s",
                    exc,
                    exc_info=True,
                    extra={"event": "agent_mode_correction", "status": "error"},
                )
                return transcribed_text

        if not self.text_correction_enabled:
            return transcribed_text

        active_provider = self._get_text_correction_service()
        if active_provider == SERVICE_NONE:
            logging.info(
                "Text correction disabled or no provider available.",
                extra={"event": "text_correction", "status": "skipped"},
            )
            return transcribed_text

        if active_provider == SERVICE_GEMINI and (
            not self.gemini_api or not getattr(self.gemini_api, "is_valid", False)
        ):
            logging.warning(
                "Gemini client unavailable for text correction.",
                extra={"event": "text_correction", "provider": "gemini", "status": "unavailable"},
            )
            return transcribed_text

        if active_provider == SERVICE_OPENROUTER and not self.openrouter_api:
            logging.warning(
                "OpenRouter client unavailable for text correction.",
                extra={"event": "text_correction", "provider": "openrouter", "status": "unavailable"},
            )
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
                        "No API key configured for the OpenRouter provider. Skipping text correction.",
                        extra={"event": "text_correction", "provider": "openrouter", "status": "no_api_key"},
                    )
                    return transcribed_text

                model = self.config_manager.get(OPENROUTER_MODEL_CONFIG_KEY)
                prompt = self.config_manager.get(OPENROUTER_PROMPT_CONFIG_KEY)
                try:
                    self.openrouter_api.reinitialize_client(api_key=api_key, model_id=model)
                except Exception as exc:
                    logging.error(
                        "Failed to reconfigure the OpenRouter client: %s",
                        exc,
                        exc_info=True,
                        extra={"event": "text_correction", "provider": "openrouter", "status": "reconfigure_failed"},
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
                logging.error(f"Unknown AI provider: {active_provider}")
                return transcribed_text
        except Exception as exc:
            logging.error(
                "Error while processing text with provider %s: %s",
                active_provider,
                exc,
                exc_info=True,
            )
            processed_text = transcribed_text
        finally:
            self.correction_in_progress = False

        if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
            logging.info(
                "Text correction produced a result.",
                extra={"event": "text_correction", "status": "completed", "details": f"chars={len(processed_text)}"},
            )

        return processed_text or transcribed_text

    def _get_dynamic_batch_size(self) -> int:
        device_in_use = (str(self.device_in_use or "").lower())
        if not (_torch_cuda_available() and device_in_use.startswith("cuda")):
            logging.info(
                "GPU unavailable or not selected; using CPU batch size (4).",
                extra={"event": "batch_size", "status": "cpu_fallback"},
            )
            self.last_dynamic_batch_size = 4
            return 4

        if self.batch_size_mode == "manual":
            logging.info(
                f"Manual batch size mode selected. Using configured value: {self.manual_batch_size}",
                extra={"event": "batch_size", "status": "manual"},
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
        assert reason, "Device fallback without documented reason."
        message = (
            f"Configured device '{preferred}' not applied; using '{actual}'. Reason: {reason}."
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
                    "Failed to propagate device fallback warning: %s",
                    callback_error,
                    exc_info=True,
                )

    def start_model_loading(self):
        core = getattr(self, "core_instance_ref", None)
        state_mgr = getattr(core, "state_manager", None) if core is not None else None
        if core is not None and hasattr(core, "notify_model_loading_started"):
            try:
                core.notify_model_loading_started()
            except Exception as exc:
                logging.debug(
                    "Failed to signal model loading start via AppCore: %s",
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
                            "Failed to update state manager to LOADING_MODEL directly.",
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
                    "Failed to update state manager to LOADING_MODEL directly.",
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
        self._apply_environment_overrides()
        self.device_in_use = "cpu"
        self.compute_type_in_use = None
        core_ref = getattr(self, "core_instance_ref", None)
        state_mgr = getattr(core_ref, "state_manager", None) if core_ref is not None else None
        if make_backend is None:
            if state_mgr is not None:
                state_mgr.set_state(sm.STATE_ERROR_MODEL)
            raise RuntimeError(
                "The CTranslate2 backend factory is unavailable. Reinstall the application with CT2 support."
            )

        backend_preference = self.config_manager.get("asr_backend") or "ctranslate2"
        requested_backend_display = (
            backend_preference.strip()
            if isinstance(backend_preference, str)
            else "ctranslate2"
        )
        backend_candidate = (
            model_manager_module.normalize_backend_label(requested_backend_display)
            or "ctranslate2"
        )

        if backend_candidate not in {"ctranslate2"}:
            message = (
                f"Unsupported ASR backend '{requested_backend_display}'. Update the settings to use "
                "the CTranslate2 runtime."
            )
            logging.error(message)
            if state_mgr is not None:
                state_mgr.set_state(sm.STATE_ERROR_MODEL)
            raise RuntimeError(message)

        try:
            self._asr_backend = make_backend(backend_candidate)
        except Exception as backend_error:
            message = (
                "Unable to initialize the CTranslate2 backend. Reinstall the optional dependencies "
                "or run the dependency remediation flow."
            )
            logging.error("%s: %s", message, backend_error)
            if state_mgr is not None:
                state_mgr.set_state(sm.STATE_ERROR_MODEL)
            raise RuntimeError(message) from backend_error

        self.backend_resolved = backend_candidate

        req_backend, req_model_id, req_device, req_compute_type = self._resolve_asr_settings()
        logging.info(
            "Resolved ASR settings: backend=%s, model=%s, device=%s, ct2_compute_type=%s",
            req_backend,
            req_model_id,
            req_device,
            req_compute_type,
        )

        available_cuda = _torch_cuda_available()
        gpu_count = _torch_cuda_device_count() if available_cuda else 0

        requested_gpu_index: int | None = None
        if isinstance(req_device, str):
            lowered_device = req_device.strip().lower()
            if lowered_device.startswith("cuda"):
                _, _, suffix = lowered_device.partition(":")
                if suffix:
                    try:
                        requested_gpu_index = int(suffix)
                    except ValueError:
                        requested_gpu_index = None

        config_gpu_idx_raw = self.config_manager.get(GPU_INDEX_CONFIG_KEY, -1)
        try:
            config_gpu_idx = int(config_gpu_idx_raw)
        except (TypeError, ValueError):
            config_gpu_idx = -1

        self.gpu_index_requested = (
            requested_gpu_index
            if requested_gpu_index is not None
            else (config_gpu_idx if config_gpu_idx >= 0 else None)
        )

        effective_device = "cpu"
        selected_gpu_index: int | None = None

        if req_device == "cpu":
            logging.info("ASR device explicitly set to CPU.")
        elif isinstance(req_device, str) and req_device.startswith("cuda"):
            if not available_cuda or gpu_count == 0:
                reason = (
                    "CUDA not available."
                    if not available_cuda
                    else "No GPUs detected."
                )
                self._emit_device_warning(req_device, "cpu", reason)
            else:
                target_idx = requested_gpu_index
                if target_idx is None or target_idx < 0:
                    if 0 <= config_gpu_idx < gpu_count:
                        target_idx = config_gpu_idx
                    else:
                        if config_gpu_idx not in (-1, 0):
                            logging.warning(
                                "Invalid GPU index %s, falling back to GPU 0.",
                                config_gpu_idx,
                            )
                        target_idx = 0
                if target_idx is not None and target_idx >= gpu_count:
                    self._emit_device_warning(
                        req_device,
                        "cpu",
                        (
                            f"Requested GPU index {target_idx} is out of range for "
                            f"{gpu_count} visible device(s)."
                        ),
                    )
                else:
                    selected_gpu_index = target_idx
                    effective_device = (
                        f"cuda:{selected_gpu_index}" if selected_gpu_index is not None else "cpu"
                    )

        self.device_in_use = effective_device
        self.gpu_index = selected_gpu_index if selected_gpu_index is not None else -1

        logging.info(
            "Effective ASR device: %s (gpu_index=%s)",
            self.device_in_use,
            self.gpu_index,
        )

        backend_device = effective_device
        backend_device_index = selected_gpu_index if selected_gpu_index is not None else None
        if not backend_device.startswith("cuda"):
            backend_device_index = None

        load_kwargs = self._build_backend_load_kwargs(
            backend_name=backend_candidate,
            asr_ct2_compute_type=req_compute_type,
            asr_cache_dir=self.config_manager.get(ASR_CACHE_DIR_CONFIG_KEY),
            backend_device=backend_device,
            backend_device_index=backend_device_index,
            asr_ct2_cpu_threads=self.config_manager.get(ASR_CT2_CPU_THREADS_CONFIG_KEY),
        )
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        if "cache_dir" in load_kwargs and load_kwargs["cache_dir"]:
            load_kwargs["cache_dir"] = str(load_kwargs["cache_dir"])

        if hasattr(self._asr_backend, "model_id"):
            try:
                self._asr_backend.model_id = req_model_id
            except Exception:
                logging.debug("Unable to propagate model_id to backend instance.", exc_info=True)
        if hasattr(self._asr_backend, "device"):
            try:
                self._asr_backend.device = backend_device
            except Exception:
                logging.debug("Unable to propagate device to backend instance.", exc_info=True)
        if hasattr(self._asr_backend, "device_index"):
            try:
                self._asr_backend.device_index = backend_device_index
            except Exception:
                logging.debug("Unable to propagate device_index to backend instance.", exc_info=True)

        self._update_model_log_context(
            backend=backend_candidate,
            model=req_model_id,
            device=effective_device,
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
            backend=backend_candidate,
            model_id=req_model_id,
            device=effective_device,
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
                backend=backend_candidate,
                model_id=req_model_id,
                device=effective_device,
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
            logging.debug("ASR backend warmup failed: %s", warmup_error)

        duration_ms = (time.perf_counter() - load_started_at) * 1000.0
        resolved_device = getattr(self._asr_backend, "device", effective_device)
        resolved_model = getattr(self._asr_backend, "model_id", req_model_id)
        resolved_compute = load_kwargs.get(
            "ct2_compute_type",
            self.config_manager.get(ASR_CT2_COMPUTE_TYPE_CONFIG_KEY),
        )
        self.compute_type_in_use = resolved_compute
        self._update_model_log_context(
            backend=backend_candidate,
            model=resolved_model,
            device=resolved_device,
            compute_type=resolved_compute,
            chunk_length_s=float(self.chunk_length_sec),
            batch_size=self.batch_size,
        )
        self._log_model_event(
            "load_success",
            backend=backend_candidate,
            device=resolved_device,
            compute_type=resolved_compute,
            duration_ms=duration_ms,
            status="warmup_failed" if warmup_failed else "ready",
        )
        self._model_load_started_at = None
        logging.info(
            "Backend '%s' initialized on device %s.",
            self.backend_resolved,
            self.device_in_use,
        )
        return self._asr_backend, None
