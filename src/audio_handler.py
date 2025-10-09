from __future__ import annotations

import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import sounddevice as sd
import soundfile as sf

# Observação: em ambientes WSL, a biblioteca sounddevice depende de servidores PulseAudio;
# mantenha esta limitação em mente ao depurar gravações.
from . import state_manager as sm
from .utils.memory import get_available_memory_mb, get_total_memory_mb

from . import state_manager as sm
from .vad_manager import VADManager
from .config_manager import (
    RECORDINGS_DIR_CONFIG_KEY,
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    STORAGE_ROOT_DIR_CONFIG_KEY,
    VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
    VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
)
from .logging_utils import (
    StructuredMessage,
    get_logger,
    join_thread_with_timeout,
    log_context,
    log_operation,
    operation_context,
    scoped_correlation_id,
)

LOGGER = get_logger("whisper_flash_transcriber.audio", component="AudioHandler")


class _AudioLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter que inclui contexto de thread e armazenamento."""

    def process(self, msg, kwargs):
        if not isinstance(kwargs, dict):
            kwargs = dict(kwargs)

        handler = self.extra.get('handler')
        storage_mode = 'memory' if getattr(handler, 'in_memory_mode', False) else 'disk'
        session_id = getattr(handler, "_session_id", None)
        thread_name = threading.current_thread().name

        detail_overrides = kwargs.pop('details', None)
        merged_details: dict[str, Any]
        if isinstance(detail_overrides, Mapping):
            merged_details = dict(detail_overrides)
        else:
            merged_details = {}

        merged_details.setdefault('storage', storage_mode)
        merged_details.setdefault('thread', thread_name)
        if session_id:
            merged_details.setdefault('session', session_id)
        kwargs['details'] = merged_details

        return msg, kwargs


AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1


class AudioHandler:
    """Manage audio recording in temporary files or in-memory buffers."""

    def __init__(
        self,
        config_manager,
        state_manager,
        on_audio_segment_ready_callback,
    ):
        self.config_manager = config_manager
        self.state_manager = state_manager
        self.on_audio_segment_ready_callback = on_audio_segment_ready_callback

        # Initialize attributes that will be configured in `update_config`
        self.record_storage_mode = "auto"
        self.record_storage_limit = 0
        self.in_memory_mode = False
        self.max_memory_seconds_mode = "auto"
        self.max_memory_seconds = 30
        self.current_max_memory_seconds = 30
        self._memory_limit_samples = int(AUDIO_SAMPLE_RATE * self.current_max_memory_seconds)
        self.min_free_ram_mb = 1000
        self.sound_enabled = True
        self.sound_frequency = 400
        self.sound_duration = 0.3
        self.sound_volume = 0.5
        self.min_record_duration = 0.5
        self.use_vad = False
        self.vad_threshold = 0.5
        self.vad_silence_duration = 1.0
        self.vad_pre_speech_padding_ms = 0.0
        self.vad_post_speech_padding_ms = 0.0
        self.vad_manager = None

        self.is_recording = False
        self.start_time = None
        self.audio_stream = None
        self.stream_started = False
        self._stop_event = threading.Event()
        self._record_thread = None
        self.sound_lock = threading.RLock()
        self.storage_lock = threading.Lock()


        self.temp_file_path: str | None = None
        self._sf_writer: sf.SoundFile | None = None
        self._audio_frames: list[np.ndarray] = []
        self._sample_count = 0
        self._memory_samples = 0
        self.recordings_dir = str(Path.cwd())

        self.storage_root_dir: Path | None = None
        self.recordings_dir: Path | None = None
        self._session_id: str | None = None
        self._last_start_failure: dict[str, Any] | None = None

        # Dedicated queue and thread for audio processing
        self.audio_queue = queue.Queue()
        # Mantém o total de workers responsáveis por consumir a fila.
        self._processing_workers = 1
        self._processing_thread = threading.Thread(
            target=self._process_audio_queue,
            daemon=True,
            name="AudioProcessThread",
        )
        self._processing_thread.start()

        self._logger = LOGGER.bind(handler_id=f"audio-{id(self):x}")
        self.models_storage_dir = (
            self.config_manager.get_models_storage_dir()
            if hasattr(self.config_manager, "get_models_storage_dir")
            else None
        )

        # Buffer pool para reutilizar arrays dos frames; reduz churn do GC,
        # mas mantém alguns buffers vivos até atingir o limite configurado.
        self._frame_pool: dict[tuple[int, ...], deque[np.ndarray]] = defaultdict(deque)
        self._frame_pool_bucket_limit = 8
        self._frame_pool_total_limit = 64
        self._frame_pool_total = 0
        self._frame_pool_lock = threading.Lock()

        self.stream_blocksize = int(AUDIO_SAMPLE_RATE / 10)  # ~100ms buffers
        self._overflow_log_window = 5.0  # seconds
        self._last_overflow_sample: tuple[float, int] | None = None
        self.update_config()  # Load the initial configuration

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    @property
    def _log(self):
        """Return a contextual logger enriched with runtime metadata."""

        storage_mode = "memory" if self.in_memory_mode else "disk"
        thread_name = threading.current_thread().name
        op_id = self._current_operation_id
        if op_id:
            return self._logger.bind(
                storage=storage_mode,
                thread=thread_name,
                operation_id=op_id,
            )
        return self._logger.bind(storage=storage_mode, thread=thread_name)

    @property
    def last_start_failure(self) -> dict[str, Any] | None:
        """Return structured information about the last start failure, if any."""

        return self._last_start_failure

    def _resolve_default_input_device(self) -> tuple[str, int | None, str | None]:
        """Return the default input device label, index and lookup error (if any)."""

        device_name = "dispositivo padrão"
        device_index: int | None = None
        lookup_error: str | None = None

        try:
            default_input = sd.query_devices(None, "input")
        except Exception as exc:  # pragma: no cover - dependent on host OS
            lookup_error = str(exc)
        else:
            if isinstance(default_input, Mapping):
                device_name = str(default_input.get("name") or device_name)
                try:
                    raw_index = default_input.get("index")
                    device_index = int(raw_index) if raw_index is not None else None
                except (TypeError, ValueError):
                    device_index = None
            else:
                device_name = str(default_input)

        return device_name, device_index, lookup_error

    def _preflight_input_stream(self) -> bool:
        """Validate that the default input device accepts the configured format."""

        device_name, device_index, lookup_error = self._resolve_default_input_device()

        try:
            sd.check_input_settings(
                device=device_index if device_index is not None else None,
                channels=AUDIO_CHANNELS,
                samplerate=AUDIO_SAMPLE_RATE,
            )
        except Exception as exc:
            channel_label = "canal" if AUDIO_CHANNELS == 1 else "canais"
            suggestion = (
                "Selecione outro microfone nas configurações ou reduza a taxa de "
                "amostragem ou o número de canais nas opções avançadas."
            )
            message = (
                f"Não foi possível iniciar a captura: o dispositivo '{device_name}' "
                f"não aceita {AUDIO_SAMPLE_RATE} Hz / {AUDIO_CHANNELS} {channel_label}."
            )
            failure_payload: dict[str, Any] = {
                "message": message,
                "suggestion": suggestion,
                "recommendation": suggestion,
                "error": str(exc),
                "device_name": device_name,
                "device_index": device_index,
                "samplerate": AUDIO_SAMPLE_RATE,
                "channels": AUDIO_CHANNELS,
            }
            if lookup_error:
                failure_payload["device_lookup_error"] = lookup_error
            self._last_start_failure = failure_payload

            log_payload: dict[str, Any] = {
                "event": "audio.recording.preflight_failed",
                "device_name": device_name,
                "device_index": device_index,
                "sample_rate": AUDIO_SAMPLE_RATE,
                "channels": AUDIO_CHANNELS,
                "suggestion": suggestion,
                "error": str(exc),
            }
            if lookup_error:
                log_payload["device_lookup_error"] = lookup_error
            self._log.error(StructuredMessage("Audio input preflight failed.", **log_payload))

            # Não promovemos o estado global para ``ERROR_AUDIO`` durante a
            # pré-checagem: isso evitaria novos ciclos de tentativa quando o
            # usuário corrigisse a configuração do dispositivo. Em vez disso,
            # apenas registramos a falha e deixamos o estado permanecer em
            # ``IDLE`` para que o AppCore permita uma nova tentativa.
            self._log.debug(
                "Skipping audio error state transition after preflight failure.",
                extra={
                    "event": "audio.preflight_no_state_change",
                    "stage": "recording",
                    "message": message,
                    "suggestion": suggestion,
                },
            )
            return False

        self._last_start_failure = None
        return True

    def _audio_callback(self, indata, frames, time_data, status):
        if status:
            status_text = str(status).strip()
            self._log.warning(
                StructuredMessage(
                    "Audio callback reported backend status.",
                    event="audio.callback_status",
                    status=status_text,
                )
            )
            self._handle_audio_overflow(status_text)
        if self.is_recording:
            # Copy avoids references to buffers reused by SoundDevice
            self.audio_queue.put(indata.copy())

    def _frame_pool_key(self, shape: Iterable[int]) -> tuple[int, ...]:
        normalized = tuple(int(dim) for dim in shape) or (0,)
        return normalized

    def _acquire_frame_buffer(self, shape: Iterable[int]) -> np.ndarray:
        key = self._frame_pool_key(shape)
        with self._frame_pool_lock:
            bucket = self._frame_pool.get(key)
            if bucket:
                self._frame_pool_total -= 1
                buffer = bucket.pop()
                return buffer
        # Fora do pool: alocação explícita e log de debug para rastrear trade-offs.
        buffer = np.empty(key, dtype=np.float32)
        if self._log.isEnabledFor(logging.DEBUG):
            self._log.debug(
                "Allocated new frame buffer.",
                extra={
                    "event": "audio.frame_buffer_alloc", 
                    "stage": "processing_loop",
                    "shape": key,
                },
            )
        return buffer

    def _clone_frame(self, source: np.ndarray) -> np.ndarray:
        array = np.asarray(source)
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        buffer = self._acquire_frame_buffer(array.shape)
        np.copyto(buffer, array, casting="no")
        return buffer

    def _release_frame_buffer(self, frame: np.ndarray | None) -> None:
        if not isinstance(frame, np.ndarray):
            return
        if frame.dtype != np.float32:
            return
        key = self._frame_pool_key(frame.shape)
        with self._frame_pool_lock:
            if self._frame_pool_total >= self._frame_pool_total_limit:
                if self._log.isEnabledFor(logging.DEBUG):
                    self._log.debug(
                        "Discarding frame buffer due to pool saturation.",
                        extra={
                            "event": "audio.frame_buffer_discard",
                            "stage": "processing_loop",
                            "shape": key,
                            "reason": "pool_total_limit",
                        },
                    )
                return
            bucket = self._frame_pool.setdefault(key, deque())
            if len(bucket) >= self._frame_pool_bucket_limit:
                if self._log.isEnabledFor(logging.DEBUG):
                    self._log.debug(
                        "Discarding frame buffer due to bucket limit.",
                        extra={
                            "event": "audio.frame_buffer_discard",
                            "stage": "processing_loop",
                            "shape": key,
                            "reason": "bucket_limit",
                        },
                    )
                return
            bucket.append(frame)
            self._frame_pool_total += 1

    def _reset_audio_frames(self) -> None:
        if self._audio_frames:
            for frame in self._audio_frames:
                self._release_frame_buffer(frame)
        self._audio_frames = []

    def _capture_memory_probe(self) -> float | None:
        try:
            return float(get_available_memory_mb())
        except Exception:
            return None

    def _handle_audio_overflow(self, status) -> None:
        try:
            status_str = str(status).strip()
        except Exception:
            status_str = repr(status)

        if "input overflow" in status_str.lower():
            now = time.time()
            last_ts, count = (
                self._last_overflow_sample
                if isinstance(self._last_overflow_sample, tuple)
                else (0.0, 0)
            )
            if now - last_ts > self._overflow_log_window:
                count = 0
            count += 1
            self._last_overflow_sample = (now, count)

            self._log.warning(
                StructuredMessage(
                    "Audio input overflow detected.",
                    event="audio.callback_overflow",
                    occurrences=count,
                    window_seconds=self._overflow_log_window,
                )
            )

    def _resolve_directory(self, raw: Any, *, fallback: Path, description: str) -> Path:
        candidates: list[tuple[Path, str]] = []
        try:
            primary = Path(str(raw)).expanduser() if raw not in (None, "") else fallback
        except Exception:
            primary = fallback
        candidates.append((primary, "requested"))
        if fallback != primary:
            candidates.append((fallback, "default"))
        working_dir = Path.cwd()
        if working_dir not in (candidate for candidate, _ in candidates):
            candidates.append((working_dir, "working"))

        seen: set[str] = set()
        for candidate, label in candidates:
            normalized = str(candidate.expanduser())
            if normalized in seen:
                continue
            seen.add(normalized)
            candidate_path = Path(normalized)
            try:
                candidate_path.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._log.warning(
                    "Unable to ensure %s directory (%s) '%s': %s",
                    description,
                    label,
                    candidate_path,
                    exc,
                )
                continue
            if label != "requested":
                self._log.info(
                    "Using %s directory '%s' via %s fallback.",
                    description,
                    candidate_path,
                    label,
                )
            return candidate_path

        self._log.error(
            "Failed to resolve a usable %s directory; defaulting to current working directory.",
            description,
        )
        return working_dir

    def _create_temp_wav_file(self) -> tempfile.NamedTemporaryFile:
        preferred_dir = self.recordings_dir if isinstance(self.recordings_dir, Path) else None
        if preferred_dir is not None:
            try:
                preferred_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._log.error(
                    "Unable to ensure recordings directory '%s': %s; falling back to system temp.",
                    preferred_dir,
                    exc,
                )
            else:
                try:
                    return tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav",
                        dir=str(preferred_dir),
                    )
                except Exception as exc:
                    self._log.error(
                        "Failed to create temporary recording inside '%s': %s; using system temp.",
                        preferred_dir,
                        exc,
                    )
        return tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

    def _process_audio_queue(self):
        """Consume audio chunks until a shutdown sentinel (``None``) is received.

        Notes
        -----
        Atualmente apenas uma thread de processamento é criada para consumir a
        fila. Caso novos workers sejam adicionados no futuro, certifique-se de
        fornecer um sentinela por worker ao encerrar.
        """
        while True:
            try:
                indata = self.audio_queue.get()
                if indata is None:  # Stop signal
                    drained = 0
                    while True:
                        try:
                            leftover = self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                        else:
                            drained += 1
                    self._log.debug(
                        log_context(
                            "Shutdown sentinel received; audio queue drained.",
                            event="audio.processing.queue_drained",
                            stage="processing_shutdown",
                            drained_items=drained,
                        )
                    )
                    break

                if not self.in_memory_mode and self._sf_writer is None:
                    continue

                frames_to_write: list[np.ndarray] = []
                memory_before_mb = self._capture_memory_probe()
                if self.use_vad and self.vad_manager:
                    try:
                        is_speech, vad_frames = self.vad_manager.process_chunk(indata)
                    except Exception as exc:
                        is_speech, vad_frames = self._handle_vad_exception(exc, indata)
                    frames_to_write.extend(vad_frames)
                    if is_speech and not vad_frames:
                        frames_to_write.append(self._clone_frame(indata))
                else:
                    frames_to_write = [self._clone_frame(indata)]

                if not frames_to_write:
                    continue

                with self.storage_lock:
                    for frame in frames_to_write:
                        if self.in_memory_mode:
                            # Mantemos a referência original para permitir retorno ao pool
                            self._audio_frames.append(frame)
                            self._memory_samples += len(frame)

                            max_samples = self._memory_limit_samples
                            if self.record_storage_mode == 'auto' and self._memory_samples > max_samples:
                                self._log.info(
                                    log_context(
                                        "Recording duration exceeded in-memory threshold; moving buffers to disk.",
                                        event="audio.storage_threshold",
                                        threshold_seconds=self.current_max_memory_seconds,
                                        stage="storage_selection",
                                    )
                                )
                                try:
                                    total_mb = get_total_memory_mb()
                                    avail_mb = get_available_memory_mb()
                                    percent_free = (avail_mb / total_mb * 100.0) if total_mb else 0.0
                                    self._log.info(
                                        log_context(
                                            "In-memory storage migration due to recording length.",
                                            event="audio.storage_migration",
                                            stage="ram_to_disk_migration",
                                            status="duration_exceeded",
                                            percent_free_ram=round(percent_free, 1),
                                        )
                                    )
                                except Exception:
                                    pass
                                self._migrate_to_file()
                            elif self.record_storage_mode == "auto":
                                total_mb = get_total_memory_mb()
                                avail_mb = get_available_memory_mb()
                                try:
                                    raw_threshold = self.config_manager.get("auto_ram_threshold_percent")
                                except Exception:
                                    self._log.warning(
                                        "Unable to read auto RAM threshold from configuration; using default.",
                                        exc_info=True,
                                        extra={
                                            "event": "auto_ram_threshold_read_failed",
                                            "stage": "storage_selection",
                                        },
                                    )
                                    thr_percent = 10
                                else:
                                    try:
                                        thr_value = int(raw_threshold)
                                    except (TypeError, ValueError):
                                        self._log.warning(
                                            "Invalid auto RAM threshold '%s'; using default.",
                                            raw_threshold,
                                            extra={
                                                "event": "auto_ram_threshold_invalid",
                                                "stage": "storage_selection",
                                            },
                                        )
                                        thr_percent = 10
                                    else:
                                        thr_percent = max(1, min(50, thr_value))
                                percent_free = (avail_mb / total_mb * 100.0) if total_mb else 0.0
                                if total_mb and percent_free < thr_percent:
                                    self._log.info(
                                        "Free RAM below configured threshold; moving buffers to disk.",
                                        extra={
                                            "event": "ram_to_disk_low_memory",
                                            "stage": "storage_selection",
                                            "details": f"percent_free={percent_free:.1f} threshold={thr_percent}",
                                        },
                                    )
                                    try:
                                        self._log.info(
                                            log_context(
                                                "In-memory storage migration due to low available RAM.",
                                                event="audio.storage_migration_low_ram",
                                                percent_free=round(percent_free, 1),
                                                threshold_percent=thr_percent,
                                                stage="ram_to_disk_migration",
                                            )
                                        )
                                    except Exception:
                                        pass
                                    self._migrate_to_file()
                        else:
                            if self._sf_writer:
                                self._sf_writer.write(frame)
                            self._release_frame_buffer(frame)

                        self._sample_count += len(frame)
                memory_after_mb = self._capture_memory_probe()
                if (
                    memory_before_mb is not None
                    and memory_after_mb is not None
                    and self._log.isEnabledFor(logging.DEBUG)
                ):
                    self._log.debug(
                        "Memory probe around queue iteration.",
                        extra={
                            "event": "audio.memory_probe",
                            "stage": "processing_loop",
                            "before_mb": round(memory_before_mb, 2),
                            "after_mb": round(memory_after_mb, 2),
                            "delta_mb": round(memory_after_mb - memory_before_mb, 2),
                        },
                    )
            except Exception as e:
                self._log.error(
                    f"Error while processing audio queue: {e}",
                    extra={"event": "audio_queue_error", "stage": "processing_loop"},
                )
                with self.storage_lock:
                    self._sf_writer = None
                self._reset_audio_frames()
                self._memory_samples = 0

    def _handle_vad_exception(self, exc: Exception, chunk: np.ndarray) -> tuple[bool, list[np.ndarray]]:
        self._log.error(
            "Error while processing VAD chunk.",
            exc_info=True,
            extra={"event": "vad_processing_error", "stage": "vad", "details": str(exc)},
        )
        frames: list[np.ndarray] = []
        if chunk is not None:
            frames.append(self._clone_frame(chunk))
        if self.vad_manager:
            try:
                self.vad_manager.reset_states()
                self.vad_manager.enable_energy_fallback("pipeline exception", exc)
            except Exception:
                self._log.debug(
                    "Failed to reset VAD states after exception.",
                    exc_info=True,
                    extra={"event": "vad_reset_failed", "stage": "vad"},
                )
        try:
            max_abs = float(np.max(np.abs(chunk))) if chunk is not None and getattr(chunk, "size", 0) else 0.0
            self._log.debug(
                "VAD fallback; chunk_shape=%s max_abs=%.4f",
                getattr(chunk, "shape", None),
                max_abs,
            )
        except Exception:
            self._log.debug(
                "Unable to compute chunk diagnostics after VAD exception.",
                exc_info=True,
                extra={"stage": "vad", "event": "vad_chunk_diagnostics_failed"},
            )
        return True, frames

    @staticmethod
    def _coerce_padding_ms(value, fallback: float, *, key: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                log_context(
                    "Invalid padding value received; using fallback.",
                    event="audio.padding_invalid",
                    raw_value=value,
                    key=key,
                    fallback_ms=fallback,
                )
            )
            return fallback
        if numeric < 0:
            LOGGER.warning(
                log_context(
                    "Negative padding value received; using fallback.",
                    event="audio.padding_negative",
                    raw_value=value,
                    key=key,
                    fallback_ms=fallback,
                )
            )
            return fallback
        return numeric

    def _record_audio_task(self):
        session_id = self._session_id
        with scoped_correlation_id(session_id, preserve_existing=True):
            self.audio_stream = None
            try:
                self._log.info(
                    StructuredMessage(
                        "Audio recording thread started.",
                        event="audio.thread.start",
                        samplerate=AUDIO_SAMPLE_RATE,
                        channels=AUDIO_CHANNELS,
                    )
                )
                if not self.is_recording:
                    LOGGER.warning("Recording flag turned off before stream start.")
                    return

                self.audio_stream = sd.InputStream(
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    callback=self._audio_callback,
                    dtype="float32",
                    blocksize=self.stream_blocksize,
                )
                self.audio_stream.start()
                self.stream_started = True
                self._log.info(
                    StructuredMessage(
                        "Audio stream opened.",
                        event="audio.stream.opened",
                        blocksize=self.stream_blocksize,
                    )
                )

                while not self._stop_event.is_set() and self.is_recording:
                    sd.sleep(100)
                self._log.info(
                    StructuredMessage(
                        "Recording flag lowered; stopping audio stream.",
                        event="audio.stream.stop_requested",
                    )
                )
            except sd.PortAudioError as e:
                self._log.error(f"PortAudio error during recording: {e}", exc_info=True)
                self.is_recording = False
                self.state_manager.set_state(sm.STATE_ERROR_AUDIO)
            except Exception as e:
                self._log.error(f"Error in audio recording thread: {e}", exc_info=True)
                self.is_recording = False
                self.state_manager.set_state(sm.STATE_ERROR_AUDIO)
            finally:
                if self.audio_stream is not None:
                    self._close_input_stream()
                    self.audio_stream = None
                self.stream_started = False
                self._stop_event.clear()
                self._record_thread = None
                self._log.info(
                    StructuredMessage(
                        "Audio recording thread finished.",
                        event="audio.thread.stop",
                    )
                )
                try:
                    self._log.info(
                        "Recording thread cleanup completed.",
                        extra={
                            "stage": "recording",
                            "event": "record_thread_finalize",
                            "duration_ms": 0,
                        },
                    )
                except Exception:
                    pass

    def _close_input_stream(self, timeout: float = 2.0):
        finished_event = threading.Event()

        def _closer():
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self._log.info(
                    StructuredMessage(
                        "Audio stream stopped and closed.",
                        event="audio.stream.closed",
                    )
                )
            except Exception as e:
                self._log.error(f"Error stopping/closing audio stream: {e}")
            finally:
                finished_event.set()

        t = threading.Thread(target=_closer, daemon=True)
        t.start()
        finished_event.wait(timeout)
        t.join(timeout)
        if t.is_alive():
            self._log.error(
                "Close thread did not finish within timeout.",
                extra={"event": "close_stream_timeout", "duration_ms": int(timeout * 1000)},
            )

    def _migrate_to_file(self):
        """Move in-memory frames into a temporary audio file."""
        raw_tmp = self._create_temp_wav_file()
        self.temp_file_path = raw_tmp.name
        raw_tmp.close()
        self._sf_writer = sf.SoundFile(
            self.temp_file_path,
            mode="w",
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
        )
        if self._audio_frames:
            data = np.concatenate(self._audio_frames, axis=0)
            self._sf_writer.write(data)
            self._reset_audio_frames()
        self._memory_samples = 0
        self.in_memory_mode = False

    def start_recording(self):
        if self.is_recording:
            self._log.warning(
                StructuredMessage(
                    "Recording request ignored because capture is already active.",
                    event="audio.recording_already_active",
                )
            )
            return False

        session_id = uuid.uuid4().hex[:8]
        self._session_id = session_id

        try:
            with scoped_correlation_id(session_id):
                with log_operation(
                    self._log,
                    "Initializing audio recording session.",
                    event="audio.recording.session",
                    details={
                        "requested_mode": self.record_storage_mode,
                        "vad_enabled": bool(self.use_vad),
                    },
                ):
                    if not self._processing_thread or not self._processing_thread.is_alive():
                        self.audio_queue = queue.Queue()
                        self._processing_thread = threading.Thread(
                            target=self._process_audio_queue,
                            daemon=True,
                        )
                        self._processing_thread.start()

                    if self._record_thread and self._record_thread.is_alive():
                        self._log.debug(
                            "Waiting for the previous recording thread to finish.",
                            extra={"event": "record_thread_join", "stage": "recording"},
                        )
                        self._stop_event.set()
                        self._record_thread.join(timeout=2)

                    self._stop_event.clear()

                    self._enforce_record_storage_limit(exclude_paths=[self.temp_file_path])

                    if not self._preflight_input_stream():
                        self._log.debug(
                            "Audio input preflight failed; aborting recording startup.",
                            extra={"event": "audio.preflight_abort", "stage": "recording"},
                        )
                        self._session_id = None
                        return False

                    if self.record_storage_mode == "memory":
                        self.in_memory_mode = True
                        reason = "configured for memory"
                    elif self.record_storage_mode == "disk":
                        self.in_memory_mode = False
                        reason = "configured for disk"
                    else:
                        available_mb = get_available_memory_mb()
                        if available_mb >= self.min_free_ram_mb and self.max_memory_seconds > 0:
                            self.in_memory_mode = True
                            reason = (
                                f"auto: free RAM {available_mb:.0f}MB >= {self.min_free_ram_mb}MB"
                            )
                        else:
                            self.in_memory_mode = False
                            reason = (
                                f"auto: free RAM {available_mb:.0f}MB < {self.min_free_ram_mb}MB"
                            )

                        if self._record_thread and self._record_thread.is_alive():
                            self._log.debug(
                                "Waiting for the previous recording thread to finish.",
                                extra={"event": "record_thread_join", "stage": "recording"},
                            )
                            self._stop_event.set()
                            self._record_thread.join(timeout=2)

                    if not self.state_manager.transition_if(
                        sm.STATE_IDLE,
                        sm.StateEvent.AUDIO_RECORDING_STARTED,
                        source="audio_handler",
                    ):
                        self._log.debug(
                            "Recording start aborted: state guard rejected transition.",
                            extra={"event": "audio.state_guard_rejected", "stage": "recording"},
                        )
                        cleanup_path: str | os.PathLike[str] | None = None
                        with self.storage_lock:
                            self.is_recording = False
                            self.start_time = None
                            self._sample_count = 0
                            self._memory_samples = 0
                            if self._sf_writer is not None:
                                try:
                                    self._sf_writer.close()
                                except Exception:
                                    self._log.debug(
                                        "Failed to close temporary writer after guard rejection.",
                                        exc_info=True,
                                        extra={"event": "audio.writer_cleanup", "stage": "recording"},
                                    )
                                self._sf_writer = None
                            if not self.in_memory_mode:
                                cleanup_path = self.temp_file_path
                        if cleanup_path:
                            try:
                                self._cleanup_temp_file(target_path=cleanup_path)
                            except Exception:
                                self._log.debug(
                                    "Failed to clean up temporary file after guard rejection.",
                                    exc_info=True,
                                    extra={"event": "audio.temp_cleanup", "stage": "recording"},
                                )
                        return False

                        self._enforce_record_storage_limit(exclude_paths=[self.temp_file_path])

                        if self.record_storage_mode == "memory":
                            self.in_memory_mode = True
                            reason = "configured for memory"
                        elif self.record_storage_mode == "disk":
                            self.in_memory_mode = False
                            reason = "configured for disk"
                        else:
                            available_mb = get_available_memory_mb()
                            if available_mb >= self.min_free_ram_mb and self.max_memory_seconds > 0:
                                self.in_memory_mode = True
                                reason = (
                                    f"auto: free RAM {available_mb:.0f}MB >= {self.min_free_ram_mb}MB"
                                )
                            else:
                                self.in_memory_mode = False
                                reason = (
                                    f"auto: free RAM {available_mb:.0f}MB < {self.min_free_ram_mb}MB"
                                )

                        if self.max_memory_seconds_mode == "auto":
                            self.current_max_memory_seconds = self._calculate_auto_memory_seconds()
                        else:
                            self.current_max_memory_seconds = self.max_memory_seconds
                        self._memory_limit_samples = int(self.current_max_memory_seconds * AUDIO_SAMPLE_RATE)

                        self._log.info(
                            StructuredMessage(
                                "Recording storage mode decided.",
                                event="audio.storage_selected",
                                in_memory=self.in_memory_mode,
                                rationale=reason,
                                max_buffer_seconds=self.current_max_memory_seconds,
                            )
                        )

                        with self.storage_lock:
                            self.is_recording = True
                            self.start_time = time.time()
                            self._sample_count = 0
                            self._memory_samples = 0

                            if self.in_memory_mode:
                                self.temp_file_path = None
                                self._sf_writer = None
                                self._audio_frames = []
                            else:
                                raw_tmp = self._create_temp_wav_file()
                                self.temp_file_path = raw_tmp.name
                                raw_tmp.close()
                                self._sf_writer = sf.SoundFile(
                                    self.temp_file_path,
                                    mode="w",
                                    samplerate=AUDIO_SAMPLE_RATE,
                                    channels=AUDIO_CHANNELS,
                                )

                        if self.use_vad and self.vad_manager:
                            try:
                                self.vad_manager.reset_states()
                            except Exception:
                                self._log.debug(
                                    "Failed to reset VAD states for new recording.",
                                    exc_info=True,
                                    extra={"event": "vad_reset_failed", "stage": "recording"},
                                )
                        self._log.debug(
                            "VAD reset for new recording.",
                            extra={"event": "vad_reset", "stage": "recording"},
                        )

                        self.state_manager.set_state(
                            "RECORDING",
                            operation_id=operation_id,
                            source="audio_handler",
                        )

                        self._record_thread = threading.Thread(
                            target=self._record_audio_task,
                            daemon=True,
                            name="AudioRecordThread",
                        )
                        self._record_thread.start()

                        threading.Thread(
                            target=self._play_generated_tone_stream,
                            kwargs={"is_start": True},
                            daemon=True,
                            name="StartSoundThread",
                        ).start()
                        return True
        except Exception:
            self._session_id = None
            self._current_operation_id = None
            raise

    def stop_recording(self):
        session_id = self._session_id
        operation_id = self._current_operation_id
        try:
            with scoped_correlation_id(session_id, preserve_existing=True):
                if not self.is_recording:
                    LOGGER.warning(
                        StructuredMessage(
                            "Stop request ignored because no recording is active.",
                            event="audio.stop_ignored",
                            operation_id=operation_id,
                        )
                    )
                    return False

                storage_mode = "memory" if self.in_memory_mode else "disk"
                self._log.info(
                    log_context(
                        "Stop recording requested.",
                        event="audio.recording.stop_request",
                        storage_mode=storage_mode,
                        samples=self._sample_count,
                        operation_id=operation_id,
                    )
                )

                self.is_recording = False
                stream_was_started = self.stream_started
                self._stop_event.set()

                sentinels_enqueued = self._signal_processing_shutdown()
                if self._processing_thread:
                    processing_thread = self._processing_thread
                    if processing_thread is threading.current_thread():
                        self._log.debug(
                            log_context(
                                "Stop recording invoked from processing thread; skipping self-join.",
                                event="audio.processing_thread.join_skipped",
                                details={"thread_name": processing_thread.name},
                            )
                        )
                    else:
                        join_thread_with_timeout(
                            processing_thread,
                            timeout=2.0,
                            logger=self._log,
                            thread_name=processing_thread.name,
                            event_prefix="audio.processing_thread",
                        )
                    self._processing_thread = None

                if self.use_vad and self.vad_manager:
                    try:
                        self.vad_manager.reset_states()
                    except Exception:
                        self._log.debug(
                            "Failed to reset VAD states when stopping recording.",
                            exc_info=True,
                        )
                self._log.debug("VAD reset when stopping recording.")

                threading.Thread(
                    target=self._play_generated_tone_stream,
                    kwargs={"is_start": False},
                    daemon=True,
                    name="StopSoundThread",
                ).start()

                if self._record_thread:
                    self._record_thread.join(timeout=2)

                with self.storage_lock:
                    if self._sf_writer is not None:
                        writer_path = getattr(self._sf_writer, "name", self.temp_file_path)
                        try:
                            self._sf_writer.close()
                        except Exception as e:
                            self._log.error("Failed to close temporary file %s: %s", writer_path, e)
                        self._sf_writer = None

                if not stream_was_started:
                    LOGGER.warning(
                        StructuredMessage(
                            "Stop request ignored because audio stream never started.",
                            event="audio.stop_without_stream",
                            operation_id=operation_id,
                        )
                    )
                    self._cleanup_temp_file()
                    self.state_manager.transition_if(
                        (sm.STATE_RECORDING, sm.STATE_IDLE),
                        sm.StateEvent.AUDIO_RECORDING_STOPPED,
                        source="audio_handler",
                    )
                    return False

                recording_duration = time.time() - self.start_time
                if self._sample_count == 0 or recording_duration < self.min_record_duration:
                    rounded = round(recording_duration, 2)
                    self._log.info(
                        StructuredMessage(
                            "Recording discarded because it is shorter than the configured minimum.",
                            event="audio.segment_too_short",
                            captured_seconds=rounded,
                            minimum_seconds=self.min_record_duration,
                            samples_recorded=self._sample_count,
                            operation_id=operation_id,
                        )
                    )
                    LOGGER.warning(
                        StructuredMessage(
                            "Recording discarded due to insufficient duration or missing samples.",
                            event="audio.segment_discarded",
                            captured_seconds=rounded,
                            minimum_seconds=self.min_record_duration,
                            samples_recorded=self._sample_count,
                            operation_id=operation_id,
                        )
                    )
                    self._cleanup_temp_file()
                    self.state_manager.transition_if(
                        (sm.STATE_RECORDING, sm.STATE_IDLE),
                        sm.StateEvent.AUDIO_RECORDING_STOPPED,
                        source="audio_handler",
                    )
                    return False

                if self.in_memory_mode:
                    audio_data = (
                        np.concatenate(self._audio_frames, axis=0)
                        if self._audio_frames
                        else np.empty((0, AUDIO_CHANNELS), dtype=np.float32)
                    )
                    self.on_audio_segment_ready_callback(
                        audio_data.flatten(),
                        operation_id=operation_id,
                    )
                else:
                    if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
                        try:
                            ts = int(time.time())
                            filename = f"temp_recording_{ts}.wav"
                            source_path = Path(self.temp_file_path)
                            recordings_base = self.recordings_dir
                            if not isinstance(recordings_base, Path):
                                recordings_base = (
                                    Path(str(recordings_base)) if recordings_base else Path.cwd()
                                )
                            try:
                                recordings_base.mkdir(parents=True, exist_ok=True)
                            except Exception as exc:
                                self._log.error(
                                    "Failed to ensure recordings directory '%s': %s; using working directory.",
                                    recordings_base,
                                    exc,
                                )
                                recordings_base = Path.cwd()
                            target_path = (recordings_base / filename).resolve()
                            shutil.move(str(source_path), target_path)
                            self.temp_file_path = str(target_path)
                            self._log.info(
                                StructuredMessage(
                                    "Temporary recording persisted to disk.",
                                    event="audio.segment_saved",
                                    path=str(target_path),
                                    size_bytes=target_path.stat().st_size
                                    if target_path.exists()
                                    else None,
                                    operation_id=operation_id,
                                )
                            )
                            self._enforce_record_storage_limit(exclude_paths=[target_path])
                        except Exception as e:
                            self._log.error(f"Failed to save temporary recording: {e}")
                            try:
                                if "target_path" in locals() and target_path.exists():
                                    target_path.unlink()
                            except Exception:
                                pass
                            self.temp_file_path = str(source_path)
                    self.on_audio_segment_ready_callback(
                        self.temp_file_path,
                        operation_id=operation_id,
                    )
                    protected_paths: list[Path] = []
                    if self.temp_file_path:
                        try:
                            protected_paths.append(Path(self.temp_file_path))
                        except Exception:
                            pass
                    self._enforce_record_storage_limit(protected_paths=protected_paths)

                self._reset_audio_frames()
                self._memory_samples = 0
                self.start_time = None

                self._log.info(
                    log_context(
                        "Recording session finalized and dispatched for processing.",
                        event="audio.recording.session",
                        storage_mode=storage_mode,
                        duration_seconds=round(recording_duration, 2),
                        samples=self._sample_count,
                        operation_id=operation_id,
                    )
                )
                return True
        finally:
            self._session_id = None
            self._current_operation_id = None

    # ------------------------------------------------------------------
    # Beep notification sound
    # ------------------------------------------------------------------
    def _generate_tone_data(self, frequency, duration, volume_factor):
        num_samples = int(duration * AUDIO_SAMPLE_RATE)
        t = np.linspace(0, duration, num_samples, False)
        tone = np.sin(2 * np.pi * frequency * t) * volume_factor
        fade_samples = int(0.01 * AUDIO_SAMPLE_RATE)
        if fade_samples * 2 < len(tone):
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
        return tone.astype(np.float32)

    class _TonePlaybackCallback:
        def __init__(self, tone_data, finished_event, log_fn: Callable[[int, str, object], None] | None = None):
            self.tone_data = tone_data
            self.read_offset = 0
            self.finished_event = finished_event
            self._log_fn = log_fn

        def __call__(self, outdata, frames, time, status):
            if status:
                message = f"Tone playback callback status: {status}"
                if self._log_fn:
                    self._log_fn(logging.WARNING, message)
                else:
                    LOGGER.warning(
                        log_context(
                            "Tone playback callback reported status.",
                            event="audio.tone_callback_status",
                            status=str(status),
                        )
                    )
            remaining_samples = len(self.tone_data) - self.read_offset
            if remaining_samples == 0:
                outdata.fill(0)
                self.finished_event.set()
                raise sd.CallbackStop()
            chunk_size = min(frames, remaining_samples)
            outdata[:chunk_size] = self.tone_data[self.read_offset:self.read_offset + chunk_size].reshape(-1, 1)
            if chunk_size < frames:
                outdata[chunk_size:].fill(0)
                self.finished_event.set()
                raise sd.CallbackStop()
            self.read_offset += chunk_size

    def _play_generated_tone_stream(self, frequency=None, duration=None, volume=None, is_start=True):
        if not self.sound_enabled:
            self._log.debug("Sound playback skipped (disabled in settings)")
            return

        freq = frequency if frequency is not None else self.sound_frequency
        dur = duration if duration is not None else self.sound_duration
        vol = volume if volume is not None else self.sound_volume

        if not is_start:
            freq = int(freq * 0.8)

        self._log.debug(f"Attempting to play tone via OutputStream: {freq}Hz, {dur}s, vol={vol}")
        finished_event = threading.Event()
        try:
            with self.sound_lock:
                tone_data = self._generate_tone_data(freq, dur, vol)
                callback_instance = self._TonePlaybackCallback(
                    tone_data,
                    finished_event,
                    lambda level, message, *args, **kwargs: self._log.log(
                        level,
                        message,
                        *args,
                        **kwargs,
                    ),
                )
                with sd.OutputStream(
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    callback=callback_instance,
                    dtype="float32",
                ) as stream:
                    stream.start()
                    self._log.debug("OutputStream started for tone playback.")
                    finished_event.wait()
                    for _ in range(10):
                        if not stream.active:
                            break
                        time.sleep(0.01)
                    if stream.active:
                        stream.stop()
                    self._log.debug("Tone playback finished (OutputStream).")
        except Exception as e:
            self._log.error(f"Error playing tone via OutputStream: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Configuration and cleanup
    # ------------------------------------------------------------------
    def update_config(self):
        """Load or refresh settings from the ConfigManager."""
        self.record_storage_mode = self.config_manager.get("record_storage_mode", "auto")
        self.record_storage_limit = self.config_manager.get("record_storage_limit", 0)
        self.max_memory_seconds_mode = self.config_manager.get(
            "max_memory_seconds_mode", "auto"
        )
        self.max_memory_seconds = self.config_manager.get("max_memory_seconds", 30)
        self.min_free_ram_mb = self.config_manager.get("min_free_ram_mb", 1000)

        storage_root_default = Path(
            self.config_manager.default_config.get(
                STORAGE_ROOT_DIR_CONFIG_KEY,
                str(Path.cwd()),
            )
        ).expanduser()
        storage_root_value = self.config_manager.get_storage_root_dir()
        self.storage_root_dir = self._resolve_directory(
            storage_root_value,
            fallback=storage_root_default,
            description="storage root",
        )

        recordings_fallback = self.storage_root_dir / "recordings"
        recordings_value = self.config_manager.get_recordings_dir()
        self.recordings_dir = self._resolve_directory(
            recordings_value,
            fallback=recordings_fallback,
            description="recordings",
        )

        self.sound_enabled = self.config_manager.get("sound_enabled", True)
        self.sound_frequency = self.config_manager.get("sound_frequency", 400)
        self.sound_duration = self.config_manager.get("sound_duration", 0.3)
        self.sound_volume = self.config_manager.get("sound_volume", 0.5)
        self.min_record_duration = self.config_manager.get("min_record_duration", 0.5)

        self.use_vad = bool(self.config_manager.get("use_vad", False))
        self.vad_threshold = float(self.config_manager.get("vad_threshold", 0.5))
        self.vad_silence_duration = self.config_manager.get("vad_silence_duration", 1.0)
        self.vad_pre_speech_padding_ms = int(
            self.config_manager.get("vad_pre_speech_padding_ms", self.vad_pre_speech_padding_ms)
        )
        self.vad_post_speech_padding_ms = int(
            self.config_manager.get("vad_post_speech_padding_ms", self.vad_post_speech_padding_ms)
        )

        default_pre_ms = float(
            self.config_manager.default_config.get(VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY, 0.0)
        )
        raw_pre_ms = self.config_manager.get(
            VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
            default_pre_ms,
        )
        self.vad_pre_speech_padding_ms = self._coerce_padding_ms(
            raw_pre_ms,
            default_pre_ms,
            key=VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
        )

        default_post_ms = float(
            self.config_manager.default_config.get(VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY, 0.0)
        )
        raw_post_ms = self.config_manager.get(
            VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
            default_post_ms,
        )
        self.vad_post_speech_padding_ms = self._coerce_padding_ms(
            raw_post_ms,
            default_post_ms,
            key=VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
        )

        if self.use_vad:
            if self.vad_manager is None:
                self.vad_manager = VADManager(
                    threshold=self.vad_threshold,
                    sampling_rate=AUDIO_SAMPLE_RATE,
                    pre_speech_padding_ms=int(self.vad_pre_speech_padding_ms),
                    post_speech_padding_ms=int(self.vad_post_speech_padding_ms),
                    config_manager=self.config_manager,
                )
            else:
                self.vad_manager.sr = AUDIO_SAMPLE_RATE
                self.vad_manager.configure(
                    threshold=self.vad_threshold,
                    pre_padding_ms=int(self.vad_pre_speech_padding_ms),
                    post_padding_ms=int(self.vad_post_speech_padding_ms),
                )
            if not self.vad_manager.enabled:
                self._log.error("VAD disabled: model not found.")
                self.use_vad = False
                self.vad_manager = None
            else:
                try:
                    self.vad_manager.reset_states()
                except Exception:
                    self._log.debug("Failed to reset VAD states after configuration.", exc_info=True)
        else:
            self.vad_manager = None

        self._log.debug(
            "VAD padding configured (pre=%.1f ms, post=%.1f ms)",
            self.vad_pre_speech_padding_ms,
            self.vad_post_speech_padding_ms,
        )

        self._log.info(
            StructuredMessage(
                "Audio handler configuration refreshed.",
                event="audio.settings_applied",
                storage_mode=self.record_storage_mode,
                storage_limit_mb=self.record_storage_limit,
                vad_enabled=self.use_vad,
                min_duration_seconds=self.min_record_duration,
            )
        )

    def _calculate_auto_memory_seconds(self) -> float:
        """Calculate the maximum in-memory duration based on free RAM."""
        available_mb = get_available_memory_mb()
        usable_mb = max(0, available_mb - self.min_free_ram_mb)
        bytes_per_sec_mb = 64 / 1024  # 64 KB em MB
        seconds = usable_mb / bytes_per_sec_mb if bytes_per_sec_mb else 0
        self._log.debug(
            "Auto-calculated limit: %.1fs (free RAM %.0fMB)",
            seconds,
            available_mb,
        )
        return max(0.0, seconds)

    def _cleanup_temp_file(self, *, target_path: str | os.PathLike[str] | None = None) -> int:
        """Remove temporary/saved recordings and return reclaimed bytes."""

        with self.storage_lock:
            if target_path is None and self.in_memory_mode:
                self._reset_audio_frames()
                self._memory_samples = 0
                self.temp_file_path = None
                return 0

            candidate: str | os.PathLike[str] | None
            if target_path is None:
                candidate = self.temp_file_path
            else:
                candidate = target_path

            if not candidate:
                if target_path is None:
                    self.temp_file_path = None
                return 0

            path = Path(candidate)
            try:
                resolved_candidate = path.resolve()
            except Exception:
                resolved_candidate = path

            is_current_temp = False
            if self.temp_file_path:
                try:
                    is_current_temp = Path(self.temp_file_path).resolve() == resolved_candidate
                except Exception:
                    try:
                        is_current_temp = Path(self.temp_file_path).absolute() == resolved_candidate.absolute()
                    except Exception:
                        is_current_temp = os.path.abspath(str(self.temp_file_path)) == os.path.abspath(str(path))

            if not path.exists():
                if is_current_temp:
                    self.temp_file_path = None
                return 0

            try:
                reclaimed_bytes = path.stat().st_size
            except (OSError, ValueError):
                reclaimed_bytes = 0

            try:
                path.unlink()
                if reclaimed_bytes:
                    self._log.info(
                        StructuredMessage(
                            "Deleted temporary audio file.",
                            event="audio.temp_file_deleted",
                            path=str(path),
                            freed_mb=reclaimed_bytes / (1024 * 1024),
                        )
                    )
                else:
                    self._log.info(
                        StructuredMessage(
                            "Deleted temporary audio file.",
                            event="audio.temp_file_deleted",
                            path=str(path),
                            freed_mb=0.0,
                        )
                    )
            except Exception as e:
                self._log.error("Failed to remove temporary file %s: %s", path, e)
                return 0

            if is_current_temp or target_path is None:
                self.temp_file_path = None

            return reclaimed_bytes

    def _enforce_record_storage_limit(
        self,
        *,
        protected_paths: Iterable[Path | str] | None = None,
        exclude_paths: Iterable[Path | str] | None = None,
    ) -> None:
        """Enforce disk quota for persisted recordings using the configured limit (in MB).

        Parameters
        ----------
        protected_paths:
            Collection of file paths that must not be deleted while enforcing the
            storage limit.
        exclude_paths:
            Backwards-compatible alias for ``protected_paths``. Both arguments are
            merged, allowing legacy callers that still reference ``exclude_paths``
            to operate without errors.
        """

        try:
            limit_mb = int(self.record_storage_limit or 0)
        except (TypeError, ValueError):
            limit_mb = 0

        if limit_mb <= 0:
            return

        limit_bytes = limit_mb * 1024 * 1024
        protected: set[Path] = set()

        def _add_protected_paths(candidates: Iterable[Path | str] | None) -> None:
            if not candidates:
                return
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    protected.add(Path(candidate).resolve())
                except Exception:
                    try:
                        protected.add(Path(candidate))
                    except Exception:
                        continue

        _add_protected_paths(protected_paths)
        _add_protected_paths(exclude_paths)

        base_dir = self.recordings_dir if isinstance(self.recordings_dir, Path) else None
        if base_dir is None:
            raw_dir = self.recordings_dir
            if raw_dir:
                try:
                    base_dir = Path(str(raw_dir))
                except Exception:
                    self._log.error(
                        "Invalid recordings directory '%s'; skipping storage enforcement.",
                        raw_dir,
                    )
                    return
        if base_dir is None:
            self._log.debug(
                "Recordings directory not configured; skipping storage enforcement.",
            )
            return

        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._log.error(
                "Unable to ensure recordings directory '%s': %s", base_dir, exc
            )
            return

        patterns = ("temp_recording_*.wav", "recording_*.wav")
        total_bytes = 0
        candidates: list[tuple[float, Path, int]] = []

        try:
            recordings_root = Path(self.recordings_dir).expanduser()
        except Exception:
            recordings_root = Path.cwd()

        for pattern in patterns:
            for file_path in base_dir.glob(pattern):
                try:
                    stat = file_path.stat()
                except (FileNotFoundError, OSError):
                    continue
                total_bytes += stat.st_size
                candidates.append((stat.st_mtime, file_path, stat.st_size))

        if total_bytes <= limit_bytes:
            return

        self._log.info(
            "Storage quota exceeded in '%s': %.2f MB used (limit=%d MB). Pruning oldest recordings.",
            base_dir,
            total_bytes / (1024 * 1024),
            limit_mb,
        )

        candidates.sort(key=lambda item: item[0])  # oldest first

        for _, file_path, _ in candidates:
            if total_bytes <= limit_bytes:
                break
            try:
                resolved = file_path.resolve()
            except Exception:
                resolved = file_path

            if resolved in protected:
                continue

            reclaimed = self._cleanup_temp_file(target_path=file_path)
            if reclaimed:
                total_bytes -= reclaimed

        if total_bytes > limit_bytes:
            self._log.warning(
                "Could not reduce stored recordings below limit; %.2f MB remain. Some files may be locked or protected.",
                total_bytes / (1024 * 1024),
            )

        # Final sanity check: if some protected or locked files prevent us from
        # freeing enough space, log the situation but avoid unbounded
        # recursion. The caller can attempt cleanup again later once the files
        # become available.
        if total_bytes > limit_bytes:
            self._log.debug(
                "Storage cleanup incomplete; remaining usage=%.2f MB (limit=%d MB).",
                total_bytes / (1024 * 1024),
                limit_mb,
            )

    def cleanup(self):
        if self.is_recording:
            self.stop_recording()
        elif self.audio_stream is not None:
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self._log.info(
                    StructuredMessage(
                        "Audio stream stopped during cleanup.",
                        event="audio.stream.cleanup",
                    )
                )
            except Exception as e:
                self._log.error(f"Failed to close audio stream: {e}")
            finally:
                self.audio_stream = None

        if self._sf_writer is not None:
            try:
                self._sf_writer.close()
            except Exception:
                pass
            self._sf_writer = None

        self._cleanup_temp_file()

        # Finaliza a thread de processamento, caso ainda esteja ativa
        sentinels_enqueued = 0
        if self.audio_queue:
            sentinels_enqueued = self._signal_processing_shutdown()
        if self._processing_thread:
            processing_thread = self._processing_thread
            if processing_thread is threading.current_thread():
                self._log.debug(
                    "Cleanup invoked from processing thread; skipping self-join."
                )
            elif processing_thread.is_alive():
                processing_thread.join(timeout=2)  # Timeout curto evita travamentos ao desligar threads no Windows
                if sentinels_enqueued:
                    self._log.debug(
                        log_context(
                            "Audio processing thread joined during cleanup.",
                            event="audio.processing.thread_joined",
                            stage="processing_shutdown",
                        )
                    )
        self._processing_thread = None

    def _signal_processing_shutdown(self, *, workers: int | None = None) -> int:
        """Enviar sentinelas de desligamento para os workers de processamento.

        Parameters
        ----------
        workers:
            Permite especificar explicitamente quantos sentinelas devem ser
            enfileirados. Por padrão utiliza ``self._processing_workers``, o que
            garante compatibilidade futura com múltiplos consumidores.
        """

        thread = self._processing_thread
        if not thread or not thread.is_alive():
            self._log.debug(
                log_context(
                    "Processing thread inactive; skipping shutdown sentinel enqueue.",
                    event="audio.processing.sentinel_skipped",
                    stage="processing_shutdown",
                )
            )
            return 0

        target_workers = workers if workers is not None else self._processing_workers
        if target_workers <= 0:
            self._log.debug(
                log_context(
                    "No processing workers registered; no sentinels enqueued.",
                    event="audio.processing.no_workers",
                    stage="processing_shutdown",
                )
            )
            return 0

        sentinels = 0
        for _ in range(target_workers):
            try:
                self.audio_queue.put(None)
            except Exception:
                self._log.debug(
                    "Failed to enqueue shutdown sentinel for audio processing thread.",
                    exc_info=True,
                    extra={
                        "event": "audio.processing.sentinel_failed",
                        "stage": "processing_shutdown",
                    },
                )
                break
            else:
                sentinels += 1

        if sentinels:
            self._log.debug(
                log_context(
                    "Shutdown sentinel enqueued for audio processing worker(s).",
                    event="audio.processing.sentinel_enqueued",
                    stage="processing_shutdown",
                    workers=sentinels,
                )
            )

        return sentinels

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @classmethod
    def probe_default_device(
        cls,
        *,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
    ) -> dict[str, Any]:
        """Perform a non-destructive probe of the default audio input device."""

        try:
            devices = sd.query_devices()
        except Exception as exc:
            LOGGER.error(
                StructuredMessage(
                    "Unable to enumerate audio devices.",
                    event="audio.diagnostics.enumeration_failed",
                    error=str(exc),
                ),
                exc_info=True,
            )
            return {
                "ok": False,
                "message": "Failed to enumerate audio interfaces via sounddevice.",
                "details": {"error": str(exc)},
                "suggestion": (
                    "Verify that audio drivers are installed and that the sounddevice "
                    "library can access the operating system's audio stack."
                ),
                "fatal": True,
            }

        input_devices = [
            device
            for device in devices
            if isinstance(device, Mapping)
            and device.get("max_input_channels", 0) > 0
        ]
        if not input_devices:
            LOGGER.warning(
                StructuredMessage(
                    "No input-capable audio devices found during diagnostics.",
                    event="audio.diagnostics.no_input_devices",
                )
            )
            return {
                "ok": False,
                "message": "No input-capable audio devices were detected.",
                "details": {
                    "device_count": len(devices),
                    "input_device_count": 0,
                },
                "suggestion": "Connect a microphone or enable it in the operating system settings.",
                "fatal": True,
            }

        try:
            default_input = sd.query_devices(None, "input")
        except Exception as exc:
            LOGGER.error(
                StructuredMessage(
                    "Failed to resolve default audio input device.",
                    event="audio.diagnostics.default_lookup_failed",
                    error=str(exc),
                ),
                exc_info=True,
            )
            return {
                "ok": False,
                "message": "Default input device could not be resolved.",
                "details": {
                    "input_device_count": len(input_devices),
                    "error": str(exc),
                },
                "suggestion": (
                    "Select a default recording device in the operating system or "
                    "choose a specific device in the application settings."
                ),
                "fatal": True,
            }

        default_name = default_input.get("name", "Unknown") if isinstance(default_input, Mapping) else str(default_input)
        default_index = default_input.get("index") if isinstance(default_input, Mapping) else None

        samplerate_check_passed = True
        samplerate_error: str | None = None
        try:
            sd.check_input_settings(
                device=default_index if default_index is not None else None,
                channels=channels,
                samplerate=sample_rate,
            )
        except Exception as exc:
            samplerate_check_passed = False
            samplerate_error = str(exc)
            LOGGER.warning(
                StructuredMessage(
                    "Default audio device rejected requested format.",
                    event="audio.diagnostics.format_incompatible",
                    device_name=default_name,
                    sample_rate=sample_rate,
                    channels=channels,
                    error=str(exc),
                ),
                exc_info=True,
            )

        message: str
        status_ok = samplerate_check_passed
        suggestion: str | None = None
        recommendation: str | None = None
        fatal = False
        channel_label = "canal" if channels == 1 else "canais"
        if samplerate_check_passed:
            message = (
                f"O dispositivo padrão '{default_name}' está apto para captura em {sample_rate} Hz / {channels} {channel_label}."
            )
            LOGGER.info(
                StructuredMessage(
                    "Audio diagnostics succeeded.",
                    event="audio.diagnostics.success",
                    device_name=default_name,
                    sample_rate=sample_rate,
                    channels=channels,
                    input_device_count=len(input_devices),
                )
            )
        else:
            message = (
                f"O dispositivo padrão '{default_name}' não é compatível com {sample_rate} Hz / {channels} {channel_label}."
            )
            suggestion = (
                "Selecione outro microfone nas configurações ou reduza a taxa de amostragem ou o número de canais nas opções avançadas."
            )
            recommendation = suggestion
            LOGGER.warning(
                StructuredMessage(
                    "Audio diagnostics detected incompatible format.",
                    event="audio.diagnostics.incompatible_format",
                    device_name=default_name,
                    sample_rate=sample_rate,
                    channels=channels,
                    error=samplerate_error,
                )
            )

        return {
            "ok": status_ok,
            "message": message,
            "details": {
                "device_name": default_name,
                "device_index": default_index,
                "input_device_count": len(input_devices),
                "sample_rate": sample_rate,
                "channels": channels,
                "format_error": samplerate_error,
            },
            "suggestion": suggestion,
            "recommendation": recommendation,
            "fatal": fatal,
        }
