from __future__ import annotations

import logging
import threading
import queue
import os
import time
import shutil

import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
from pathlib import Path
from typing import Callable

# Observação: em ambientes WSL, a biblioteca sounddevice depende de servidores PulseAudio;
# mantenha esta limitação em mente ao depurar gravações.
from .utils.memory import get_available_memory_mb, get_total_memory_mb

from .vad_manager import VADManager
from .config_manager import (
    SAVE_TEMP_RECORDINGS_CONFIG_KEY,
    VAD_PRE_SPEECH_PADDING_MS_CONFIG_KEY,
    VAD_POST_SPEECH_PADDING_MS_CONFIG_KEY,
)

LOGGER = logging.getLogger('whisper_flash_transcriber.audio')


class _AudioLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter que inclui contexto de thread e armazenamento."""

    def process(self, msg, kwargs):
        handler = self.extra.get('handler')
        storage_mode = 'memory' if getattr(handler, 'in_memory_mode', False) else 'disk'
        thread_name = threading.current_thread().name
        prefix = f"[thread={thread_name}][storage={storage_mode}] "
        return prefix + str(msg), kwargs


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
        self.max_memory_seconds_mode = "manual"
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

        # Dedicated queue and thread for audio processing
        self.audio_queue = queue.Queue()
        self._processing_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self._processing_thread.start()

        self._audio_log = _AudioLoggerAdapter(LOGGER, {'handler': self})

        self.stream_blocksize = int(AUDIO_SAMPLE_RATE / 10)  # ~100ms buffers
        self._overflow_log_window = 5.0  # seconds
        self._last_overflow_sample: tuple[float, int] | None = None
        self.update_config()  # Load the initial configuration

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_data, status):
        if status:
            LOGGER.warning(f"Audio callback status: {status}")
            self._handle_audio_overflow(status)
        if self.is_recording:
            # Copy avoids references to buffers reused by SoundDevice
            self.audio_queue.put(indata.copy())

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

            LOGGER.warning(
                "Audio input overflow detected (%s occurrences in last %.0fs).",
                count,
                self._overflow_log_window,
            )

    def _process_audio_queue(self):
        while True:
            try:
                indata = self.audio_queue.get()
                if indata is None:  # Stop signal
                    break

                if not self.in_memory_mode and self._sf_writer is None:
                    continue

                frames_to_write: list[np.ndarray] = []
                if self.use_vad and self.vad_manager:
                    try:
                        is_speech, vad_frames = self.vad_manager.process_chunk(indata)
                    except Exception as exc:
                        is_speech, vad_frames = self._handle_vad_exception(exc, indata)
                    frames_to_write.extend(vad_frames)
                    if is_speech and not vad_frames:
                        frames_to_write.append(np.asarray(indata, dtype=np.float32).copy())
                else:
                    frames_to_write = [np.asarray(indata, dtype=np.float32).copy()]

                if not frames_to_write:
                    continue

                with self.storage_lock:
                    for frame in frames_to_write:
                        if self.in_memory_mode:
                            self._audio_frames.append(frame.copy())
                            self._memory_samples += len(frame)

                            max_samples = self._memory_limit_samples
                            if self.record_storage_mode == 'auto' and self._memory_samples > max_samples:
                                self._audio_log.info(
                                    f"Recording duration exceeded {self.current_max_memory_seconds}s. Moving from RAM to disk."
                                )
                                try:
                                    total_mb = get_total_memory_mb()
                                    avail_mb = get_available_memory_mb()
                                    percent_free = (avail_mb / total_mb * 100.0) if total_mb else 0.0
                                    self._audio_log.info(
                                        f"[METRIC] stage=ram_to_disk_migration reason=time_exceeded percent_free={percent_free:.1f}"
                                    )
                                except Exception:
                                    pass
                                self._migrate_to_file()
                            elif self.record_storage_mode == 'auto':
                                total_mb = get_total_memory_mb()
                                avail_mb = get_available_memory_mb()
                                try:
                                    thr_percent = max(
                                        1, min(50, int(self.config_manager.get("auto_ram_threshold_percent")))
                                    )
                                except Exception:
                                    thr_percent = 10
                                percent_free = (avail_mb / total_mb * 100.0) if total_mb else 0.0
                                if total_mb and percent_free < thr_percent:
                                    self._audio_log.info(
                                        f"Free RAM below {thr_percent}% of total ({percent_free:.1f}%). Moving from RAM to disk."
                                    )
                                    try:
                                        self._audio_log.info(
                                            f"[METRIC] stage=ram_to_disk_migration reason=low_free_ram percent_free={percent_free:.1f} threshold={thr_percent}"
                                        )
                                    except Exception:
                                        pass
                                    self._migrate_to_file()
                        else:
                            if self._sf_writer:
                                self._sf_writer.write(frame)

                        self._sample_count += len(frame)
            except Exception as e:
                self._audio_log.error(f"Error while processing audio queue: {e}")
                with self.storage_lock:
                    self._sf_writer = None
                self._audio_frames = []
                self._memory_samples = 0

    def _handle_vad_exception(self, exc: Exception, chunk: np.ndarray) -> tuple[bool, list[np.ndarray]]:
        self._audio_log.error("Error while processing VAD: %s", exc, exc_info=True)
        frames: list[np.ndarray] = []
        if chunk is not None:
            frames.append(np.asarray(chunk, dtype=np.float32).copy())
        if self.vad_manager:
            try:
                self.vad_manager.reset_states()
                self.vad_manager.enable_energy_fallback("pipeline exception", exc)
            except Exception:
                self._audio_log.debug("Failed to reset VAD states after exception.", exc_info=True)
        try:
            max_abs = float(np.max(np.abs(chunk))) if chunk is not None and getattr(chunk, "size", 0) else 0.0
            self._audio_log.debug(
                "VAD fallback; chunk_shape=%s max_abs=%.4f",
                getattr(chunk, "shape", None),
                max_abs,
            )
        except Exception:
            self._audio_log.debug("Unable to compute chunk diagnostics after VAD exception.", exc_info=True)
        return True, frames

    @staticmethod
    def _coerce_padding_ms(value, fallback: float, *, key: str) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            LOGGER.warning(
                "Invalid value '%s' for %s; using default %.1f ms.",
                value,
                key,
                fallback,
            )
            return fallback
        if numeric < 0:
            LOGGER.warning(
                "Negative value '%s' for %s; using default %.1f ms.",
                value,
                key,
                fallback,
            )
            return fallback
        return numeric

    def _record_audio_task(self):
        self.audio_stream = None
        try:
            self._audio_log.info("Audio recording thread started.")
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
            self._audio_log.info("Audio stream started.")

            while not self._stop_event.is_set() and self.is_recording:
                sd.sleep(100)
            self._audio_log.info("Recording flag is off. Stopping audio stream.")
        except sd.PortAudioError as e:
            self._audio_log.error(f"PortAudio error during recording: {e}", exc_info=True)
            self.is_recording = False
            self.state_manager.set_state("ERROR_AUDIO")
        except Exception as e:
            self._audio_log.error(f"Error in audio recording thread: {e}", exc_info=True)
            self.is_recording = False
            self.state_manager.set_state("ERROR_AUDIO")
        finally:
            if self.audio_stream is not None:
                self._close_input_stream()
                self.audio_stream = None
            self.stream_started = False
            self._stop_event.clear()
            self._record_thread = None
            self._audio_log.info("Audio recording thread finished.")
            # Stop overhead metric for the recording thread (event-based approximation)
            try:
                self._audio_log.info("[METRIC] stage=record_thread_finalize value_ms=0")
            except Exception:
                pass

    def _close_input_stream(self, timeout: float = 2.0):
        finished_event = threading.Event()

        def _closer():
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self._audio_log.info("Audio stream stopped and closed.")
            except Exception as e:
                self._audio_log.error(f"Error stopping/closing audio stream: {e}")
            finally:
                finished_event.set()

        t = threading.Thread(target=_closer, daemon=True)
        t.start()
        finished_event.wait(timeout)
        t.join(timeout)
        if t.is_alive():
            self._audio_log.error("Close thread did not finish within %ss", timeout)

    def _migrate_to_file(self):
        """Move in-memory frames into a temporary audio file."""
        raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
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
            self._audio_frames = []
        self._memory_samples = 0
        self.in_memory_mode = False

    def start_recording(self):
        if self.is_recording:
            LOGGER.warning("Recording is already active.")
            return False
        if not self._processing_thread or not self._processing_thread.is_alive():
            self.audio_queue = queue.Queue()
            self._processing_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
            self._processing_thread.start()

        if self._record_thread and self._record_thread.is_alive():
            self._audio_log.debug("Waiting for the previous recording thread to finish.")
            self._stop_event.set()
            self._record_thread.join(timeout=2)  # Timeout curto evita travamentos ao desligar threads no Windows

        self._stop_event.clear()

        if self.record_storage_mode == "memory":
            self.in_memory_mode = True
            reason = "configured for memory"
        elif self.record_storage_mode == "disk":
            self.in_memory_mode = False
            reason = "configured for disk"
        else:  # Automatic mode
            available_mb = get_available_memory_mb()
            if available_mb >= self.min_free_ram_mb and self.max_memory_seconds > 0:
                self.in_memory_mode = True
                reason = f"auto: free RAM {available_mb:.0f}MB >= {self.min_free_ram_mb}MB"
            else:
                self.in_memory_mode = False
                reason = f"auto: free RAM {available_mb:.0f}MB < {self.min_free_ram_mb}MB"
        self._audio_log.info(
            "Storage decision: in_memory=%s (%s)",
            self.in_memory_mode,
            reason,
        )

        if self.max_memory_seconds_mode == "auto":
            self.current_max_memory_seconds = self._calculate_auto_memory_seconds()
        else:
            self.current_max_memory_seconds = self.max_memory_seconds
        self._memory_limit_samples = int(self.current_max_memory_seconds * AUDIO_SAMPLE_RATE)

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
                raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                self.temp_file_path = raw_tmp.name
                raw_tmp.close()
                self._sf_writer = sf.SoundFile(
                    self.temp_file_path, mode="w", samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS
                )

        if self.use_vad and self.vad_manager:
            try:
                self.vad_manager.reset_states()
            except Exception:
                self._audio_log.debug("Failed to reset VAD states for new recording.", exc_info=True)
        self._audio_log.debug("VAD reset for new recording.")

        self.state_manager.set_state("RECORDING")

        self._record_thread = threading.Thread(target=self._record_audio_task, daemon=True, name="AudioRecordThread")
        self._record_thread.start()

        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": True}, daemon=True, name="StartSoundThread").start()
        return True

    def stop_recording(self):
        if not self.is_recording:
            LOGGER.warning("Recording is not active and cannot be stopped.")
            return False

        self.is_recording = False
        stream_was_started = self.stream_started
        self._stop_event.set()

        # Encerra a thread consumidora
        try:
            self.audio_queue.put(None)
        except Exception:
            pass
        if self._processing_thread:
            processing_thread = self._processing_thread
            if processing_thread is threading.current_thread():
                self._audio_log.debug(
                    "Stop recording invoked from processing thread; skipping self-join."
                )
            elif processing_thread.is_alive():
                processing_thread.join()
            self._processing_thread = None

        if self.use_vad and self.vad_manager:
            try:
                self.vad_manager.reset_states()
            except Exception:
                self._audio_log.debug("Failed to reset VAD states when stopping recording.", exc_info=True)
        self._audio_log.debug("VAD reset when stopping recording.")

        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": False}, daemon=True, name="StopSoundThread").start()

        if self._record_thread:
            self._record_thread.join(timeout=2)  # Timeout curto evita travamentos ao desligar threads no Windows

        with self.storage_lock:
            if self._sf_writer is not None:
                writer_path = getattr(self._sf_writer, 'name', self.temp_file_path)
                try:
                    self._sf_writer.close()
                except Exception as e:
                    self._audio_log.error("Failed to close temporary file %s: %s", writer_path, e)
                self._sf_writer = None

        if not stream_was_started:
            LOGGER.warning("Stop recording called but audio stream never started. Ignoring data.")
            self._cleanup_temp_file()
            self.state_manager.set_state("IDLE")
            return False

        recording_duration = time.time() - self.start_time
        if self._sample_count == 0 or recording_duration < self.min_record_duration:
            self._audio_log.info(
                f"Recorded duration {recording_duration:.2f}s below configured minimum {self.min_record_duration}s; discarding segment."
            )
            LOGGER.warning(
                f"Recording shorter than {self.min_record_duration}s or empty; discarding segment."
            )
            self._cleanup_temp_file()
            self.state_manager.set_state("IDLE")
            return False

        if self.in_memory_mode:
            audio_data = (
                np.concatenate(self._audio_frames, axis=0)
                if self._audio_frames
                else np.empty((0, AUDIO_CHANNELS), dtype=np.float32)
            )
            self.on_audio_segment_ready_callback(audio_data.flatten())
        else:
            # When not in memory mode, the temporary file already contains all data
            if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
                try:
                    ts = int(time.time())
                    filename = f"temp_recording_{ts}.wav"
                    source_path = Path(self.temp_file_path)
                    target_path = (Path.cwd() / filename).resolve()
                    shutil.move(str(source_path), target_path)
                    self.temp_file_path = str(target_path)
                    self._audio_log.info(f"Temporary recording saved at {target_path}")
                except Exception as e:
                    self._audio_log.error(f"Failed to save temporary recording: {e}")
                    try:
                        if "target_path" in locals() and target_path.exists():
                            target_path.unlink()
                    except Exception:
                        pass
                    # Keep the original path so downstream consumers can still use the source file
                    self.temp_file_path = str(source_path)
            self.on_audio_segment_ready_callback(self.temp_file_path)

        # Cleanup happens downstream after transcription completes when temporary
        # recordings are not being kept on disk.

        # Clear in-memory data; temporary file is kept for downstream processing
        self._audio_frames = []
        self._memory_samples = 0
        self.start_time = None
        return True

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
                    LOGGER.warning(message)
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
            self._audio_log.debug("Sound playback skipped (disabled in settings)")
            return

        freq = frequency if frequency is not None else self.sound_frequency
        dur = duration if duration is not None else self.sound_duration
        vol = volume if volume is not None else self.sound_volume

        if not is_start:
            freq = int(freq * 0.8)

        self._audio_log.debug(f"Attempting to play tone via OutputStream: {freq}Hz, {dur}s, vol={vol}")
        finished_event = threading.Event()
        try:
            with self.sound_lock:
                tone_data = self._generate_tone_data(freq, dur, vol)
                callback_instance = self._TonePlaybackCallback(
                    tone_data,
                    finished_event,
                    lambda level, message, *args, **kwargs: self._audio_log.log(
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
                    self._audio_log.debug("OutputStream started for tone playback.")
                    finished_event.wait()
                    for _ in range(10):
                        if not stream.active:
                            break
                        time.sleep(0.01)
                    if stream.active:
                        stream.stop()
                    self._audio_log.debug("Tone playback finished (OutputStream).")
        except Exception as e:
            self._audio_log.error(f"Error playing tone via OutputStream: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Configuration and cleanup
    # ------------------------------------------------------------------
    def update_config(self):
        """Load or refresh settings from the ConfigManager."""
        self.record_storage_mode = self.config_manager.get("record_storage_mode", "auto")
        self.record_storage_limit = self.config_manager.get("record_storage_limit", 0)
        self.max_memory_seconds_mode = self.config_manager.get("max_memory_seconds_mode", "manual")
        self.max_memory_seconds = self.config_manager.get("max_memory_seconds", 30)
        self.min_free_ram_mb = self.config_manager.get("min_free_ram_mb", 1000)

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
                self._audio_log.error("VAD disabled: model not found.")
                self.use_vad = False
                self.vad_manager = None
            else:
                try:
                    self.vad_manager.reset_states()
                except Exception:
                    self._audio_log.debug("Failed to reset VAD states after configuration.", exc_info=True)
        else:
            self.vad_manager = None

        self._audio_log.debug(
            "VAD padding configured (pre=%.1f ms, post=%.1f ms)",
            self.vad_pre_speech_padding_ms,
            self.vad_post_speech_padding_ms,
        )

        self._audio_log.info(
            "AudioHandler: Settings updated (mode=%s, limit=%s)",
            self.record_storage_mode,
            self.record_storage_limit,
        )

    def _calculate_auto_memory_seconds(self) -> float:
        """Calculate the maximum in-memory duration based on free RAM."""
        available_mb = get_available_memory_mb()
        usable_mb = max(0, available_mb - self.min_free_ram_mb)
        bytes_per_sec_mb = 64 / 1024  # 64 KB em MB
        seconds = usable_mb / bytes_per_sec_mb if bytes_per_sec_mb else 0
        self._audio_log.debug(
            "Auto-calculated limit: %.1fs (free RAM %.0fMB)",
            seconds,
            available_mb,
        )
        return max(0.0, seconds)

    def _cleanup_temp_file(self):
        with self.storage_lock:
            if self.in_memory_mode:
                self._audio_frames = []
            elif self.temp_file_path and os.path.exists(self.temp_file_path):
                temp_path = self.temp_file_path
                try:
                    os.remove(temp_path)
                    self._audio_log.info("Deleted temp audio file: %s", temp_path)
                except Exception as e:
                    self._audio_log.error("Failed to remove temporary file %s: %s", temp_path, e)
            self.temp_file_path = None

    def cleanup(self):
        if self.is_recording:
            self.stop_recording()
        elif self.audio_stream is not None:
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                self._audio_log.info("Audio stream stopped during cleanup.")
            except Exception as e:
                self._audio_log.error(f"Failed to close audio stream: {e}")
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
        if self.audio_queue:
            try:
                self.audio_queue.put(None)
            except Exception:
                pass
        if self._processing_thread:
            processing_thread = self._processing_thread
            if processing_thread is threading.current_thread():
                self._audio_log.debug(
                    "Cleanup invoked from processing thread; skipping self-join."
                )
            elif processing_thread.is_alive():
                processing_thread.join(timeout=2)  # Timeout curto evita travamentos ao desligar threads no Windows
            self._processing_thread = None
