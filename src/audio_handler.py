import logging
import threading
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
from pathlib import Path
from .utils.memory import get_available_memory_mb, get_total_memory_mb

from .vad_manager import VADManager
from .config_manager import SAVE_TEMP_RECORDINGS_CONFIG_KEY

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1


class AudioHandler:
    """Gerencia a gravação de áudio em arquivo temporário ou memória."""

    def __init__(
        self,
        config_manager,
        on_audio_segment_ready_callback,
        on_recording_state_change_callback,
    ):
        self.config_manager = config_manager
        self.on_audio_segment_ready_callback = on_audio_segment_ready_callback
        self.on_recording_state_change_callback = on_recording_state_change_callback

        # Inicializa atributos que serão configurados em `update_config`
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
        self.vad_manager = None

        self.is_recording = False
        self.start_time = None
        self.audio_stream = None
        self.stream_started = False
        self._stop_event = threading.Event()
        self._record_thread = None
        self.sound_lock = threading.RLock()
        self.storage_lock = threading.Lock()

        self._vad_silence_counter = 0.0

        self.temp_file_path: str | None = None
        self._sf_writer: sf.SoundFile | None = None
        self._audio_frames: list[np.ndarray] = []
        self._sample_count = 0
        self._memory_samples = 0

        self.update_config()  # Carrega a configuração inicial

    # ------------------------------------------------------------------
    # Grava\u00e7\u00e3o
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_data, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if not self.is_recording:
            return
        if not self.in_memory_mode and self._sf_writer is None:
            return

        write_data = None
        if self.use_vad and self.vad_manager:
            is_speech = self.vad_manager.is_speech(indata[:, 0])
            if is_speech:
                self._vad_silence_counter = 0.0
                write_data = indata.copy()
            else:
                self._vad_silence_counter += len(indata) / AUDIO_SAMPLE_RATE
                if self._vad_silence_counter <= self.vad_silence_duration:
                    write_data = indata.copy()
        else:
            write_data = indata.copy()

        if write_data is not None:
            with self.storage_lock:
                if self.in_memory_mode:
                    self._audio_frames.append(write_data.copy())
                    self._memory_samples += len(write_data)

                    # Verifica se é necessário mover para o disco
                    max_samples = self._memory_limit_samples
                    if self.record_storage_mode == 'auto' and self._memory_samples > max_samples:
                        logging.info(
                            f"Duração da gravação excedeu {self.current_max_memory_seconds}s. Migrando da RAM para o disco."
                        )
                        self._migrate_to_file()
                    elif self.record_storage_mode == 'auto':
                        total_mb = get_total_memory_mb()
                        avail_mb = get_available_memory_mb()
                        if total_mb and avail_mb / total_mb < 0.1:
                            logging.info(
                                "RAM livre abaixo de 10% do total. Migrando da RAM para o disco."
                            )
                            self._migrate_to_file()
                else:
                    if self._sf_writer:
                        self._sf_writer.write(write_data)

                self._sample_count += len(write_data)

    def _record_audio_task(self):
        self.audio_stream = None
        try:
            logging.info("Audio recording thread started.")
            if not self.is_recording:
                logging.warning("Recording flag turned off before stream start.")
                return

            self.audio_stream = sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                callback=self._audio_callback,
                dtype="float32",
            )
            self.audio_stream.start()
            self.stream_started = True
            logging.info("Audio stream started.")

            while not self._stop_event.is_set() and self.is_recording:
                sd.sleep(100)
            logging.info("Recording flag is off. Stopping audio stream.")
        except sd.PortAudioError as e:
            logging.error(f"PortAudio error during recording: {e}", exc_info=True)
            self.is_recording = False
            self.on_recording_state_change_callback("ERROR_AUDIO")
        except Exception as e:
            logging.error(f"Error in audio recording thread: {e}", exc_info=True)
            self.is_recording = False
            self.on_recording_state_change_callback("ERROR_AUDIO")
        finally:
            if self.audio_stream is not None:
                self._close_input_stream()
                self.audio_stream = None
            self.stream_started = False
            self._stop_event.clear()
            self._record_thread = None
            logging.info("Audio recording thread finished.")

    def _close_input_stream(self, timeout: float = 2.0):
        finished_event = threading.Event()

        def _closer():
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                logging.info("Audio stream stopped and closed.")
            except Exception as e:
                logging.error(f"Error stopping/closing audio stream: {e}")
            finally:
                finished_event.set()

        t = threading.Thread(target=_closer, daemon=True)
        t.start()
        finished_event.wait(timeout)
        t.join(timeout)
        if t.is_alive():
            logging.error("Thread de fechamento n\u00e3o terminou em %ss", timeout)

    def _migrate_to_file(self):
        """Move os quadros gravados em memória para um arquivo temporário."""
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
            logging.warning("Grava\u00e7\u00e3o j\u00e1 est\u00e1 ativa.")
            return False

        if self._record_thread and self._record_thread.is_alive():
            logging.debug("Aguardando t\u00e9rmino da thread de grava\u00e7\u00e3o anterior.")
            self._stop_event.set()
            self._record_thread.join(timeout=2)

        self._stop_event.clear()

        if self.record_storage_mode == "memory":
            self.in_memory_mode = True
            reason = "configurado para memory"
        elif self.record_storage_mode == "disk":
            self.in_memory_mode = False
            reason = "configurado para disk"
        else:  # Modo Automático
            available_mb = get_available_memory_mb()
            if available_mb >= self.min_free_ram_mb and self.max_memory_seconds > 0:
                self.in_memory_mode = True
                reason = f"auto: RAM livre {available_mb:.0f}MB >= {self.min_free_ram_mb}MB"
            else:
                self.in_memory_mode = False
                reason = f"auto: RAM livre {available_mb:.0f}MB < {self.min_free_ram_mb}MB"
        logging.info(
            "Decis\u00e3o de armazenamento: in_memory=%s (%s)",
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
            self.vad_manager.reset_states()
        self._vad_silence_counter = 0.0
        logging.debug("VAD reiniciado e contador de sil\u00eancio zerado para nova grava\u00e7\u00e3o.")

        self.on_recording_state_change_callback("RECORDING")

        self._record_thread = threading.Thread(target=self._record_audio_task, daemon=True, name="AudioRecordThread")
        self._record_thread.start()

        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": True}, daemon=True, name="StartSoundThread").start()
        return True

    def stop_recording(self):
        if not self.is_recording:
            logging.warning("Grava\u00e7\u00e3o n\u00e3o est\u00e1 ativa para ser parada.")
            return False

        self.is_recording = False
        stream_was_started = self.stream_started
        self._stop_event.set()

        if self.use_vad and self.vad_manager:
            self.vad_manager.reset_states()
        self._vad_silence_counter = 0.0
        logging.debug("VAD reiniciado e contador de sil\u00eancio zerado ao parar a grava\u00e7\u00e3o.")

        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": False}, daemon=True, name="StopSoundThread").start()

        if self._record_thread:
            self._record_thread.join(timeout=2)

        with self.storage_lock:
            if self._sf_writer is not None:
                try:
                    self._sf_writer.close()
                except Exception as e:
                    logging.error(f"Erro ao fechar arquivo temporário: {e}")
                self._sf_writer = None

        if not stream_was_started:
            logging.warning("Stop recording called but audio stream never started. Ignoring data.")
            self._cleanup_temp_file()
            self.on_recording_state_change_callback("IDLE")
            return False

        recording_duration = time.time() - self.start_time
        if self._sample_count == 0 or recording_duration < self.min_record_duration:
            logging.warning(
                f"Grava\u00e7\u00e3o muito curta (< {self.min_record_duration}s) ou vazia. Descartando."
            )
            self._cleanup_temp_file()
            self.on_recording_state_change_callback("IDLE")
            return False

        if self.in_memory_mode:
            audio_data = (
                np.concatenate(self._audio_frames, axis=0)
                if self._audio_frames
                else np.empty((0, AUDIO_CHANNELS), dtype=np.float32)
            )
            self.on_audio_segment_ready_callback(audio_data.flatten())
        else:
            # Se não estiver no modo de memória, o arquivo temporário já contém todos os dados
            if self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
                try:
                    ts = int(time.time())
                    filename = f"temp_recording_{ts}.wav"
                    Path(self.temp_file_path).rename(filename)
                    # Usa sf.write para forçar a criação de um arquivo vazio
                    sf.write(filename, np.empty((0, 1), dtype=np.float32), AUDIO_SAMPLE_RATE)
                    self.temp_file_path = filename
                    logging.info(f"Gravação temporária salva em {filename}")
                except Exception as e:
                    logging.error(f"Falha ao salvar gravação temporária: {e}")
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception:
                        pass
                    self.temp_file_path = filename
            self.on_audio_segment_ready_callback(self.temp_file_path)

        if not self.config_manager.get(SAVE_TEMP_RECORDINGS_CONFIG_KEY):
            self._cleanup_temp_file()

        # Clear in-memory data; temporary file is kept for downstream processing
        self._audio_frames = []
        self._memory_samples = 0
        self.start_time = None
        self.on_recording_state_change_callback("TRANSCRIBING")
        return True

    # ------------------------------------------------------------------
    # Som de aviso (beep)
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
        def __init__(self, tone_data, finished_event):
            self.tone_data = tone_data
            self.read_offset = 0
            self.finished_event = finished_event

        def __call__(self, outdata, frames, time, status):
            if status:
                logging.warning(f"Tone playback callback status: {status}")
            remaining_samples = len(self.tone_data) - self.read_offset
            if remaining_samples == 0:
                outdata.fill(0)
                self.finished_event.set()
                raise sd.CallbackStop()
            chunk_size = min(frames, remaining_samples)
            outdata[:chunk_size] = self.tone_data[self.read_offset : self.read_offset + chunk_size].reshape(-1, 1)
            if chunk_size < frames:
                outdata[chunk_size:].fill(0)
                self.finished_event.set()
                raise sd.CallbackStop()
            self.read_offset += chunk_size

    def _play_generated_tone_stream(self, frequency=None, duration=None, volume=None, is_start=True):
        if not self.sound_enabled:
            logging.debug("Sound playback skipped (disabled in settings)")
            return

        freq = frequency if frequency is not None else self.sound_frequency
        dur = duration if duration is not None else self.sound_duration
        vol = volume if volume is not None else self.sound_volume

        if not is_start:
            freq = int(freq * 0.8)

        logging.debug(f"Attempting to play tone via OutputStream: {freq}Hz, {dur}s, vol={vol}")
        finished_event = threading.Event()
        try:
            with self.sound_lock:
                tone_data = self._generate_tone_data(freq, dur, vol)
                callback_instance = self._TonePlaybackCallback(tone_data, finished_event)
                with sd.OutputStream(
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    callback=callback_instance,
                    dtype="float32",
                    start=False,
                ) as stream:
                    stream.start()
                    logging.debug("OutputStream started for tone playback.")
                    finished_event.wait()
                    for _ in range(10):
                        if not stream.active:
                            break
                        time.sleep(0.01)
                    if stream.active:
                        stream.stop()
                    logging.debug("Tone playback finished (OutputStream).")
        except Exception as e:
            logging.error(f"Error playing tone via OutputStream: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Configura\u00e7\u00f5es e limpeza
    # ------------------------------------------------------------------
    def update_config(self):
        """Carrega ou atualiza as configurações a partir do ConfigManager."""
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

        self.use_vad = self.config_manager.get("use_vad", False)
        self.vad_threshold = self.config_manager.get("vad_threshold", 0.5)
        self.vad_silence_duration = self.config_manager.get("vad_silence_duration", 1.0)

        if self.use_vad:
            if self.vad_manager is None:
                self.vad_manager = VADManager(threshold=self.vad_threshold)
            else:
                self.vad_manager.threshold = self.vad_threshold
            if not self.vad_manager.enabled:
                logging.error("VAD desativado: modelo não encontrado.")
                self.use_vad = False
                self.vad_manager = None
        else:
            self.vad_manager = None

        logging.info(
            "AudioHandler: Settings updated (mode=%s, limit=%s)",
            self.record_storage_mode,
            self.record_storage_limit,
        )

    def _calculate_auto_memory_seconds(self) -> float:
        """Calcula o tempo máximo em memória baseado na RAM livre."""
        available_mb = get_available_memory_mb()
        usable_mb = max(0, available_mb - self.min_free_ram_mb)
        bytes_per_sec_mb = 64 / 1024  # 64 KB em MB
        seconds = usable_mb / bytes_per_sec_mb if bytes_per_sec_mb else 0
        logging.debug(
            "Limite automático calculado: %.1fs (RAM livre %.0fMB)",
            seconds,
            available_mb,
        )
        return max(0.0, seconds)

    def _cleanup_temp_file(self):
        with self.storage_lock:
            if self.in_memory_mode:
                self._audio_frames = []
            elif self.temp_file_path and os.path.exists(self.temp_file_path):
                try:
                    os.remove(self.temp_file_path)
                    logging.info(f"Deleted temp audio file: {self.temp_file_path}")
                except Exception as e:
                    logging.error(f"Erro ao remover arquivo temporário: {e}")
            self.temp_file_path = None

    def cleanup(self):
        if self.is_recording:
            self.stop_recording()
        elif self.audio_stream is not None:
            try:
                if self.audio_stream.active:
                    self.audio_stream.stop()
                self.audio_stream.close()
                logging.info("Audio stream stopped during cleanup.")
            except Exception as e:
                logging.error(f"Erro ao fechar stream de \u00e1udio: {e}")
            finally:
                self.audio_stream = None

        if self._sf_writer is not None:
            try:
                self._sf_writer.close()
            except Exception:
                pass
            self._sf_writer = None

        self._cleanup_temp_file()
