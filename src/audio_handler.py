import sounddevice as sd
import numpy as np
import threading
import logging
import time
import soundfile as sf
from .vad_manager import VADManager # Assumindo que vad_manager.py está na raiz ou em um path acessível
import os
import wave

# Constantes de áudio (movidas de whisper_tkinter.py)
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

class AudioHandler:
    def __init__(self, config_manager, on_audio_segment_ready_callback, on_recording_state_change_callback):
        self.config_manager = config_manager
        self.on_audio_segment_ready_callback = on_audio_segment_ready_callback # Callback para enviar segmento para transcrição
        self.on_recording_state_change_callback = on_recording_state_change_callback # Callback para atualizar estado da UI

        self.is_recording = False
        self.start_time = None
        self.recording_data = []
        self.audio_stream = None
        self.sound_lock = threading.RLock()
        self.stream_started = False
        self.save_temp_recordings = self.config_manager.get("save_temp_recordings")
        self.temp_file_path = None

        # Carregar configurações de som
        self.sound_enabled = self.config_manager.get("sound_enabled")
        self.sound_frequency = self.config_manager.get("sound_frequency")
        self.sound_duration = self.config_manager.get("sound_duration")
        self.sound_volume = self.config_manager.get("sound_volume")
        self.min_record_duration = self.config_manager.get("min_record_duration")

        self.use_vad = self.config_manager.get("use_vad")
        self.vad_threshold = self.config_manager.get("vad_threshold")
        self.vad_silence_duration = self.config_manager.get("vad_silence_duration")
        self.vad_manager = VADManager(threshold=self.vad_threshold) if self.use_vad else None
        if self.use_vad and self.vad_manager and not self.vad_manager.enabled:
            logging.error("VAD desativado: modelo n\u00e3o encontrado.")
            self.use_vad = False
            self.vad_manager = None
        self._vad_silence_counter = 0.0
        self._stop_event = threading.Event()
        self._record_thread = None

    def _audio_callback(self, indata, frames, time_data, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if self.is_recording:
            if self.use_vad:
                is_speech = self.vad_manager.is_speech(indata[:, 0])
                if is_speech:
                    self._vad_silence_counter = 0.0
                    self.recording_data.append(indata.copy())
                else:
                    self._vad_silence_counter += len(indata) / AUDIO_SAMPLE_RATE
                    if self._vad_silence_counter <= self.vad_silence_duration:
                        self.recording_data.append(indata.copy())
            else:
                self.recording_data.append(indata.copy())

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
                dtype='float32'
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
            self.on_recording_state_change_callback("ERROR_AUDIO") # Notificar erro
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

        t = threading.Thread(target=_closer)
        t.start()
        finished_event.wait(timeout)
        t.join(timeout)
        if t.is_alive():
            logging.error("Thread de fechamento n\u00e3o terminou em %ss", timeout)


    def start_recording(self):
        if self.is_recording:
            logging.warning("Gravação já está ativa.")
            return False

        if self._record_thread and self._record_thread.is_alive():
            logging.debug("Aguardando término da thread de gravação anterior.")
            self._stop_event.set()
            self._record_thread.join(timeout=2)

        self._stop_event.clear()

        self.is_recording = True
        self.start_time = time.time()
        self.recording_data.clear()

        if self.use_vad and self.vad_manager:
            self.vad_manager.reset_states()
        self._vad_silence_counter = 0.0
        logging.debug("VAD reiniciado e contador de silêncio zerado para nova gravação.")

        self.on_recording_state_change_callback("RECORDING")

        if self.use_vad and self.vad_manager:
            self.vad_manager.reset_states()
            logging.debug("Estados do VAD reiniciados.")

        self._record_thread = threading.Thread(target=self._record_audio_task, daemon=True, name="AudioRecordThread")
        self._record_thread.start()
        
        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": True}, daemon=True, name="StartSoundThread").start()
        return True

    def stop_recording(self):
        if not self.is_recording:
            logging.warning("Gravação não está ativa para ser parada.")
            return False

        self.is_recording = False
        stream_was_started = self.stream_started
        self._stop_event.set()

        if self.use_vad and self.vad_manager:
            self.vad_manager.reset_states()
        self._vad_silence_counter = 0.0
        logging.debug("VAD reiniciado e contador de silêncio zerado ao parar a gravação.")

        threading.Thread(target=self._play_generated_tone_stream, kwargs={"is_start": False}, daemon=True, name="StopSoundThread").start()

        if self._record_thread:
            self._record_thread.join(timeout=2)

        if not stream_was_started:
            logging.warning("Stop recording called but audio stream never started. Ignoring data.")
            self.recording_data.clear()
            self.on_recording_state_change_callback("IDLE")
            return False

        recording_duration = time.time() - self.start_time
        if not self.recording_data:
            logging.warning("Nenhum áudio capturado. Gravacao interrompida antes do início.")
            self.on_recording_state_change_callback("IDLE")
            return False
        if recording_duration < self.min_record_duration:
            logging.warning(f"Gravação muito curta (< {self.min_record_duration}s). Descartando.")
            self.recording_data.clear()
            self.on_recording_state_change_callback("IDLE")
            return False

        full_audio = np.concatenate(self.recording_data)
        self.recording_data.clear()

        # Garantir que o áudio seja mono (1 canal) e tenha a dimensão correta (1D)
        if full_audio.ndim > 1:
            if full_audio.shape[1] > 1:
                logging.info(f"Áudio gravado tem {full_audio.shape[1]} canais. Convertendo para mono.")
                full_audio = np.mean(full_audio, axis=1)
            else:
                # Se já for mono mas em formato 2D (n, 1), achata para 1D (n,)
                logging.info("Áudio já é mono, achatando para formato 1D.")
                full_audio = full_audio.flatten()

        if self.config_manager.get("save_temp_recordings"):
            try:
                ts = int(time.time())
                filename = f"temp_recording_{ts}.wav"
                sf.write(filename, full_audio, AUDIO_SAMPLE_RATE)
                logging.info(f"Temporary recording saved to {filename}")
            except Exception as e:
                logging.error(f"Failed to save temporary recording: {e}")

        self.start_time = None
        # Mudar o estado para TRANSCRIBING ANTES de enviar o áudio para processamento
        self.on_recording_state_change_callback("TRANSCRIBING")
        
        self.on_audio_segment_ready_callback(full_audio)
        return True

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
        stream = None
        try:
            with self.sound_lock:
                tone_data = self._generate_tone_data(freq, dur, vol)
                callback_instance = self._TonePlaybackCallback(tone_data, finished_event)
                stream = sd.OutputStream(
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=AUDIO_CHANNELS,
                    callback=callback_instance,
                    dtype='float32'
                )
                stream.start()
                logging.debug("OutputStream started for tone playback.")
            finished_event.wait()
            logging.debug("Tone playback finished (OutputStream).")
        except Exception as e:
            logging.error(f"Error playing tone via OutputStream: {e}", exc_info=True)
        finally:
            if stream is not None:
                try:
                    if stream.active:
                        stream.stop()
                    stream.close()
                    logging.debug("OutputStream stopped and closed.")
                except Exception as e:
                    logging.error(f"Error stopping/closing OutputStream: {e}")

    def update_config(self):
        """Recarrega configurações e atualiza componentes internos."""
        self.sound_enabled = self.config_manager.get("sound_enabled")
        self.sound_frequency = self.config_manager.get("sound_frequency")
        self.sound_duration = self.config_manager.get("sound_duration")
        self.sound_volume = self.config_manager.get("sound_volume")
        self.min_record_duration = self.config_manager.get("min_record_duration")
        self.save_temp_recordings = self.config_manager.get("save_temp_recordings")

        self.use_vad = self.config_manager.get("use_vad")
        self.vad_threshold = self.config_manager.get("vad_threshold")
        self.vad_silence_duration = self.config_manager.get("vad_silence_duration")

        if self.use_vad:
            if self.vad_manager is None:
                self.vad_manager = VADManager(threshold=self.vad_threshold)
            else:
                self.vad_manager.threshold = self.vad_threshold
            if not self.vad_manager.enabled:
                logging.error("VAD desativado: modelo n\u00e3o encontrado.")
                self.use_vad = False
                self.vad_manager = None
        else:
            self.vad_manager = None

        self._vad_silence_counter = 0.0

        logging.info("AudioHandler: Configurações atualizadas.")
