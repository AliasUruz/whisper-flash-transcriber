import threading
import time
import numpy as np
import sounddevice as sd
import logging
from typing import Callable

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

class AudioHandler:
    def __init__(self, on_complete: Callable[[np.ndarray], None], min_duration: float = 0.5):
        self.on_complete = on_complete
        self.min_duration = min_duration
        self.is_recording = False
        self.start_time = None
        self.stream = None
        self.data = []
        self.lock = threading.RLock()

    def start(self):
        with self.lock:
            if self.is_recording:
                logging.warning("Gravação já em andamento")
                return
            self.is_recording = True
            self.start_time = time.time()
            self.data = []
        threading.Thread(target=self._record_task, daemon=True, name="AudioRecord").start()
        logging.info("Gravação iniciada")

    def stop(self):
        with self.lock:
            if not self.is_recording:
                return
            self.is_recording = False
        logging.info("Parando gravação")

    def _record_task(self):
        try:
            with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=AUDIO_CHANNELS, dtype='float32') as self.stream:
                while True:
                    with self.lock:
                        if not self.is_recording:
                            break
                    audio_chunk, _ = self.stream.read(1024)
                    self.data.append(audio_chunk.copy())
            logging.debug("Stream encerrada")
        except Exception as e:
            logging.error(f"Erro na captura de áudio: {e}")
        finally:
            self.stream = None
            duration = time.time() - self.start_time if self.start_time else 0
            audio = np.concatenate(self.data, axis=0) if self.data else np.array([], dtype=np.float32)
            if duration >= self.min_duration:
                self.on_complete(audio)
            else:
                logging.info("Gravação descartada por ser muito curta")
            self.data.clear()
            self.start_time = None
