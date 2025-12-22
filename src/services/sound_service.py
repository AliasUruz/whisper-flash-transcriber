import struct
import math
import winsound
import threading
import logging

class SoundService:
    def __init__(self):
        logging.info("Initializing SoundService...")
        self.sample_rate = 44100

    def _generate_tone(self, frequency: int, duration_ms: int, volume: int) -> bytes:
        """Generates a WAVE byte string for a sine wave."""
        if volume <= 0: return b""
        
        num_samples = int(self.sample_rate * (duration_ms / 1000.0))
        amplitude = int(32767 * (volume / 100.0))
        
        # RIFF Header
        data_size = num_samples * 2
        header = b'RIFF' + struct.pack('<I', 36 + data_size) + b'WAVE'
        header += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, self.sample_rate, self.sample_rate * 2, 2, 16)
        header += b'data' + struct.pack('<I', data_size)
        
        data = bytearray(header)
        
        # Sine wave generation
        for i in range(num_samples):
            t = float(i) / self.sample_rate
            sample = int(amplitude * math.sin(2 * math.pi * frequency * t))
            data.extend(struct.pack('<h', sample))
            
        return bytes(data)

    def _play_worker(self, wav_data: bytes):
        try:
            winsound.PlaySound(wav_data, winsound.SND_MEMORY)
        except Exception as err:
            logging.error(f"SoundService PlaySound error: {err}")

    def play_tone(self, frequency: int, duration_ms: int = 150, volume: int = 50, enabled: bool = True):
        """Plays a generated tone asynchronously."""
        if not enabled or volume <= 0:
            return

        try:
            wav_data = self._generate_tone(frequency, duration_ms, volume)
            if wav_data:
                threading.Thread(target=self._play_worker, args=(wav_data,), daemon=True).start()
        except Exception as e:
            logging.error(f"SoundService error: {e}")
