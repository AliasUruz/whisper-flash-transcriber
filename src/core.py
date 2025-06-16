import logging
import pystray
from typing import Optional

from config_manager import ConfigManager
from audio_handler import AudioHandler
from transcription_handler import TranscriptionHandler
from ui_manager import UIManager, STATE_IDLE, STATE_LOADING_MODEL, STATE_RECORDING, STATE_TRANSCRIBING

class AppCore:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.transcriber = TranscriptionHandler(self.config)
        self.ui = UIManager(self.shutdown)
        self.audio = AudioHandler(self._on_audio_ready, self.config.get('min_record_duration', 0.5))
        menu = pystray.Menu(pystray.MenuItem('Gravar', self.toggle_recording), pystray.MenuItem('Sair', self.shutdown))
        self.ui.start_tray(STATE_IDLE, menu)

    def _on_audio_ready(self, audio):
        self.ui.update_state(STATE_TRANSCRIBING)
        text = self.transcriber.transcribe(audio)
        if text:
            logging.info(f"Transcrição: {text}")
        self.ui.update_state(STATE_IDLE)

    def start_recording(self):
        if not self.audio.is_recording:
            self.audio.start()
            self.ui.update_state(STATE_RECORDING)

    def stop_recording(self):
        if self.audio.is_recording:
            self.audio.stop()

    def toggle_recording(self, icon=None, item=None):
        if self.audio.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def shutdown(self, icon=None, item=None):
        self.audio.stop()
        self.ui.stop()
