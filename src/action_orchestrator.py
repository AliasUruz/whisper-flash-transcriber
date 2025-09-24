
import pyperclip
import pyautogui
from transcription_handler import TranscriptionHandler
from config_manager import ConfigManager
from state_manager import StateManager

class ActionOrchestrator:
    def __init__(self, state_manager: StateManager, transcription_handler: TranscriptionHandler, config_manager: ConfigManager):
        self.state_manager = state_manager
        self.transcription_handler = transcription_handler
        self.config_manager = config_manager

    def handle_audio_segment(self, audio_source):
        """
        Handles the audio segment by initiating transcription.
        """
        self.transcription_handler.transcribe_audio(audio_source, self._handle_transcription_result)

    def _handle_transcription_result(self, result):
        """
        Handles the result of the transcription, copying to clipboard and pasting.
        """
        if self.config_manager.get_setting('paste_on_transcribe'):
            pyperclip.copy(result)
            pyautogui.hotkey('ctrl', 'v')
