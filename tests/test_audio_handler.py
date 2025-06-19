import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Módulos falsos para dependências nativas ausentes
fake_sd = types.SimpleNamespace(
    PortAudioError=Exception,
    InputStream=MagicMock()
)
fake_onnx = types.ModuleType('onnxruntime')
fake_onnx.InferenceSession = MagicMock()
fake_torch = types.ModuleType('torch')
fake_torch.from_numpy = MagicMock(return_value=types.SimpleNamespace())

sys.modules['sounddevice'] = fake_sd
sys.modules['onnxruntime'] = fake_onnx
sys.modules['torch'] = fake_torch

from src.audio_handler import AudioHandler

class DummyConfig:
    def __init__(self):
        self.data = {
            'sound_enabled': False,
            'sound_frequency': 440,
            'sound_duration': 0.1,
            'sound_volume': 0.5,
            'min_record_duration': 0,
            'use_vad': False,
            'vad_threshold': 0.5,
            'vad_silence_duration': 0.5,
        }
    def get(self, key):
        return self.data.get(key)

class AudioHandlerTest(unittest.TestCase):
    def setUp(self):
        self.config = DummyConfig()
        self.handler = AudioHandler(self.config, lambda x: None, lambda x: None)

    def test_stream_cleanup_on_portaudio_error(self):
        fake_sd.InputStream.side_effect = fake_sd.PortAudioError()
        with patch('src.audio_handler.sd', fake_sd):
            self.handler.is_recording = True
            self.handler._record_audio_task()
            self.assertIsNone(self.handler.audio_stream)
            self.assertFalse(self.handler.stream_started)

if __name__ == '__main__':
    unittest.main()
