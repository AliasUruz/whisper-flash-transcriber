import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import types
import time
import numpy as np
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


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

from src.audio_handler import AudioHandler  # noqa: E402


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

        def noop(_):
            return None

        self.handler = AudioHandler(self.config, noop, noop)

    def test_stream_cleanup_on_portaudio_error(self):
        fake_sd.InputStream.side_effect = fake_sd.PortAudioError()
        with patch('src.audio_handler.sd', fake_sd):
            self.handler.is_recording = True
            self.handler._record_audio_task()
            self.assertIsNone(self.handler.audio_stream)
            self.assertFalse(self.handler.stream_started)

    def test_start_and_stop_recording_success(self):
        results = []

        def on_ready(audio):
            results.append(audio)

        handler = AudioHandler(self.config, on_ready, lambda *_: None)

        def fake_record_audio_task(self):
            self.stream_started = True
            while not self._stop_event.is_set() and self.is_recording:
                self.recording_data.append(np.zeros((2, 1), dtype=np.float32))
                time.sleep(0.01)
            self.stream_started = False
            self._stop_event.clear()
            self._record_thread = None

        with patch.object(AudioHandler, '_record_audio_task', fake_record_audio_task):
            with patch.object(AudioHandler, '_play_generated_tone_stream', lambda *a, **k: None):
                with patch('logging.warning') as mock_warn:
                    started = handler.start_recording()
                    time.sleep(0.05)
                    stopped = handler.stop_recording()

        self.assertTrue(started)
        self.assertTrue(stopped)
        self.assertTrue(len(results) == 1)
        mock_warn.assert_not_called()

    def test_temp_recording_saved(self):
        config = DummyConfig()
        config.data['save_audio_for_debug'] = True
        handler = AudioHandler(config, lambda *_: None, lambda *_: None)
        handler.is_recording = True
        handler.start_time = time.time() - 1
        handler.stream_started = True
        handler.recording_data = [np.zeros((2, 1), dtype=np.float32)]

        with patch.object(AudioHandler, '_play_generated_tone_stream', lambda *a, **k: None):
            with patch('src.audio_handler.sf.write') as mock_write:
                with patch('src.audio_handler.time.strftime', return_value='20240101_120000'):
                    handler.stop_recording()

        mock_write.assert_called_once()
        expected = 'temp_recording_20240101_120000.wav'
        self.assertEqual(handler.temp_file_path, expected)


if __name__ == '__main__':
    unittest.main()
