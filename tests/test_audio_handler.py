import sys
import os
import glob
import unittest
from unittest.mock import MagicMock, patch
import types
import time
import numpy as np
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

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
from src.config_manager import SAVE_TEMP_RECORDINGS_CONFIG_KEY  # noqa: E402


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
            SAVE_TEMP_RECORDINGS_CONFIG_KEY: False,
        }

    def get(self, key):
        return self.data.get(key)


class AudioHandlerTest(unittest.TestCase):
    def setUp(self):
        self.config = DummyConfig()

        def noop(_):
            return None

        self.handler = AudioHandler(self.config, noop, noop)

    def tearDown(self):
        """Remove any temporary recordings created during tests."""
        for f in glob.glob('temp_recording_*.wav'):
            os.remove(f)
        self.handler.temp_file_path = None

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

    def test_temp_recording_saved_and_cleanup(self):
        """Garante que o arquivo temporário é salvo e removido ao final."""
        self.config.data[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = True

        handler = AudioHandler(self.config, lambda *_: None, lambda *_: None)

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
                with patch('time.time', return_value=1111111111):
                    handler.start_recording()
                    time.sleep(0.05)
                    handler.stop_recording()
                    self.assertEqual(handler.temp_file_path, 'temp_recording_1111111111.wav')

        # Confere se o arquivo temporário foi criado corretamente
        filename = 'temp_recording_1111111111.wav'
        self.assertTrue(os.path.exists(filename))
        self.assertEqual(handler.temp_file_path, filename)

        # Limpeza explícita de todos os arquivos temporários gerados
        for f in glob.glob('temp_recording_*.wav'):
            os.remove(f)
        handler.temp_file_path = None

        self.assertFalse(os.path.exists(filename))
        self.assertIsNone(handler.temp_file_path)

    def test_temp_recording_save_error(self):
        self.config.data[SAVE_TEMP_RECORDINGS_CONFIG_KEY] = True
        handler = AudioHandler(self.config, lambda *_: None, lambda *_: None)

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
                with patch('time.time', return_value=2222222222):
                    with patch('src.audio_handler.sf.write', side_effect=Exception('fail')):
                        handler.start_recording()
                        time.sleep(0.05)
                        handler.stop_recording()

        self.assertIsNone(handler.temp_file_path)

    def test_close_input_stream_thread_does_not_block(self):
        class SlowStream:
            def __init__(self):
                self.active = True

            def stop(self):
                time.sleep(0.2)
                self.active = False

            def close(self):
                time.sleep(0.2)

        self.handler.audio_stream = SlowStream()

        captured = {}
        orig_thread = threading.Thread

        def capture_thread(*args, **kwargs):
            t = orig_thread(*args, **kwargs)
            captured['t'] = t
            return t

        with patch('src.audio_handler.threading.Thread', side_effect=capture_thread):
            start = time.time()
            self.handler._close_input_stream(timeout=0.05)
            elapsed = time.time() - start

        t = captured['t']
        self.assertTrue(elapsed < 0.15)
        self.assertTrue(t.daemon)
        alive_after = t.is_alive()
        t.join()
        self.assertTrue(alive_after)


if __name__ == '__main__':
    unittest.main()
