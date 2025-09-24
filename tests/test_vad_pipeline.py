import unittest

import numpy as np

try:
    from src.vad_manager import VADManager
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.vad_manager import VADManager


class TestVADPipeline(unittest.TestCase):
    def test_prepare_input_for_mono_chunk(self):
        chunk = np.random.uniform(-0.5, 0.5, size=320).astype(np.float64)
        prepared, peak = VADManager._prepare_input(chunk)
        self.assertEqual(prepared.shape, (1, 320))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1.0 + 1e-6)
        self.assertGreaterEqual(peak, 0.0)

    def test_prepare_input_for_stereo_chunk(self):
        frame_count = 160
        left = np.linspace(-1.2, 1.2, frame_count, dtype=np.float32)
        right = np.linspace(1.2, -1.2, frame_count, dtype=np.float32)
        chunk = np.stack([left, right], axis=1)
        prepared, peak = VADManager._prepare_input(chunk)
        self.assertEqual(prepared.shape, (1, frame_count))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1.0 + 1e-6)
        self.assertGreaterEqual(peak, 0.0)

    def test_prepare_input_for_channel_first_chunk(self):
        frame_count = 80
        left = np.linspace(-0.8, 0.8, frame_count, dtype=np.float32)
        right = -left
        chunk = np.stack([left, right], axis=0)
        prepared, peak = VADManager._prepare_input(chunk)
        self.assertEqual(prepared.shape, (1, frame_count))
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1.0 + 1e-6)
        self.assertGreaterEqual(peak, 0.0)

    def test_energy_gate_detects_loud_audio(self):
        quiet = np.zeros(1600, dtype=np.float32)
        detected_quiet, peak_quiet, rms_quiet, adjusted = VADManager._energy_gate(quiet, 0.5)
        self.assertFalse(detected_quiet)
        self.assertEqual(peak_quiet, 0.0)
        self.assertEqual(rms_quiet, 0.0)
        self.assertGreater(adjusted, 0.0)

        loud = np.ones(1600, dtype=np.float32) * 0.2
        detected_loud, peak_loud, rms_loud, adjusted_loud = VADManager._energy_gate(loud, 0.5)
        self.assertTrue(detected_loud)
        self.assertGreater(peak_loud, 0.0)
        self.assertGreater(rms_loud, 0.0)
        self.assertAlmostEqual(adjusted, adjusted_loud)

    def test_is_speech_uses_energy_fallback_when_session_missing(self):
        manager = VADManager.__new__(VADManager)
        manager.threshold = 0.5
        manager.sr = 16000
        manager.enabled = True
        manager._chunk_counter = 0
        manager.session = None
        manager._state = np.zeros((2, 1, 128), dtype=np.float32)
        manager._use_energy_fallback = True
        chunk = np.ones(1600, dtype=np.float32) * 0.2
        self.assertTrue(VADManager.is_speech(manager, chunk))


if __name__ == "__main__":
    unittest.main()
