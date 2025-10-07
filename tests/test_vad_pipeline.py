import unittest
from collections import deque

import numpy as np

try:
    from src.vad_manager import VADManager
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.vad_manager import VADManager


class DummyConfigManager:
    def __init__(self, overrides: dict | None = None):
        base = {
            "vad_threshold": 0.5,
            "vad_pre_speech_padding_ms": 0,
            "vad_post_speech_padding_ms": 0,
        }
        if overrides:
            base.update(overrides)
        self._data = base

    def get(self, key, default=None):
        return self._data.get(key, default)


class TestVADPipeline(unittest.TestCase):
    def _make_energy_fallback_manager(self, sampling_rate: int, overrides: dict | None = None) -> VADManager:
        manager = VADManager.__new__(VADManager)
        manager.config_manager = DummyConfigManager(overrides)
        manager.sr = int(sampling_rate)
        manager.threshold = float(manager.config_manager.get("vad_threshold", 0.5))
        manager.pre_speech_padding_ms = int(
            manager.config_manager.get("vad_pre_speech_padding_ms", 0)
        )
        manager.post_speech_padding_ms = int(
            manager.config_manager.get("vad_post_speech_padding_ms", 0)
        )
        manager._chunk_counter = 0
        manager.session = None
        manager._use_energy_fallback = True
        manager.enabled = True
        manager._state = manager._coerce_state_tensor(None)
        manager._pre_buffer = deque()
        manager._pre_buffer_samples = 0
        manager._speech_active = False
        manager._post_silence_samples = 0
        manager._update_padding_samples()
        return manager

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
        manager = VADManager(
            sampling_rate=16000,
            config_manager=DummyConfigManager({"vad_threshold": 0.1}),
        )
        manager.session = None
        manager._use_energy_fallback = True
        chunk = np.ones(1600, dtype=np.float32) * 0.2
        self.assertTrue(manager.is_speech(chunk))

    def test_process_chunk_contract(self):
        manager = VADManager(
            sampling_rate=16000,
            config_manager=DummyConfigManager(
                {
                    "vad_threshold": 0.01,
                    "vad_pre_speech_padding_ms": 0,
                    "vad_post_speech_padding_ms": 0,
                }
            ),
        )
        manager.session = None
        manager._use_energy_fallback = True

        chunk = np.ones(320, dtype=np.float32) * 0.2
        is_speech, frames = manager.process_chunk(chunk)
        self.assertTrue(is_speech)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].dtype, np.float32)
        np.testing.assert_allclose(frames[0], chunk)

        manager.reset_states()
        silence = np.zeros(320, dtype=np.float32)
        is_speech_silence, frames_silence = manager.process_chunk(silence)
        self.assertFalse(is_speech_silence)
        self.assertEqual(frames_silence, [])

    def test_padding_samples_scale_for_8k_and_48k(self):
        overrides = {
            "vad_threshold": 0.2,
            "vad_pre_speech_padding_ms": 100,
            "vad_post_speech_padding_ms": 250,
        }
        manager_8k = self._make_energy_fallback_manager(8000, overrides)
        manager_48k = self._make_energy_fallback_manager(48000, overrides)

        self.assertEqual(manager_8k._pre_padding_samples, 800)
        self.assertEqual(manager_8k._post_padding_samples, 2000)
        self.assertEqual(manager_48k._pre_padding_samples, 4800)
        self.assertEqual(manager_48k._post_padding_samples, 12000)

    def test_process_chunk_with_saturated_audio(self):
        manager = self._make_energy_fallback_manager(
            16000,
            {
                "vad_threshold": 0.1,
                "vad_pre_speech_padding_ms": 0,
                "vad_post_speech_padding_ms": 0,
            },
        )
        saturated = np.ones(1600, dtype=np.float32) * 4.0
        prepared, peak = VADManager._prepare_input(saturated)
        self.assertGreater(peak, 1.0)
        self.assertLessEqual(float(np.max(np.abs(prepared))), 1.0 + 1e-6)

        is_speech, frames = manager.process_chunk(saturated)
        self.assertTrue(is_speech)
        self.assertEqual(len(frames), 1)
        np.testing.assert_allclose(frames[0], saturated)

    def test_process_chunk_respects_post_padding_at_different_rates(self):
        overrides = {
            "vad_threshold": 0.05,
            "vad_pre_speech_padding_ms": 0,
            "vad_post_speech_padding_ms": 125,
        }
        manager_8k = self._make_energy_fallback_manager(8000, overrides)
        manager_48k = self._make_energy_fallback_manager(48000, overrides)

        chunk_8k = np.ones(400, dtype=np.float32) * 0.3
        self.assertTrue(manager_8k.process_chunk(chunk_8k)[0])
        for _ in range(3):
            manager_8k.process_chunk(np.zeros_like(chunk_8k))
        self.assertFalse(manager_8k.process_chunk(np.zeros_like(chunk_8k))[0])

        chunk_48k = np.ones(960, dtype=np.float32) * 0.3
        self.assertTrue(manager_48k.process_chunk(chunk_48k)[0])
        for _ in range(12):
            manager_48k.process_chunk(np.zeros_like(chunk_48k))
        self.assertFalse(manager_48k.process_chunk(np.zeros_like(chunk_48k))[0])

    def test_prepare_input_for_large_multichannel_chunk(self):
        samples = 96000
        channels = 6
        chunk = np.random.uniform(-1.5, 1.5, size=(samples, channels)).astype(np.float64)
        prepared, peak = VADManager._prepare_input(chunk)

        self.assertEqual(prepared.shape, (1, samples))
        self.assertEqual(prepared.dtype, np.float32)
        self.assertTrue(prepared.flags["C_CONTIGUOUS"])
        self.assertGreaterEqual(peak, 0.0)


if __name__ == "__main__":
    unittest.main()
