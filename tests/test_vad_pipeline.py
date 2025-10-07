import builtins
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest import mock

import numpy as np

if not hasattr(builtins, "Mapping"):
    builtins.Mapping = Mapping

try:
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager


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


class TestKeyboardHotkeys(unittest.TestCase):
    def setUp(self):
        self.keyboard_mock = mock.Mock()
        self.keyboard_patch = mock.patch(
            "src.keyboard_hotkey_manager.keyboard",
            self.keyboard_mock,
        )
        self.keyboard_patch.start()
        self.addCleanup(self.keyboard_patch.stop)

        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_path = Path(self.temp_dir.name) / "hotkeys.json"

    def _create_manager(self):
        return KeyboardHotkeyManager(config_file=self.config_path)

    def test_stop_clears_hotkey_state(self):
        manager = self._create_manager()
        manager.is_running = True
        manager.hotkey_handlers = {
            "record:press": ["handle-1", "handle-2"],
            "agent:press": ["handle-3"],
        }

        manager.stop()

        self.assertEqual(
            self.keyboard_mock.unhook.call_args_list,
            [
                mock.call("handle-1"),
                mock.call("handle-2"),
                mock.call("handle-3"),
            ],
        )
        self.assertEqual(manager.hotkey_handlers, {})
        self.assertFalse(manager.is_running)

    def test_restart_debounces_and_restarts_handlers(self):
        manager = self._create_manager()
        manager.is_running = True
        manager.hotkey_handlers = {"record:press": ["handle-1"]}

        call_order: list[str] = []
        original_stop = manager.stop

        def stop_side_effect():
            call_order.append("stop")
            return original_stop()

        def start_side_effect():
            call_order.append("start")
            manager.is_running = True
            manager.hotkey_handlers = {"record:press": ["new-handle"]}
            return True

        with mock.patch.object(manager, "stop", side_effect=stop_side_effect) as mock_stop:
            with mock.patch.object(manager, "start", side_effect=start_side_effect) as mock_start:
                with mock.patch("src.keyboard_hotkey_manager.time.sleep") as mock_sleep:
                    mock_sleep.side_effect = lambda duration: call_order.append(f"sleep:{duration}")
                    result = manager.restart()

        self.assertTrue(result)
        self.assertEqual(call_order, ["stop", "sleep:0.5", "sleep:0.5", "start"])
        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        self.assertEqual(
            mock_sleep.call_args_list,
            [mock.call(0.5), mock.call(0.5)],
        )
        self.assertEqual(
            self.keyboard_mock.unhook.call_args_list,
            [mock.call("handle-1")],
        )
        self.assertIn("record:press", manager.hotkey_handlers)
        self.assertNotIn("handle-1", manager.hotkey_handlers["record:press"])


if __name__ == "__main__":
    unittest.main()
