import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import keyboard  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - provide lightweight stub
    keyboard = types.SimpleNamespace()

    def _dummy_handle(*_args, **_kwargs):
        return object()

    keyboard.add_hotkey = _dummy_handle
    keyboard.remove_hotkey = lambda *_args, **_kwargs: None
    keyboard.on_release_key = _dummy_handle
    keyboard.hook = _dummy_handle
    keyboard.unhook = lambda *_args, **_kwargs: None
    sys.modules["keyboard"] = keyboard

try:
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    import os

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


class ImmediateThread:
    def __init__(self, *_, target=None, **__):
        self._target = target

    def start(self):
        if self._target:
            self._target()


class TestHotkeyDebounce(unittest.TestCase):
    def test_toggle_hotkey_respects_debounce_window(self):
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyboardHotkeyManager(config_file=Path(tmpdir) / "hotkey.json")
            manager.set_callbacks(toggle=lambda: events.append("toggle"))
            manager.set_debounce_window(200)

            with mock.patch("src.keyboard_hotkey_manager.threading.Thread", new=ImmediateThread):
                with mock.patch(
                    "src.keyboard_hotkey_manager.time.perf_counter",
                    side_effect=[1.0, 1.15, 1.41],
                ):
                    manager._on_toggle_key()
                    manager._on_toggle_key()
                    manager._on_toggle_key()

        self.assertEqual(events, ["toggle", "toggle"])
        self.assertIn("toggle", manager._last_trigger_ts)
        self.assertGreaterEqual(manager._last_trigger_ts["toggle"], 0.0)

    def test_agent_hotkey_without_debounce_triggers_every_time(self):
        events: list[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = KeyboardHotkeyManager(config_file=Path(tmpdir) / "hotkey.json")
            manager.set_callbacks(agent=lambda: events.append("agent"))
            manager.set_debounce_window(0)

            with mock.patch("src.keyboard_hotkey_manager.threading.Thread", new=ImmediateThread):
                with mock.patch(
                    "src.keyboard_hotkey_manager.time.perf_counter",
                    side_effect=[5.0, 5.05],
                ):
                    manager._on_agent_key()
                    manager._on_agent_key()

        self.assertEqual(events, ["agent", "agent"])


if __name__ == "__main__":
    unittest.main()
