import importlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

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
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager

try:
    import src.config_manager as config_manager_module
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    import src.config_manager as config_manager_module


@contextmanager
def isolated_config_environment(profile_dir: str, *, working_dir: str | None = None):
    env_var = "WHISPER_FLASH_PROFILE_DIR"
    previous_env = os.environ.get(env_var)
    previous_cwd = os.getcwd()
    try:
        if working_dir is not None:
            os.chdir(working_dir)
        os.environ[env_var] = profile_dir
        module = importlib.reload(config_manager_module)
        yield module
    finally:
        if previous_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = previous_env
        os.chdir(previous_cwd)
        importlib.reload(config_manager_module)


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


class TestConfigMigration(unittest.TestCase):
    def test_legacy_record_to_memory_key_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps({"record_to_memory": True}), encoding="utf-8")

            with isolated_config_environment(tmp_dir) as config_module:
                manager = config_module.ConfigManager(config_file=str(config_path))

                self.assertEqual(manager.config.get("record_storage_mode"), "memory")
                self.assertNotIn("record_to_memory", manager.config)

                persisted_text = config_path.read_text(encoding="utf-8")
                self.assertNotIn("record_to_memory", persisted_text)

                persisted = json.loads(persisted_text)
                storage_settings = (
                    persisted.get("advanced", {})
                    .get("storage", {})
                    .get("record_storage_mode")
                )
                self.assertEqual(storage_settings, "memory")

    def test_directory_paths_are_materialized_during_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir) / "profile"
            profile_dir.mkdir()
            config_path = profile_dir / "config.json"

            storage_root = Path(tmp_dir) / "legacy_storage"
            payload = {
                "record_to_memory": False,
                "storage_root_dir": str(storage_root / "root"),
                "models_storage_dir": str(storage_root / "root" / "models"),
                "recordings_dir": str(storage_root / "recordings"),
                "deps_install_dir": str(storage_root / "deps"),
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            with isolated_config_environment(str(profile_dir)) as config_module:
                manager = config_module.ConfigManager(config_file=str(config_path))

                storage_root_path = Path(manager.config["storage_root_dir"])
                models_path = Path(manager.config["models_storage_dir"])
                recordings_path = Path(manager.config["recordings_dir"])
                deps_path = Path(manager.config["deps_install_dir"])

                for path in (storage_root_path, models_path, recordings_path, deps_path):
                    resolved = path.resolve()
                    self.assertTrue(resolved.exists())
                    self.assertTrue(resolved.is_dir())

                self.assertEqual(
                    storage_root_path,
                    (storage_root / "root").resolve(),
                )

                self.assertEqual(manager.config.get("record_storage_mode"), "disk")
                self.assertNotIn("record_to_memory", manager.config)

    def test_legacy_config_file_is_migrated_to_profile_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            legacy_dir = Path(tmp_dir) / "legacy"
            profile_dir = Path(tmp_dir) / "profile"
            legacy_dir.mkdir()
            profile_dir.mkdir()

            legacy_config = legacy_dir / "config.json"
            legacy_config.write_text(json.dumps({"record_to_memory": True}), encoding="utf-8")

            with isolated_config_environment(
                str(profile_dir), working_dir=str(legacy_dir)
            ) as config_module:
                manager = config_module.ConfigManager()
                expected_profile_config = profile_dir / "config.json"

                self.assertTrue(expected_profile_config.exists())
                self.assertFalse(legacy_config.exists())
                self.assertEqual(
                    Path(manager.config_file).resolve(), expected_profile_config.resolve()
                )
                self.assertEqual(manager.config.get("record_storage_mode"), "memory")
                self.assertNotIn("record_to_memory", manager.config)
                self.assertNotIn(
                    "record_to_memory",
                    expected_profile_config.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
