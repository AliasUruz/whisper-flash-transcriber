import builtins
import contextlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@contextlib.contextmanager
def isolated_config_environment(profile_dir: str, *, working_dir: str | None = None):
    """Context manager that reloads ``src.config_manager`` with isolated paths."""

    module_name = "src.config_manager"
    previous_cwd = Path.cwd()
    previous_profile = os.environ.get("WHISPER_FLASH_PROFILE_DIR")

    try:
        os.environ["WHISPER_FLASH_PROFILE_DIR"] = profile_dir
        if working_dir is not None:
            os.chdir(working_dir)
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        yield module
    finally:
        if previous_profile is None:
            os.environ.pop("WHISPER_FLASH_PROFILE_DIR", None)
        else:
            os.environ["WHISPER_FLASH_PROFILE_DIR"] = previous_profile
        if working_dir is not None:
            os.chdir(previous_cwd)
        sys.modules.pop(module_name, None)

if not hasattr(builtins, "Mapping"):
    builtins.Mapping = Mapping


@contextmanager
def isolated_config_environment(profile_dir: str, working_dir: str | None = None):
    original_env = os.environ.get("WHISPER_FLASH_PROFILE_DIR")
    original_cwd = os.getcwd()
    try:
        os.environ["WHISPER_FLASH_PROFILE_DIR"] = str(profile_dir)
        if working_dir is not None:
            os.chdir(working_dir)
        if "src.config_manager" in sys.modules:
            del sys.modules["src.config_manager"]
        config_module = importlib.import_module("src.config_manager")
        try:
            yield config_module
        finally:
            if working_dir is not None:
                os.chdir(original_cwd)
    finally:
        if original_env is None:
            os.environ.pop("WHISPER_FLASH_PROFILE_DIR", None)
        else:
            os.environ["WHISPER_FLASH_PROFILE_DIR"] = original_env
        if "src.config_manager" in sys.modules:
            del sys.modules["src.config_manager"]

try:
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager


@contextmanager
def isolated_config_environment(
    profile_dir: str | os.PathLike[str],
    *,
    working_dir: str | os.PathLike[str] | None = None,
):
    """Load ``ConfigManager`` inside an isolated profile directory."""

    profile_path = Path(profile_dir).expanduser().resolve()
    working_path = (
        Path(working_dir).expanduser().resolve() if working_dir is not None else profile_path
    )

    profile_path.mkdir(parents=True, exist_ok=True)

    original_env = os.environ.get("WHISPER_FLASH_PROFILE_DIR")
    original_cwd = Path.cwd()
    module_name = "src.config_manager"
    cached_module = sys.modules.pop(module_name, None)

    try:
        os.environ["WHISPER_FLASH_PROFILE_DIR"] = str(profile_path)
        os.chdir(str(working_path))
        module = importlib.import_module(module_name)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        if cached_module is not None:
            sys.modules[module_name] = cached_module
        elif module_name in sys.modules:
            del sys.modules[module_name]

        if original_env is not None:
            os.environ["WHISPER_FLASH_PROFILE_DIR"] = original_env
        else:
            os.environ.pop("WHISPER_FLASH_PROFILE_DIR", None)

        os.chdir(str(original_cwd))


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
