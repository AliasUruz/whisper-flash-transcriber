import builtins
import contextlib
import copy
import importlib
import json
import os
import sys
import tempfile
import threading
import unittest
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest import mock
from types import ModuleType, SimpleNamespace

import numpy as np

from src.model_manager import (
    HardwareProfile,
    build_runtime_catalog,
    select_recommended_model,
)
from src import state_manager as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = ModuleType("sounddevice")


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
    from src.core import AppCore
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager
    from src.model_manager import (
        HardwareProfile,
        build_runtime_catalog,
        select_recommended_model,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when running directly
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.core import AppCore
    from src.vad_manager import VADManager
    from src.keyboard_hotkey_manager import KeyboardHotkeyManager
    from src.model_manager import (
        HardwareProfile,
        build_runtime_catalog,
        select_recommended_model,
    )


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


class TestHardwareProfileFallback(unittest.TestCase):
    def test_fallback_populates_cuda_fields_when_torch_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_dir = Path(tmp_dir) / "profile"
            profile_dir.mkdir()
            config_path = profile_dir / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            with isolated_config_environment(profile_dir) as config_module:
                real_find_spec = importlib.util.find_spec
                real_import_module = importlib.import_module

                def fake_find_spec(name, *args, **kwargs):
                    if name == "torch":
                        return None
                    if name == "ctranslate2":
                        return SimpleNamespace()
                    if name == "pynvml":
                        return None
                    return real_find_spec(name, *args, **kwargs)

                def fake_import_module(name, *args, **kwargs):
                    if name == "ctranslate2":
                        return SimpleNamespace(get_cuda_device_count=lambda: 2)
                    return real_import_module(name, *args, **kwargs)

                def fake_check_output(cmd, *, encoding=None, stderr=None):
                    self.assertIn("nvidia-smi", cmd[0])
                    return "24576\n8192\n"

                with mock.patch(
                    "src.config_manager.importlib.util.find_spec",
                    side_effect=fake_find_spec,
                ), mock.patch(
                    "src.config_manager.importlib.import_module",
                    side_effect=fake_import_module,
                ), mock.patch(
                    "src.config_manager.subprocess.check_output",
                    side_effect=fake_check_output,
                ):
                    manager = config_module.ConfigManager(config_file=str(config_path))

        hardware = manager._runtime_hardware_profile
        self.assertTrue(hardware.has_cuda)
        self.assertEqual(hardware.gpu_count, 2)
        self.assertGreaterEqual(hardware.max_vram_mb, 24576)

        turbo_entry = next(
            entry
            for entry in manager._runtime_model_catalog
            if entry.get("id") == "openai/whisper-large-v3-turbo"
        )
        self.assertNotEqual(turbo_entry.get("hardware_status"), "blocked")


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


class TestKeyboardHotkeyManagerLifecycle(unittest.TestCase):
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

    def _make_manager(self) -> KeyboardHotkeyManager:
        return KeyboardHotkeyManager(config_file=self.config_path)

    def test_start_stop_roundtrip_unregisters_drivers(self):
        manager = self._make_manager()
        fake_driver = mock.Mock()
        fake_driver.name = "mock-driver"

        def register_side_effect():
            manager._drivers = [fake_driver]
            manager._active_driver = fake_driver
            manager._active_driver_index = 0
            manager.hotkey_handlers = {"record": ["handle-1"]}
            return True

        with mock.patch.object(
            manager,
            "_probe_available_driver_names",
            return_value=["mock-driver"],
        ), mock.patch.object(
            manager,
            "_resolve_primary_driver_name",
            return_value="mock-driver",
        ), mock.patch.object(
            manager,
            "_determine_fallback",
            return_value=False,
        ), mock.patch.object(
            manager,
            "_register_hotkeys",
            side_effect=register_side_effect,
        ):
            self.assertTrue(manager.start())

        self.assertTrue(manager.is_running)

        manager.stop()

        fake_driver.unregister.assert_called()
        self.assertEqual(
            self.keyboard_mock.unhook.call_args_list,
            [mock.call("handle-1")],
        )
        self.assertFalse(manager.is_running)
        self.assertEqual(manager.hotkey_handlers, {})

    def test_unregister_hotkeys_failure_is_reported(self):
        manager = self._make_manager()
        failing_driver = mock.Mock()
        failing_driver.name = "broken-driver"
        failing_driver.unregister.side_effect = RuntimeError("boom")
        manager._drivers = [failing_driver]
        manager._active_driver = failing_driver
        manager._active_driver_index = 0

        result = manager._unregister_hotkeys()

        self.assertFalse(result)
        failing_driver.unregister.assert_called_once()


class TestModelRecommendations(unittest.TestCase):
    LARGE_TURBO_ID = "openai/whisper-large-v3-turbo"

    def test_select_recommended_model_prefers_large_turbo_with_high_end_gpu(self):
        hardware = HardwareProfile(
            system_ram_mb=32768,
            has_cuda=True,
            gpu_count=1,
            max_vram_mb=12288,
        )
        runtime_catalog = build_runtime_catalog(hardware)
        recommended = select_recommended_model(runtime_catalog)

        self.assertIsNotNone(recommended)
        assert recommended is not None  # for type checkers
        self.assertEqual(recommended["id"], self.LARGE_TURBO_ID)
        self.assertEqual(recommended.get("ui_group"), "recommended")
        self.assertEqual(recommended.get("hardware_status"), "ok")

    def test_apply_runtime_overrides_auto_selects_large_turbo_on_gpu(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with isolated_config_environment(tmp_dir) as config_module:
                original_import_module = importlib.import_module
                original_find_spec = importlib.util.find_spec

                fake_cuda = SimpleNamespace(
                    is_available=lambda: True,
                    device_count=lambda: 1,
                    get_device_properties=lambda _: SimpleNamespace(
                        total_memory=13 * 1024 * 1024 * 1024
                    ),
                )

                def fake_find_spec(name: str, package: str | None = None):
                    if name == "torch":
                        return object()
                    return original_find_spec(name, package)

                def fake_import_module(name: str, package: str | None = None):
                    if name == "torch":
                        return SimpleNamespace(cuda=fake_cuda)
                    return original_import_module(name, package)

                with mock.patch("importlib.util.find_spec", side_effect=fake_find_spec):
                    with mock.patch(
                        "importlib.import_module",
                        side_effect=fake_import_module,
                    ):
                        with mock.patch(
                            "src.config_manager.get_total_memory_mb",
                            return_value=32768,
                        ):
                            manager = config_module.ConfigManager()

                recommendation = manager._runtime_recommendation
                self.assertIsNotNone(recommendation)
                assert recommendation is not None  # for type checkers
                self.assertEqual(recommendation["id"], self.LARGE_TURBO_ID)
                self.assertTrue(recommendation.get("auto_applied"))
        self.assertEqual(
            manager.config[config_module.ASR_MODEL_ID_CONFIG_KEY],
            self.LARGE_TURBO_ID,
        )


class TestCoreModuleImport(unittest.TestCase):
    def test_core_import_succeeds_with_minimal_dependencies(self):
        module_name = "src.core"

        fake_messagebox = SimpleNamespace(
            showinfo=mock.Mock(),
            showerror=mock.Mock(),
            showwarning=mock.Mock(),
            askyesno=mock.Mock(return_value=False),
        )

        fake_tk = ModuleType("tkinter")
        fake_tk.messagebox = fake_messagebox

        fake_sounddevice = ModuleType("sounddevice")

        class _DummyStream:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def start(self):
                return None

            def stop(self):
                return None

        fake_sounddevice.query_devices = mock.Mock(return_value={})
        fake_sounddevice.check_input_settings = mock.Mock(return_value=None)
        fake_sounddevice.InputStream = _DummyStream
        fake_sounddevice.OutputStream = _DummyStream
        fake_sounddevice.sleep = mock.Mock(return_value=None)
        fake_sounddevice.PortAudioError = type("PortAudioError", (Exception,), {})
        fake_sounddevice.CallbackStop = type("CallbackStop", (Exception,), {})

        fake_wizard = ModuleType("src.onboarding.first_run_wizard")

        class _DownloadProgressPanel:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                pass

        class _FirstRunWizard:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                pass

        class _WizardResult:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                self.config_updates = {}
                self.hotkey_preferences = {}
                self.download_request = None

        fake_wizard.DownloadProgressPanel = _DownloadProgressPanel
        fake_wizard.FirstRunWizard = _FirstRunWizard
        fake_wizard.WizardResult = _WizardResult

        original_core = sys.modules.pop(module_name, None)

        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.dict(
                        sys.modules,
                        {
                            "tkinter": fake_tk,
                            "tkinter.messagebox": fake_messagebox,
                            "sounddevice": fake_sounddevice,
                            "src.onboarding.first_run_wizard": fake_wizard,
                        },
                    )
                )

                module = importlib.import_module(module_name)

            self.assertIsNotNone(module)
        finally:
            sys.modules.pop(module_name, None)
            if original_core is not None:
                sys.modules[module_name] = original_core

class TestStateManagerOperationIdPropagation(unittest.TestCase):
    def test_set_state_uses_explicit_argument_when_payload_missing(self):
        manager = sm.StateManager(sm.STATE_IDLE)
        notifications: list[sm.StateNotification] = []
        manager.subscribe(notifications.append)

        manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_STARTED,
            details={"message": "Starting download"},
            source="unit-test",
            operation_id="test-op-123",
        )

        self.assertEqual(manager.get_current_state(), sm.STATE_LOADING_MODEL)
        self.assertEqual(len(notifications), 1)

        notification = notifications[0]
        self.assertEqual(notification.operation_id, "test-op-123")
        self.assertEqual(notification.state, sm.STATE_LOADING_MODEL)
        self.assertEqual(notification.previous_state, sm.STATE_IDLE)

    def test_set_state_payload_operation_id_overrides_argument(self):
        manager = sm.StateManager(sm.STATE_IDLE)
        notifications: list[sm.StateNotification] = []
        manager.subscribe(notifications.append)

        manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_STARTED,
            details={"message": "Starting download", "operation_id": "payload-op"},
            source="unit-test",
            operation_id="argument-op",
        )

        self.assertEqual(manager.get_current_state(), sm.STATE_LOADING_MODEL)
        self.assertEqual(len(notifications), 1)
        notification = notifications[0]
        self.assertEqual(notification.operation_id, "payload-op")

    def test_transition_if_respects_operation_id_precedence(self):
        manager = sm.StateManager(sm.STATE_IDLE)
        notifications: list[sm.StateNotification] = []
        manager.subscribe(notifications.append)

        transitioned = manager.transition_if(
            sm.STATE_IDLE,
            sm.StateEvent.MODEL_DOWNLOAD_STARTED,
            details={"message": "Starting download"},
            source="unit-test",
            operation_id="transition-op",
        )

        self.assertTrue(transitioned)
        self.assertEqual(manager.get_current_state(), sm.STATE_LOADING_MODEL)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[-1].operation_id, "transition-op")

        notifications.clear()
        manager.set_state(sm.StateEvent.MODEL_READY, source="unit-test")
        notifications.clear()

        transitioned = manager.transition_if(
            sm.STATE_IDLE,
            sm.StateEvent.MODEL_DOWNLOAD_STARTED,
            details={"message": "Starting download", "operation_id": "payload-op"},
            source="unit-test",
            operation_id="argument-op",
        )

        self.assertTrue(transitioned)
        self.assertEqual(manager.get_current_state(), sm.STATE_LOADING_MODEL)
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[-1].operation_id, "payload-op")


class TestStateManagerDuplicateSuppression(unittest.TestCase):
    def _create_manager(self) -> tuple[sm.StateManager, list[sm.StateNotification]]:
        manager = sm.StateManager(sm.STATE_IDLE)
        notifications: list[sm.StateNotification] = []
        manager.subscribe(notifications.append)
        return manager, notifications

    def test_transitions_with_different_details_are_not_suppressed(self):
        manager, notifications = self._create_manager()

        manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_PROGRESS,
            details={"message": "progress", "percent": 0.1},
            operation_id="download-1",
        )
        manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_PROGRESS,
            details={"message": "progress", "percent": 0.5},
            operation_id="download-1",
        )

        self.assertEqual(manager.get_current_state(), sm.STATE_LOADING_MODEL)
        self.assertEqual(len(notifications), 2)
        self.assertListEqual(
            [notification.details["percent"] for notification in notifications],
            [0.1, 0.5],
        )

    def test_identical_details_are_still_suppressed(self):
        manager, notifications = self._create_manager()

        manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_PROGRESS,
            details={"message": "progress", "percent": 0.3},
            operation_id="download-2",
        )
        manager.set_state(
            sm.StateEvent.MODEL_DOWNLOAD_PROGRESS,
            details={"percent": 0.3, "message": "progress"},
            operation_id="download-2",
        )

        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].details["percent"], 0.3)
        self.assertEqual(notifications[0].state, sm.STATE_LOADING_MODEL)


class TestHotkeyManagerReregistration(unittest.TestCase):
    def _build_core(self):
        core = AppCore.__new__(AppCore)
        core.keyboard_lock = threading.RLock()
        core.hotkey_lock = threading.RLock()
        core._cleanup_hotkeys = mock.Mock(name="cleanup")
        core._refresh_hotkey_driver_ui = mock.Mock(name="refresh_ui")
        core._log_status = mock.Mock(name="log_status")
        core.toggle_recording = mock.Mock(name="toggle_cb")
        core.start_recording = mock.Mock(name="start_cb")
        core.stop_recording_if_needed = mock.Mock(name="stop_cb")
        core.start_agent_command = mock.Mock(name="agent_cb")
        core.record_key = "f9"
        core.agent_key = "f10"
        core.record_mode = "toggle"
        core.ahk_running = False
        core.state_manager = mock.Mock(name="state_manager")

        manager = mock.create_autospec(KeyboardHotkeyManager, instance=True)
        manager.update_config.return_value = True
        manager.restart.return_value = True
        manager.get_active_driver_name.return_value = "keyboard"
        manager.is_using_fallback.return_value = False
        core.ahk_manager = manager
        return core, manager

    def test_reload_reuses_existing_manager_instance(self):
        core, manager = self._build_core()
        with mock.patch("src.core.time.sleep", return_value=None):
            result = AppCore._reload_keyboard_and_suppress(core)

        self.assertTrue(result)
        self.assertIs(core.ahk_manager, manager)
        manager.update_config.assert_called_once_with(
            record_key="f9", agent_key="f10", record_mode="toggle"
        )
        manager.set_callbacks.assert_called_once_with(
            toggle=core.toggle_recording,
            start=core.start_recording,
            stop=core.stop_recording_if_needed,
            agent=core.start_agent_command,
        )
        manager.restart.assert_called_once()
        core._cleanup_hotkeys.assert_called_once()
        core._refresh_hotkey_driver_ui.assert_called()
        core._log_status.assert_called_once()
        message, *_ = core._log_status.call_args[0]
        self.assertIn(core.record_key.upper(), message)
        self.assertTrue(core.ahk_running)

    def test_reload_failure_keeps_instance_and_reports_error(self):
        core, manager = self._build_core()
        manager.restart.side_effect = [False, False, False]

        with mock.patch("src.core.time.sleep", return_value=None):
            result = AppCore._reload_keyboard_and_suppress(core)

        self.assertFalse(result)
        self.assertIs(core.ahk_manager, manager)
        self.assertEqual(manager.restart.call_count, 3)
        self.assertEqual(manager.update_config.call_count, 3)
        self.assertEqual(manager.set_callbacks.call_count, 3)
        core._cleanup_hotkeys.assert_called_once()
        core._log_status.assert_not_called()
        self.assertFalse(core.ahk_running)


if __name__ == "__main__":
    unittest.main()
