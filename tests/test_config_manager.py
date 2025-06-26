import os
import sys
import time
import os, sys
import json
import pytest
from unittest import mock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src import config_manager


def test_skip_save_when_unchanged(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))
    cm = config_manager.ConfigManager(
        config_file=str(cfg_path),
        default_config=config_manager.DEFAULT_CONFIG,
    )

    mtime_cfg = os.path.getmtime(cfg_path)
    if secrets_path.exists():
        mtime_sec = os.path.getmtime(secrets_path)
    else:
        mtime_sec = None

    time.sleep(1)
    cm.save_config()

    assert os.path.getmtime(cfg_path) == mtime_cfg
    if mtime_sec is not None:
        assert os.path.getmtime(secrets_path) == mtime_sec
    else:
        assert not secrets_path.exists()


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("false", False), (1, True), (0, False)],
)
def test_parse_bool_values(tmp_path, monkeypatch, value, expected):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    config = {
        "auto_paste": value,
        "display_transcripts_in_terminal": value,
        "save_temp_recordings": value,
        "use_vad": value,
    }

    cfg_path.write_text(json.dumps(config))
    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))
    cm = config_manager.ConfigManager(
        config_file=str(cfg_path),
        default_config=config_manager.DEFAULT_CONFIG,
    )

    assert cm.get("auto_paste") is expected
    assert cm.get(config_manager.DISPLAY_TRANSCRIPTS_KEY) is expected
    assert cm.get(config_manager.SAVE_TEMP_RECORDINGS_CONFIG_KEY) is expected
    assert cm.get(config_manager.USE_VAD_CONFIG_KEY) is expected


def test_config_validation_and_fallback(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    secrets_path = tmp_path / "secrets.json"

    invalid_config = {
        "record_mode": "invalid_mode",
        "sound_frequency": "not_a_number",
        "sound_duration": "abc",
        "sound_volume": 2.0, # Out of range
        "batch_size": 0, # Invalid value
        "manual_batch_size": -5, # Invalid value
        "gpu_index": -2, # Invalid value
        "min_transcription_duration": 0.05, # Out of range
        "vad_threshold": "invalid",
        "vad_silence_duration": 0.0, # Out of range
        "hotkey_stability_service_enabled": "not_a_bool",
        "text_correction_enabled": "not_a_bool",
        "text_correction_service": "unknown_service",
        "gemini_model_options": "not_a_list",
        "min_record_duration": -1.0,
        "keyboard_library": "unknown_lib",
        "openrouter_model": 123,
        "gemini_api_key": 456,
        "prompt_agentico": 789,
    }

    cfg_path.write_text(json.dumps(invalid_config))
    monkeypatch.setattr(config_manager, "SECRETS_FILE", str(secrets_path))
    
    # Mock logging.warning to capture warnings
    with mock.patch('logging.warning') as mock_warning:
        cm = config_manager.ConfigManager(
            config_file=str(cfg_path),
            default_config=config_manager.DEFAULT_CONFIG,
        )

        # Assertions for record_mode
        assert cm.get("record_mode") == config_manager.DEFAULT_CONFIG["record_mode"]
        mock_warning.assert_any_call(f"Invalid record_mode 'invalid_mode'. Falling back to '{config_manager.DEFAULT_CONFIG['record_mode']}'.")

        # Assertions for sound settings
        assert cm.get("sound_frequency") == config_manager.DEFAULT_CONFIG["sound_frequency"]
        assert cm.get("sound_duration") == config_manager.DEFAULT_CONFIG["sound_duration"]
        assert cm.get("sound_volume") == config_manager.DEFAULT_CONFIG["sound_volume"]
        mock_warning.assert_any_call(f"Invalid sound_frequency value 'not_a_number' in config. Using default ({config_manager.DEFAULT_CONFIG['sound_frequency']}).")
        mock_warning.assert_any_call(f"Invalid sound_duration value 'abc' in config. Using default ({config_manager.DEFAULT_CONFIG['sound_duration']}).")
        mock_warning.assert_any_call(f"Invalid sound_volume value '2.0'. Must be between 0.0 and 1.0. Using default ({config_manager.DEFAULT_CONFIG['sound_volume']}).")

        # Assertions for batch_size settings
        assert cm.get("batch_size") == config_manager.DEFAULT_CONFIG["batch_size"]
        assert cm.get("manual_batch_size") == config_manager.DEFAULT_CONFIG["manual_batch_size"]
        mock_warning.assert_any_call(f"Invalid batch_size value '0'. Must be positive. Using default ({config_manager.DEFAULT_CONFIG['batch_size']}).")
        mock_warning.assert_any_call(f"Invalid manual_batch_size value '-5'. Must be positive. Using default ({config_manager.DEFAULT_CONFIG['manual_batch_size']}).")

        # Assertions for gpu_index
        assert cm.get("gpu_index") == -1 # Should fall back to auto-select
        mock_warning.assert_any_call(f"Invalid GPU index '-2'. Must be -1 (auto) or >= 0. Using auto (-1).")

        # Assertions for min_transcription_duration
        assert cm.get("min_transcription_duration") == config_manager.DEFAULT_CONFIG["min_transcription_duration"]
        mock_warning.assert_any_call(f"Invalid min_transcription_duration '0.05'. Must be between 0.1 and 10.0. Using default ({config_manager.DEFAULT_CONFIG['min_transcription_duration']}).")

        # Assertions for VAD settings
        assert cm.get("vad_threshold") == config_manager.DEFAULT_CONFIG["vad_threshold"]
        assert cm.get("vad_silence_duration") == config_manager.DEFAULT_CONFIG["vad_silence_duration"]
        mock_warning.assert_any_call(f"Invalid vad_threshold value 'invalid' in config. Using default ({config_manager.DEFAULT_CONFIG['vad_threshold']}).")
        mock_warning.assert_any_call(f"Invalid vad_silence_duration '0.0'. Must be >= 0.1. Using default ({config_manager.DEFAULT_CONFIG['vad_silence_duration']}).")

        # Assertions for boolean flags
        assert cm.get("hotkey_stability_service_enabled") == config_manager.DEFAULT_CONFIG["hotkey_stability_service_enabled"]
        assert cm.get("text_correction_enabled") == config_manager.DEFAULT_CONFIG["text_correction_enabled"]

        # Assertions for text_correction_service
        assert cm.get("text_correction_service") == config_manager.DEFAULT_CONFIG["text_correction_service"]
        mock_warning.assert_any_call(f"Invalid text_correction_service 'unknown_service'. Using default ('{config_manager.DEFAULT_CONFIG['text_correction_service']}').")

        # Assertions for gemini_model_options
        assert cm.get("gemini_model_options") == config_manager.DEFAULT_CONFIG["gemini_model_options"]
        mock_warning.assert_any_call(f"Invalid gemini_model_options. Must be a list. Using default.")

        # Assertions for min_record_duration
        assert cm.get("min_record_duration") == config_manager.DEFAULT_CONFIG["min_record_duration"]
        mock_warning.assert_any_call(f"Invalid min_record_duration '-1.0'. Must be non-negative. Using default ({config_manager.DEFAULT_CONFIG['min_record_duration']}).")

        # Assertions for keyboard_library
        assert cm.get("keyboard_library") == config_manager.DEFAULT_CONFIG["keyboard_library"]
        mock_warning.assert_any_call(f"Invalid keyboard_library 'unknown_lib'. Using default ('{config_manager.DEFAULT_CONFIG['keyboard_library']}').")

        # Assertions for AI provider models and keys
        assert cm.get("openrouter_model") == config_manager.DEFAULT_CONFIG["openrouter_model"]
        assert cm.get("gemini_api_key") == "" # Should be empty string due to invalid type
        assert cm.get("prompt_agentico") == config_manager.DEFAULT_CONFIG["prompt_agentico"]
        mock_warning.assert_any_call(f"Invalid type for 'openrouter_model'. Must be a string. Using default.")
        mock_warning.assert_any_call(f"Invalid type for 'gemini_api_key'. Must be a string. Using empty string.")
        mock_warning.assert_any_call(f"Invalid type for 'prompt_agentico'. Must be a string. Using default.")
