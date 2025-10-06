import logging
import os
import sys
from contextlib import suppress

if os.name == "nt":
    import winreg

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "WhisperTranscriber"

def set_launch_at_startup(
    enable: bool, target: str | None = None, name: str = APP_NAME
) -> None:
    """Enable or disable launching the app at Windows startup."""
    if os.name != "nt":
        return
    if target is None:
        target = os.path.abspath(sys.argv[0])
    access_rights = winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY
    key_handle = None
    try:
        try:
            key_handle = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, RUN_KEY, 0, access_rights
            )
        except FileNotFoundError:
            key_handle = winreg.CreateKeyEx(
                winreg.HKEY_CURRENT_USER, RUN_KEY, 0, access_rights
            )

        if enable:
            winreg.SetValueEx(key_handle, name, 0, winreg.REG_SZ, f'"{target}"')
        else:
            with suppress(FileNotFoundError):
                winreg.DeleteValue(key_handle, name)
    except OSError:
        logging.exception("Failed to update autostart registry key.")
    finally:
        if key_handle is not None:
            winreg.CloseKey(key_handle)


def is_launch_at_startup_enabled(name: str = APP_NAME) -> bool:
    """Check if launch at startup is enabled."""
    if os.name != "nt":
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, name)
            return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logging.error(f"Failed to query autostart registry key: {e}")
        return False
