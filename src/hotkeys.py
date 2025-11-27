import keyboard
import logging
import threading
import time

class HotkeyManager:
    def __init__(self, core):
        self.core = core
        self.current_hotkey = None
        self.error_callback = None
        logging.info("HotkeyManager initialized (Keyboard Lib Edition).")

    def _on_activate(self):
        try:
            self.core.toggle_recording()
        except Exception as e:
            logging.error(f"Hotkey action failed: {e}")

    def start_listening(self):
        # Get hotkey from settings
        hk_str = self.core.settings.get('hotkey', 'f3')
        if not hk_str: return

        # Avoid re-binding if same
        if self.current_hotkey == hk_str:
            return

        self.stop_listening() # Clean up old

        logging.info(f"Binding hotkey: '{hk_str}' with suppression...")
        try:
            # suppress=True prevents the key from reaching other apps
            keyboard.add_hotkey(hk_str, self._on_activate, suppress=True)
            self.current_hotkey = hk_str
            logging.info("Hotkey bound successfully.")
        except Exception as e:
            msg = f"Bind failed for '{hk_str}': {e}"
            logging.error(msg)
            self.current_hotkey = None
            if self.error_callback:
                self.error_callback("Hotkey Error", f"Could not bind '{hk_str}'.\nReason: {e}")

    def set_error_callback(self, callback):
        self.error_callback = callback

    def stop_listening(self):
        if self.current_hotkey:
            logging.info(f"Unbinding hotkey: {self.current_hotkey}")
            try:
                keyboard.remove_hotkey(self.current_hotkey)
            except Exception: pass
            self.current_hotkey = None

    def restart_listening(self):
        logging.info("Restarting listener...")
        self.stop_listening()
        time.sleep(0.1)
        self.start_listening()
