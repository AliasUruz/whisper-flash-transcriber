from pynput import keyboard
import logging
import threading
import time

class HotkeyManager:
    def __init__(self, core):
        self.core = core
        self.listener = None
        self.error_callback = None
        logging.info("HotkeyManager initialized.")

    def _get_pynput_hotkey_string(self):
        raw = self.core.settings.get('hotkey', 'f3')
        if not raw: return None

        parts = raw.lower().split('+')
        formatted = []
        
        map_keys = {
            'ctrl': '<ctrl>', 'control': '<ctrl>', 'lctrl': '<ctrl_l>', 'rctrl': '<ctrl_r>',
            'alt': '<alt>', 'lalt': '<alt_l>', 'ralt': '<alt_gr>',
            'shift': '<shift>', 'lshift': '<shift>', 'rshift': '<shift_r>',
            'cmd': '<cmd>', 'win': '<cmd>', 'command': '<cmd>',
            'esc': '<esc>', 'enter': '<enter>', 'space': '<space>', 'tab': '<tab>'
        }

        for part in parts:
            part = part.strip()
            if not part: continue
            
            if part in map_keys:
                formatted.append(map_keys[part])
            elif len(part) > 1: 
                formatted.append(f'<{part}>')
            else:
                formatted.append(part)
        
        result = '+'.join(formatted)
        logging.info(f"Parsed hotkey: '{result}'")
        return result

    def _on_activate(self):
        try:
            self.core.toggle_recording()
        except Exception as e:
            logging.error(f"Hotkey action failed: {e}")

    def start_listening(self):
        if self.listener: return

        hk_str = self._get_pynput_hotkey_string()
        if not hk_str: return

        try:
            self.listener = keyboard.GlobalHotKeys({hk_str: self._on_activate})
            self.listener.start()
            logging.info("Hotkey listener started.")
        except Exception as e:
            msg = f"Bind failed for '{hk_str}': {e}"
            logging.error(msg)
            self.listener = None
            if self.error_callback:
                self.error_callback("Hotkey Error", f"Could not bind '{hk_str}'.\nReason: {e}")

    def set_error_callback(self, callback):
        self.error_callback = callback

    def stop_listening(self):
        if self.listener:
            logging.info("Stopping listener...")
            try:
                self.listener.stop()
            except Exception: pass
            self.listener = None

    def restart_listening(self):
        logging.info("Restarting listener...")
        self.stop_listening()
        time.sleep(0.3) # Espera OS liberar
        self.start_listening()
