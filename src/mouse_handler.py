from pynput import mouse
import logging
import threading
import time

class MouseHandler:
    def __init__(self, core):
        self.core = core
        self.listener = None
        self.pressed_buttons = set()
        self.trigger_active = False
        logging.info("MouseHandler initialized.")

    def _on_click(self, x, y, button, pressed):
        if pressed:
            self.pressed_buttons.add(button)
            # Check for Chord (LMB + RMB)
            if mouse.Button.left in self.pressed_buttons and mouse.Button.right in self.pressed_buttons:
                if not self.trigger_active:
                    logging.info("Mouse Chord Triggered (LMB + RMB)")
                    self.trigger_active = True
                    # Run in separate thread to avoid blocking the listener
                    threading.Thread(target=self._safe_toggle, daemon=True).start()
        else:
            if button in self.pressed_buttons:
                self.pressed_buttons.remove(button)
            self.trigger_active = False

    def _safe_toggle(self):
        try:
            self.core.toggle_recording()
        except Exception as e:
            logging.error(f"Mouse toggle failed: {e}")

    def start_listening(self):
        if self.listener: return
        
        logging.info("Starting mouse listener...")
        try:
            self.listener = mouse.Listener(on_click=self._on_click)
            self.listener.start()
        except Exception as e:
            logging.error(f"Failed to start mouse listener: {e}")
            self.listener = None

    def stop_listening(self):
        if self.listener:
            logging.info("Stopping mouse listener...")
            try:
                self.listener.stop()
            except Exception: pass
            self.listener = None
            self.pressed_buttons.clear()
            self.trigger_active = False
