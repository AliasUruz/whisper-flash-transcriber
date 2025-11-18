import pystray
from pystray import MenuItem as item, Menu
import threading
import logging
from icons import create_icon

class SystemTray:
    def __init__(self, core_service, restore_callback, quit_callback):
        self.core = core_service
        self.restore_callback = restore_callback
        self.quit_callback = quit_callback
        self.icon = None
        self.running = False

    def run(self):
        """Starts the tray icon in a separate thread."""
        self.running = True
        threading.Thread(target=self._run_icon, daemon=True).start()

    def _run_icon(self):
        image = create_icon("idle")
        self.icon = pystray.Icon("whisper_flash", image, "Whisper Flash", self._create_menu())
        self.icon.run()

    def _create_menu(self):
        # Determine label based on core state
        # Note: We access core.state directly. Ensure thread safety if needed, 
        # but for reading a string it's generally fine in this context.
        is_recording = self.core.state == "recording"
        toggle_label = "Stop & Transcribe" if is_recording else "Start Recording"
        
        return Menu(
            item(toggle_label, self._on_toggle_recording, default=True),
            item('Settings', self._on_settings),
            item('Exit', self._on_exit)
        )

    def update_state(self, state):
        """Updates the icon and menu based on state."""
        try:
            if self.icon:
                self.icon.icon = create_icon(state)
                # Refresh menu to update label
                self.icon.menu = self._create_menu()
        except Exception as e:
            logging.error(f"Tray update failed: {e}")

    def stop(self):
        if self.icon:
            self.icon.stop()
        self.running = False

    def _on_toggle_recording(self, icon, item):
        self.core.toggle_recording()

    def _on_settings(self, icon, item):
        if self.restore_callback:
            self.restore_callback()

    def _on_exit(self, icon, item):
        if self.quit_callback:
            self.quit_callback()
