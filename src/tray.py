import pystray
from pystray import MenuItem as item, Menu
import threading
import logging
from icons import create_icon

class SystemTray:
    def __init__(self, core_service, restore_callback, quit_callback, menu_factory=None):
        self.core = core_service
        self.restore_callback = restore_callback
        self.quit_callback = quit_callback
        self.menu_factory = menu_factory
        self.icon = None
        self.running = False

    def run(self):
        """Starts the tray icon in a separate thread."""
        self.running = True
        threading.Thread(target=self._run_icon, daemon=True).start()

    def _run_icon(self):
        image = create_icon("idle")
        menu = self.menu_factory() if self.menu_factory else self._create_default_menu()
        self.icon = pystray.Icon("whisper_flash", image, "Whisper Flash", menu)
        self.icon.run()

    def _create_default_menu(self):
        # Determine label based on core state
        # core.state is Enum, so checking .value is safer if type is not guaranteed
        current_val = self.core.state.value if hasattr(self.core.state, 'value') else str(self.core.state)
        is_recording = current_val == "recording"
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
                if self.menu_factory:
                    self.icon.menu = self.menu_factory()
                else:
                    self.icon.menu = self._create_default_menu()
        except Exception as e:
            logging.error(f"Tray update failed: {e}")

    def update_menu(self):
        """Force menu refresh (e.g. when settings change)."""
        try:
            if self.icon and self.menu_factory:
                self.icon.menu = self.menu_factory()
        except Exception as e:
            logging.error(f"Menu update failed: {e}")

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
