import flet as ft
import threading
import logging
import sys
import os
import time
from ui import AppUI
from core import CoreService
from hotkeys import HotkeyManager
from tray import SystemTray

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main(page: ft.Page):
    page.title = "Whisper Flash Transcriber"
    page.window_width = 420
    page.window_height = 480 # Adjusted to fit content tightly
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 20
    page.window_resizable = True

    # Initialization
    core = CoreService()
    
    # Check first run logic
    if not core.settings.get("first_run", True):
        page.window_visible = False
    else:
        # If first run, update setting so next time it starts hidden
        core.settings["first_run"] = False
        core.save_settings()

    ui = AppUI(page, core)
    hotkey_manager = HotkeyManager(core)
    
    # System Tray Integration
    def restore_window():
        page.window_visible = True
        page.update()

    def quit_app():
        cleanup_and_exit()

    tray = SystemTray(core, restore_window, quit_app)
    tray.run()

    # Wiring
    def update_status_wrapper(status, tooltip):
        ui.update_status(status, tooltip)
        tray.update_state(status)

    core.set_ui_update_callback(update_status_wrapper)
    core.set_error_popup_callback(ui.show_error_popup)
    core.set_hotkey_manager(hotkey_manager)
    
    # New wiring for bug fixes
    hotkey_manager.set_error_callback(ui.show_error_popup)

    # Shutdown Control
    def cleanup_and_exit():
        logging.info("App shutting down...")
        try:
            ui.update_status("shutdown", "Closing app...")
            page.update()
        except Exception:
            pass
            
        # Fast exit: Don't wait for threads, just kill process
        # if hotkey_manager: hotkey_manager.stop_listening() # Too slow
        # if core: core.shutdown() # Too slow
        tray.stop()
        page.window_destroy()
        time.sleep(0.5) # Give Flet time to send the destroy command
        os._exit(0)

    def window_event(e):
        if e.data == "close":
            cleanup_and_exit()
        elif e.data == "minimize":
            page.window_visible = False
            page.update()

    page.window_prevent_close = True
    page.on_window_event = window_event
    
    # Fix for Zombie Process: Ensure UI exit button triggers full cleanup
    ui.set_exit_callback(cleanup_and_exit)

    page.add(ui.build_controls())

    if ui.tray_supported:
        page.window_visible = False
    else:
        page.window_center()

    # Start Threads
    threading.Thread(target=core.load_model_async, daemon=True, name="Loader").start()
    hotkey_manager.start_listening()

    ui.update_status("transcribing", "Booting engine...")
    page.update()

if __name__ == "__main__":
    ft.app(target=main)
