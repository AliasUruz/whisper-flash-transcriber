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
    page.window_width = 600 # Increased width for better visibility
    page.window_height = 850 # Adjusted for generous space
    page.scroll = ft.ScrollMode.AUTO # Enable auto-scroll to prevent cutting content
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0 # Edge-to-edge content
    page.bgcolor = "#202028" # Mica Alt Dark background
    page.window_resizable = True

    # Initialization
    tray = None # Initialize early to prevent UnboundLocalError in cleanup
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
    
    # Shutdown Control (Defined early to be available for Tray)
    def cleanup_and_exit():
        logging.info("App shutting down...")
        try:
            # Trigger auto-save to capture any pending edits (e.g. focused field)
            if ui: ui._trigger_auto_save()

            ui.update_status("shutdown", "Closing app...")
            page.update()
        except Exception:
            pass
            
        # Fast exit: Don't wait for threads, just kill process
        if tray: 
            try:
                tray.stop()
            except Exception: pass
            
        page.window_destroy()
        time.sleep(0.5) # Give Flet time to send the destroy command
        os._exit(0)

    # System Tray Integration
    def restore_window():
        page.window_visible = True
        page.update()
        # Refresh UI in case settings changed via Tray while minimized
        ui.refresh_ui_from_settings()

    def quit_app():
        cleanup_and_exit()

    def set_ai_model(model_name):
        core.settings["gemini_enabled"] = True
        core.settings["gemini_model"] = model_name
        core.save_settings()
        ui.refresh_ui_from_settings()
        tray.update_menu()

    def toggle_ai_off():
        core.settings["gemini_enabled"] = False
        core.save_settings()
        ui.refresh_ui_from_settings()
        tray.update_menu()

    # Custom Menu Builder for Tray
    def create_tray_menu():
        import pystray
        from PIL import Image
        
        # Helper to check state for radio buttons
        def is_checked(item):
            if item.text == "Off":
                return not core.settings.get("gemini_enabled", False)
            elif item.text == "Gemini 2.5 Flash Lite":
                return core.settings.get("gemini_enabled") and core.settings.get("gemini_model") == "gemini-2.5-flash-lite"
            elif item.text == "Gemini 2.5 Flash":
                return core.settings.get("gemini_enabled") and core.settings.get("gemini_model") == "gemini-2.5-flash"
            return False

        return pystray.Menu(
            pystray.MenuItem("Open", restore_window, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("AI Correction", pystray.Menu(
                pystray.MenuItem("Off", lambda: toggle_ai_off(), checked=is_checked, radio=True),
                pystray.MenuItem("Gemini 2.5 Flash Lite", lambda: set_ai_model("gemini-2.5-flash-lite"), checked=is_checked, radio=True),
                pystray.MenuItem("Gemini 2.5 Flash", lambda: set_ai_model("gemini-2.5-flash"), checked=is_checked, radio=True)
            )),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", restore_window),
            pystray.MenuItem("Exit", quit_app)
        )

    # Initialize Tray with custom menu factory
    tray = SystemTray(core, restore_window, quit_app, menu_factory=create_tray_menu)
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

    def window_event(e):
        if e.data == "close":
            cleanup_and_exit()
        elif e.data == "minimize":
            page.window_visible = False
            page.update()
            # Ensure settings are saved on minimize
            ui._trigger_auto_save()

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

    # Start mouse handler if enabled in settings
    if core.settings.get("mouse_hotkey", False):
        core.mouse_handler.start_listening()

    ui.update_status("transcribing", "Booting engine...")
    page.update()

if __name__ == "__main__":
    ft.app(target=main)
