import flet as ft
import threading
import logging
import sys
import os
import time
from ui import AppUI
from core import CoreService, AppState
from hotkeys import HotkeyManager
from tray import SystemTray
from settings import VALID_MODELS

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main(page: ft.Page):
    page.title = "Whisper Flash"
    page.window_width = 460 # Slightly wider for tabs
    page.window_height = 650 # Taller for modern UI
    page.scroll = ft.ScrollMode.AUTO 
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0 
    page.bgcolor = "#202028" 
    page.window_resizable = True

    # Initialization
    tray = None 
    core = CoreService()
    
    # Check first run logic - Object Access
    if not core.settings.first_run:
        page.window_visible = False
    else:
        # If first run, update setting so next time it starts hidden
        core.save_settings({"first_run": False})

    ui = AppUI(page, core)
    hotkey_manager = HotkeyManager(core)
    
    # Shutdown Control
    def cleanup_and_exit():
        logging.info("App shutting down...")
        try:
            # We don't need ui._trigger_auto_save() anymore as new tabs use active on_change listeners
            
            if page:
                ui.update_status("shutdown", "Closing app...")
                try: page.update()
                except Exception: pass
        except Exception as e:
            logging.error(f"UI Cleanup error: {e}")
            
        if core: 
            logging.info("Stopping Core...")
            core.shutdown()

        if tray: 
            try: tray.stop()
            except Exception: pass
        
        try: page.window_destroy()
        except Exception: pass
        
        time.sleep(0.5)
        logging.info("Bye.")
        os._exit(0)

    # System Tray Integration
    def restore_window():
        page.window_visible = True
        page.update()
        ui.refresh_ui_from_settings()

    def quit_app():
        cleanup_and_exit()

    def set_ai_model(model_name):
        # Update via method to ensure persistence
        core.save_settings({
            "gemini_enabled": True,
            "gemini_model": model_name
        })
        ui.refresh_ui_from_settings()
        tray.update_menu()

    def toggle_ai_off():
        core.save_settings({"gemini_enabled": False})
        ui.refresh_ui_from_settings()
        tray.update_menu()

    # Custom Menu Builder for Tray
    def create_tray_menu():
        import pystray
        
        def is_checked(item):
            if item.text == "Off":
                return not core.settings.gemini_enabled
            
            current_model = core.settings.gemini_model
            is_enabled = core.settings.gemini_enabled
            
            if not is_enabled: return False
            
            display_name = current_model.replace("-", " ").title()
            return item.text == display_name

        ai_menu_items = []
        
        def make_model_callback(m_id):
            return lambda icon, item: set_ai_model(m_id)

        ai_menu_items.append(
            pystray.MenuItem(
                "Off", 
                lambda icon, item: toggle_ai_off(), 
                checked=is_checked, 
                radio=True
            )
        )
        
        for model in VALID_MODELS:
            display_name = model.replace("-", " ").title()
            ai_menu_items.append(
                pystray.MenuItem(
                    display_name, 
                    make_model_callback(model), 
                    checked=is_checked, 
                    radio=True
                )
            )

        return pystray.Menu(
            pystray.MenuItem("Open", restore_window, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("AI Correction", pystray.Menu(*ai_menu_items)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", restore_window),
            pystray.MenuItem("Exit", quit_app)
        )

    # Initialize Tray
    tray = SystemTray(core, restore_window, quit_app, menu_factory=create_tray_menu)
    tray.run()
    ui.tray_supported = True

    # Wiring
    def update_status_wrapper(status, tooltip):
        # status comes as Enum string value from CoreService
        ui.update_status(status, tooltip)
        tray.update_state(status)

    core.set_ui_update_callback(update_status_wrapper)
    core.set_error_popup_callback(ui.show_error_popup)
    core.set_hotkey_manager(hotkey_manager)
    hotkey_manager.set_error_callback(ui.show_error_popup)

    def window_event(e):
        if e.data == "close":
            cleanup_and_exit()
        elif e.data == "minimize":
            page.window_visible = False
            page.update()

    page.window_prevent_close = True
    page.on_window_event = window_event
    
    ui.set_exit_callback(cleanup_and_exit)
    page.add(ui.build_controls())

    if ui.tray_supported:
        page.window_visible = False
    else:
        page.window_center()

    # Start Threads
    threading.Thread(target=core.load_model_async, daemon=True, name="Loader").start()
    hotkey_manager.start_listening()

    if core.settings.mouse_hotkey:
        core.mouse_hook.start()

    ui.update_status(AppState.LOADING.value, "Booting engine...")
    page.update()

if __name__ == "__main__":
    ft.app(target=main)
