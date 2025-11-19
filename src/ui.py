import flet as ft
from core import CoreService
import logging
import sys
from pathlib import Path

# Path fix for PyInstaller (sys._MEIPASS)
def get_asset_path(filename):
    if hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent
    return str(base / "assets" / filename)

ICON_MAP = {
    "idle": get_asset_path("icon_idle.ico"),
    "recording": get_asset_path("icon_recording.ico"),
    "transcribing": get_asset_path("icon_transcribing.ico"),
    "error": get_asset_path("icon_error.ico"),
}
DEFAULT_ICON = get_asset_path("icon.ico")

STATUS_COLOR_MAP = {
    "idle": ft.colors.GREEN_400,
    "recording": ft.colors.RED_400,
    "transcribing": ft.colors.PURPLE_400,
    "error": ft.colors.ORANGE_400,
    "shutdown": ft.colors.GREY_500,
}

class AppUI:
    def __init__(self, page: ft.Page, core: CoreService):
        self.page = page
        self.core = core
        self.page.title = "Whisper Flash"
        
        self.hotkey_field: ft.TextField | None = None
        self.mouse_hotkey_switch: ft.Switch | None = None
        self.auto_paste_switch: ft.Switch | None = None
        self.mic_dropdown: ft.Dropdown | None = None
        self.model_path_field: ft.TextField | None = None
        self.model_path_field: ft.TextField | None = None
        
        # Tray is now handled by pystray in main.py
        self.tray_supported = False 

    def build_controls(self) -> ft.Column:
        # Hotkey Field
        self.hotkey_field = ft.TextField(
            label="Global Hotkey",
            value=self.core.settings.get("hotkey", "f3"),
            hint_text="ex: ctrl+f3, alt+space",
            width=250, text_size=14
        )
        
        # Microphone Selector
        devices = self.core.get_audio_devices()
        current_mic = self.core.settings.get("input_device_index")
        
        mic_options = [ft.dropdown.Option(key=str(d['id']), text=d['name']) for d in devices]
        
        self.mic_dropdown = ft.Dropdown(
            label="Microphone (Optional)",
            options=mic_options,
            value=str(current_mic) if current_mic is not None else None,
            width=250,
            text_size=12,
            on_change=lambda e: self._trigger_auto_save()
        )

        # Auto Paste Switch
        self.auto_paste_switch = ft.Switch(
            label="Auto-paste result", 
            value=self.core.settings.get("auto_paste", True),
            active_color=ft.colors.GREEN_400,
            on_change=lambda e: self._trigger_auto_save()
        )

        # Mouse Hotkey Switch
        self.mouse_hotkey_switch = ft.Switch(
            label="Mouse Shortcut (LMB + RMB)", 
            value=self.core.settings.get("mouse_hotkey", False),
            active_color=ft.colors.BLUE_400,
            on_change=lambda e: self._trigger_auto_save()
        )

        # Model Path Field
        self.model_path_field = ft.TextField(
            label="Custom Model Path (Optional)",
            value=self.core.settings.get("model_path", ""),
            hint_text="e.g. D:\\WhisperModels",
            width=250, text_size=12,
            on_blur=lambda e: self._trigger_auto_save()
        )
        
        settings_col = ft.Column(
            [
                ft.Text("Configuration", size=18, weight=ft.FontWeight.BOLD),
                self.hotkey_field,
                self.mic_dropdown,
                self.auto_paste_switch,
                self.mouse_hotkey_switch,
                ft.Divider(),
                ft.Divider(),
                self.model_path_field,
                ft.Text("Note: Restart app if hotkey fails to register.", size=11, color=ft.colors.GREY_400),
            ],
            spacing=10
        )

        # Simplified Main Layout
        main_content = ft.Column(
            [
                ft.Text("Whisper Flash", size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                settings_col,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            expand=True
        )

        return ft.Container(content=main_content, padding=10, expand=True)

    def update_status(self, status: str, tooltip: str):
        if not self.page: return
        icon_path = ICON_MAP.get(status, DEFAULT_ICON)
        color = STATUS_COLOR_MAP.get(status, ft.colors.GREY_700)

        try:
            if self.tray_supported and self.tray_icon:
                self.tray_icon.tooltip = f"WF: {tooltip}"
                self.tray_icon.icon = icon_path
                self.tray_icon.update()
                
            # Status container removed from UI
        except Exception as e:
            logging.debug(f"UI update skipped: {e}")

    def show_error_popup(self, title: str, message: str):
        if not self.page: return
        def close_dialog(e):
            try:
                self.page.dialog.open = False
                self.page.update()
            except Exception: pass

        self.page.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, color=ft.colors.RED),
            content=ft.Text(message, selectable=True),
            actions=[ft.TextButton("OK", on_click=close_dialog)],
        )
        self.page.dialog.open = True
        try: self.page.update()
        except Exception: pass

    def _restore_window(self, e):
        try:
            self.page.window_visible = True
            self.page.window_minimized = False
            self.page.update()
        except Exception: pass

    def _trigger_auto_save(self):
        """Saves settings silently."""
        try:
            new_hotkey = self.hotkey_field.value.strip()
            new_mic = int(self.mic_dropdown.value) if self.mic_dropdown.value else None
            new_paste = self.auto_paste_switch.value
            new_mouse = self.mouse_hotkey_switch.value
            new_model_path = self.model_path_field.value.strip()

            settings = {
                "hotkey": new_hotkey,
                "mouse_hotkey": new_mouse,
                "input_device_index": new_mic,
                "auto_paste": new_paste,
                "model_path": new_model_path
            }
            
            self.core.save_settings(settings)
            logging.info("Auto-save triggered.")
            
        except Exception as e:
            logging.error(f"Auto-save failed: {e}")

    def _save_settings(self, e):
        # Deprecated: Kept only if needed for manual trigger, but UI button is removed.
        self._trigger_auto_save()
        if self.tray_supported:
            self.page.window_visible = False
        
        self.page.update()

    def set_exit_callback(self, callback):
        self.on_exit_callback = callback

    def _quit_app(self, e):
        logging.info("Exit requested via UI.")
        if hasattr(self, 'on_exit_callback') and self.on_exit_callback:
            self.on_exit_callback()
        else:
            self.page.window_destroy()
