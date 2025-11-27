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

# --- Theme Constants (Balanced Modern) ---
COLOR_BG = "#202028"          # Mica Alt Dark
COLOR_SECTION_BG = "#2A2A35"  # Subtle Lighter Block
COLOR_TEXT_PRIMARY = "#FFFFFF"
COLOR_TEXT_SECONDARY = "#AAAAAA"
COLOR_ACCENT = "#00F0FF"      # Neon Cyan
COLOR_BORDER = "#3A3A45"
BORDER_RADIUS = 10

class AppUI:
    def __init__(self, page: ft.Page, core: CoreService):
        self.page = page
        self.core = core
        self.page.title = "Whisper Flash"
        
        self.hotkey_field: ft.TextField | None = None
        self.mouse_hotkey_switch: ft.Switch | None = None
        self.auto_paste_switch: ft.Switch | None = None
        self.mic_dropdown: ft.Dropdown | None = None
        self.gemini_key_field: ft.TextField | None = None
        self.gemini_model_dropdown: ft.Dropdown | None = None
        
        # Tray is now handled by pystray in main.py
        self.tray_supported = False

    def build_controls(self) -> ft.Container:
        header = self._build_header()
        general_section = self._build_general_section()
        ai_section = self._build_ai_section()
        status_footer = self._build_status_footer()

        main_col = ft.Column(
            [
                header,
                ft.Container(height=10), # Spacer
                general_section,
                ft.Container(height=10), # Spacer
                ai_section,
                ft.Container(expand=True), # Push footer down
                status_footer
            ],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )

        return ft.Container(
            content=main_col, 
            padding=25, 
            expand=True,
            bgcolor=COLOR_BG
        )

    def _build_header(self):
        return ft.Container(
            content=ft.Text(
                "Whisper Flash Configuration", 
                size=24, 
                weight=ft.FontWeight.BOLD, 
                color=COLOR_TEXT_PRIMARY
            ),
            padding=ft.padding.only(bottom=10)
        )

    def _build_general_section(self):
        # Hotkey Field
        self.hotkey_field = ft.TextField(
            label="Global Hotkey",
            value=self.core.settings.get("hotkey", "f3"),
            hint_text="ex: f3, ctrl+space",
            text_size=14,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=12),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=15,
            on_blur=lambda e: self._trigger_auto_save()
        )
        
        # Microphone Selector
        devices = self.core.get_audio_devices()
        current_mic = self.core.settings.get("input_device_index")
        mic_options = [ft.dropdown.Option(key=str(d['id']), text=d['name']) for d in devices]
        
        # Validate current_mic index exists in available devices
        valid_keys = [opt.key for opt in mic_options]
        if current_mic is not None and str(current_mic) not in valid_keys:
            logging.warning(f"Saved mic index {current_mic} not found. Resetting to default.")
            current_mic = None
        
        self.mic_dropdown = ft.Dropdown(
            label="Microphone",
            options=mic_options,
            value=str(current_mic) if current_mic is not None else None,
            text_size=14,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=12),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=15,
            on_change=lambda e: self._trigger_auto_save()
        )

        # Switches
        self.auto_paste_switch = ft.Switch(
            label="Auto-paste result", 
            value=self.core.settings.get("auto_paste", True),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY),
            on_change=lambda e: self._trigger_auto_save()
        )

        self.mouse_hotkey_switch = ft.Switch(
            label="Mouse Shortcut (LMB + RMB)", 
            value=self.core.settings.get("mouse_hotkey", False),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY),
            on_change=lambda e: self._trigger_auto_save()
        )

        return ft.Column(
            [
                ft.Text("General", size=16, weight=ft.FontWeight.W_600, color=COLOR_TEXT_PRIMARY),
                ft.Container(height=5),
                self.hotkey_field,
                self.mic_dropdown,
                ft.Container(height=5),
                ft.Row([self.auto_paste_switch, self.mouse_hotkey_switch], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ],
            spacing=15
        )

    def _build_ai_section(self):
        # API Key
        self.gemini_key_field = ft.TextField(
            label="Gemini API Key",
            value=self.core.settings.get("gemini_api_key", ""),
            password=True,
            can_reveal_password=True,
            text_size=14,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=12),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=15,
            visible=self.core.settings.get("gemini_enabled", False),
            on_blur=lambda e: self._trigger_auto_save()
        )

        # Model Dropdown
        self.gemini_model_dropdown = ft.Dropdown(
            label="AI Model",
            options=[
                ft.dropdown.Option("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite"),
                ft.dropdown.Option("gemini-2.5-flash", "Gemini 2.5 Flash"),
            ],
            value=self.core.settings.get("gemini_model", "gemini-2.5-flash-lite"),
            text_size=14,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=12),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=15,
            visible=self.core.settings.get("gemini_enabled", False),
            on_change=lambda e: self._trigger_auto_save()
        )
        
        # Prompt Field
        self.gemini_prompt_field = ft.TextField(
            label="System Prompt",
            value=self.core.settings.get("gemini_prompt", ""),
            multiline=True,
            min_lines=2, max_lines=3,
            text_size=14,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=12),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=15,
            visible=self.core.settings.get("gemini_enabled", False),
            on_blur=lambda e: self._trigger_auto_save()
        )

        # Toggle for enabling AI
        self.gemini_switch = ft.Switch(
            label="Enable AI Correction",
            value=self.core.settings.get("gemini_enabled", False),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY),
            on_change=lambda e: self._toggle_gemini_fields()
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row([
                        ft.Text("AI Correction", size=16, weight=ft.FontWeight.W_600, color=COLOR_TEXT_PRIMARY),
                        self.gemini_switch
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    ft.Container(height=5),
                    self.gemini_key_field,
                    self.gemini_model_dropdown,
                    self.gemini_prompt_field
                ],
                spacing=15
            ),
            bgcolor=COLOR_SECTION_BG,
            border_radius=BORDER_RADIUS,
            padding=20
        )

    def _build_status_footer(self):
        # Simple footer text for status, since we removed the big status container
        self.status_text = ft.Text("Ready", size=12, color=COLOR_TEXT_SECONDARY)
        return ft.Container(
            content=ft.Row([self.status_text], alignment=ft.MainAxisAlignment.CENTER),
            padding=10
        )

    def update_status(self, status: str, tooltip: str):
        if not self.page: return
        icon_path = ICON_MAP.get(status, DEFAULT_ICON)
        
        try:
            if hasattr(self, 'status_text'):
                self.status_text.value = f"Status: {tooltip}"
                self.status_text.color = COLOR_ACCENT if status == "recording" else COLOR_TEXT_SECONDARY
                self.status_text.update()

            if self.tray_supported and self.tray_icon:
                self.tray_icon.tooltip = f"WF: {tooltip}"
                self.tray_icon.icon = icon_path
                self.tray_icon.update()
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
            bgcolor=COLOR_SECTION_BG,
        )
        self.page.dialog.open = True
        try: self.page.update()
        except Exception: pass

    def _trigger_auto_save(self):
        """Saves settings silently."""
        try:
            new_hotkey = self.hotkey_field.value.strip()
            new_mic = int(self.mic_dropdown.value) if self.mic_dropdown.value else None
            new_paste = self.auto_paste_switch.value
            new_mouse = self.mouse_hotkey_switch.value
            
            # Note: model_path field was removed in mockup, assuming user doesn't need it or it's advanced.
            # Preserving existing value if not in UI.
            new_model_path = self.core.settings.get("model_path", "")

            settings = {
                "hotkey": new_hotkey,
                "mouse_hotkey": new_mouse,
                "input_device_index": new_mic,
                "auto_paste": new_paste,
                "model_path": new_model_path,
                "gemini_enabled": self.gemini_switch.value,
                "gemini_api_key": self.gemini_key_field.value.strip(),
                "gemini_model": self.gemini_model_dropdown.value,
                "gemini_prompt": self.gemini_prompt_field.value.strip()
            }
            
            self.core.save_settings(settings)
            logging.info("Auto-save triggered.")
            
        except Exception as e:
            logging.error(f"Auto-save failed: {e}")

    def set_exit_callback(self, callback):
        self.on_exit_callback = callback

    def _toggle_gemini_fields(self):
        visible = self.gemini_switch.value
        self.gemini_key_field.visible = visible
        self.gemini_model_dropdown.visible = visible
        self.gemini_prompt_field.visible = visible
        self.page.update()
        self._trigger_auto_save()

    def refresh_ui_from_settings(self):
        """Updates UI elements from core settings."""
        try:
            # Sync General Section
            self.hotkey_field.value = self.core.settings.get("hotkey", "f3")
            
            current_mic = self.core.settings.get("input_device_index")
            self.mic_dropdown.value = str(current_mic) if current_mic is not None else None
            
            self.auto_paste_switch.value = self.core.settings.get("auto_paste", True)
            self.mouse_hotkey_switch.value = self.core.settings.get("mouse_hotkey", False)

            # Sync AI Section
            self.gemini_switch.value = self.core.settings.get("gemini_enabled", False)
            self.gemini_model_dropdown.value = self.core.settings.get("gemini_model", "gemini-2.5-flash-lite")
            self.gemini_key_field.value = self.core.settings.get("gemini_api_key", "")
            self.gemini_prompt_field.value = self.core.settings.get("gemini_prompt", "")
            
            self.page.update()
            self._toggle_gemini_fields()
        except Exception as e:
            logging.error(f"UI Refresh failed: {e}")
