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
        # Define Tabs
        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            indicator_color=COLOR_ACCENT,
            label_color=COLOR_ACCENT,
            unselected_label_color=COLOR_TEXT_SECONDARY,
            divider_color=COLOR_BORDER,
            expand=True,
            tabs=[
                ft.Tab(
                    text="General",
                    icon=ft.icons.SETTINGS,
                    content=self._build_general_tab_content()
                ),
                ft.Tab(
                    text="Sound",
                    icon=ft.icons.VOLUME_UP,
                    content=self._build_sound_tab_content()
                ),
                ft.Tab(
                    text="AI",
                    icon=ft.icons.AUTO_AWESOME,
                    content=self._build_ai_tab_content()
                ),
            ]
        )

        header = self._build_header()
        status_footer = self._build_status_footer()

        main_col = ft.Column(
            [
                header,
                ft.Container(content=self.tabs, expand=True), # Tabs take available space
                status_footer
            ],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )

        return ft.Container(
            content=main_col, 
            padding=15, # Reduced padding
            expand=True,
            bgcolor=COLOR_BG
        )

    def _build_header(self):
        return ft.Container(
            content=ft.Text(
                "Whisper Flash", 
                size=20, # Reduced size
                weight=ft.FontWeight.BOLD, 
                color=COLOR_TEXT_PRIMARY
            ),
            padding=ft.padding.only(bottom=5, top=5),
            alignment=ft.alignment.center
        )

    def _build_general_tab_content(self):
        # Hotkey Field
        self.hotkey_field = ft.TextField(
            label="Global Hotkey",
            value=self.core.settings.get("hotkey", "f3"),
            hint_text="ex: f3, ctrl+space",
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            on_blur=lambda e: self._trigger_auto_save()
        )
        
        # Microphone Selector
        devices = self.core.get_audio_devices()
        current_mic = self.core.settings.get("input_device_index")
        mic_options = [ft.dropdown.Option(key=str(d['id']), text=d['name']) for d in devices]
        
        valid_keys = [opt.key for opt in mic_options]
        if current_mic is not None and str(current_mic) not in valid_keys:
            logging.warning(f"Saved mic index {current_mic} not found. Resetting to default.")
            current_mic = None
        
        self.mic_dropdown = ft.Dropdown(
            label="Microphone",
            options=mic_options,
            value=str(current_mic) if current_mic is not None else None,
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            on_change=lambda e: self._trigger_auto_save()
        )

        # Switches
        self.auto_paste_switch = ft.Switch(
            label="Auto-paste result", 
            value=self.core.settings.get("auto_paste", True),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, size=13),
            on_change=lambda e: self._trigger_auto_save()
        )

        self.mouse_hotkey_switch = ft.Switch(
            label="Mouse Shortcut (LMB+RMB)", 
            value=self.core.settings.get("mouse_hotkey", False),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, size=13),
            on_change=lambda e: self._trigger_auto_save()
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=10),
                    self.hotkey_field,
                    self.mic_dropdown,
                    ft.Divider(color=COLOR_BORDER),
                    self.auto_paste_switch,
                    self.mouse_hotkey_switch,
                ],
                spacing=15,
                scroll=ft.ScrollMode.AUTO
            ),
            padding=10
        )

    def _build_sound_tab_content(self):
        # Sound Feedback Controls
        self.sound_switch = ft.Switch(
            label="Enable Sound Feedback", 
            value=self.core.settings.get("sound_enabled", True),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, size=14),
            on_change=lambda e: self._trigger_auto_save()
        )

        self.volume_slider = ft.Slider(
            min=0, max=100, divisions=20,
            value=self.core.settings.get("sound_volume", 50),
            label="Volume: {value}%",
            active_color=COLOR_ACCENT,
            on_change_end=lambda e: self._trigger_auto_save()
        )

        self.start_freq_slider = ft.Slider(
            min=200, max=2000, divisions=18, 
            value=self.core.settings.get("sound_freq_start", 800),
            label="Start Tone: {value}Hz",
            active_color=COLOR_ACCENT,
            on_change_end=lambda e: self._trigger_auto_save()
        )

        self.stop_freq_slider = ft.Slider(
            min=200, max=2000, divisions=18, 
            value=self.core.settings.get("sound_freq_stop", 500),
            label="Stop Tone: {value}Hz",
            active_color=COLOR_ACCENT,
            on_change_end=lambda e: self._trigger_auto_save()
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=10),
                    self.sound_switch,
                    ft.Divider(color=COLOR_BORDER),
                    ft.Text("Volume", size=12, color=COLOR_TEXT_SECONDARY),
                    self.volume_slider,
                    ft.Container(height=5),
                    ft.Text("Start Tone Frequency", size=12, color=COLOR_TEXT_SECONDARY),
                    self.start_freq_slider,
                    ft.Container(height=5),
                    ft.Text("Stop Tone Frequency", size=12, color=COLOR_TEXT_SECONDARY),
                    self.stop_freq_slider,
                ],
                spacing=10,
                scroll=ft.ScrollMode.AUTO
            ),
            padding=10
        )

    def _build_ai_tab_content(self):
        # API Key
        self.gemini_key_field = ft.TextField(
            label="Gemini API Key",
            value=self.core.settings.get("gemini_api_key", ""),
            password=True,
            can_reveal_password=True,
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
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
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            visible=self.core.settings.get("gemini_enabled", False),
            on_change=lambda e: self._trigger_auto_save()
        )
        
        # Prompt Field with Custom Label Row
        def reset_prompt(e):
            default_prompt = "Correct the text's punctuation and grammar without altering its meaning. Make it more expressive where appropriate, remove unnecessary repetitions, and improve flow. Combine sentences that make sense together. Maintain the original language and tone."
            self.gemini_prompt_field.value = default_prompt
            self.gemini_prompt_field.update()
            self._trigger_auto_save()

        prompt_label = ft.Row(
            [
                ft.Text("System Prompt", size=12, color=COLOR_TEXT_SECONDARY),
                ft.IconButton(
                    icon=ft.icons.RESTORE, 
                    icon_size=16, 
                    icon_color=COLOR_ACCENT,
                    tooltip="Reset to Default", 
                    on_click=reset_prompt,
                    style=ft.ButtonStyle(padding=0)
                )
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            visible=self.core.settings.get("gemini_enabled", False)
        )
        
        # Keep reference to label row to toggle visibility
        self.prompt_label_row = prompt_label

        self.gemini_prompt_field = ft.TextField(
            value=self.core.settings.get("gemini_prompt", ""),
            multiline=True,
            min_lines=3, max_lines=5,
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            visible=self.core.settings.get("gemini_enabled", False),
            on_blur=lambda e: self._trigger_auto_save()
        )

        # Toggle for enabling AI
        self.gemini_switch = ft.Switch(
            label="Enable AI Correction",
            value=self.core.settings.get("gemini_enabled", False),
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
            on_change=lambda e: self._toggle_gemini_fields()
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(height=10),
                    self.gemini_switch,
                    ft.Divider(color=COLOR_BORDER),
                    self.gemini_key_field,
                    self.gemini_model_dropdown,
                    self.prompt_label_row,
                    self.gemini_prompt_field
                ],
                spacing=12,
                scroll=ft.ScrollMode.AUTO
            ),
            padding=10
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
            # General
            new_hotkey = self.hotkey_field.value.strip()
            new_mic = int(self.mic_dropdown.value) if self.mic_dropdown.value else None
            new_paste = self.auto_paste_switch.value
            new_mouse = self.mouse_hotkey_switch.value
            
            # Sound
            new_sound_enabled = self.sound_switch.value
            new_sound_volume = int(self.volume_slider.value)
            new_start_freq = int(self.start_freq_slider.value)
            new_stop_freq = int(self.stop_freq_slider.value)
            
            # AI
            new_gemini_enabled = self.gemini_switch.value
            new_gemini_key = self.gemini_key_field.value.strip()
            new_gemini_model = self.gemini_model_dropdown.value
            new_gemini_prompt = self.gemini_prompt_field.value.strip()

            settings = {
                "hotkey": new_hotkey,
                "mouse_hotkey": new_mouse,
                "input_device_index": new_mic,
                "auto_paste": new_paste,
                "sound_enabled": new_sound_enabled,
                "sound_volume": new_sound_volume,
                "sound_freq_start": new_start_freq,
                "sound_freq_stop": new_stop_freq,
                "gemini_enabled": new_gemini_enabled,
                "gemini_api_key": new_gemini_key,
                "gemini_model": new_gemini_model,
                "gemini_prompt": new_gemini_prompt
            }
            
            # Create a full copy of settings to preserve other keys (like first_run)
            # and pass to save_settings so it can detect changes (e.g. hotkey)
            full_settings = self.core.settings.copy()
            full_settings.update(settings)
            
            self.core.save_settings(full_settings)
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
        if hasattr(self, 'prompt_label_row'):
            self.prompt_label_row.visible = visible
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

            # Sync Sound Section
            self.sound_switch.value = self.core.settings.get("sound_enabled", True)
            self.volume_slider.value = self.core.settings.get("sound_volume", 50)
            self.start_freq_slider.value = self.core.settings.get("sound_freq_start", 800)
            self.stop_freq_slider.value = self.core.settings.get("sound_freq_stop", 500)

            # Sync AI Section
            self.gemini_switch.value = self.core.settings.get("gemini_enabled", False)
            self.gemini_model_dropdown.value = self.core.settings.get("gemini_model", "gemini-2.5-flash-lite")
            self.gemini_key_field.value = self.core.settings.get("gemini_api_key", "")
            self.gemini_prompt_field.value = self.core.settings.get("gemini_prompt", "")
            
            self.page.update()
            self._toggle_gemini_fields()
        except Exception as e:
            logging.error(f"UI Refresh failed: {e}")
