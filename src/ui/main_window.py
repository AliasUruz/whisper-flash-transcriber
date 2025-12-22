import flet as ft
import logging
from ui.theme import *
from ui.tabs.general_tab import GeneralTab
from ui.tabs.sound_tab import SoundTab
from ui.tabs.ai_tab import AITab
from core import AppState

class AppUI:
    def __init__(self, page: ft.Page, core):
        self.page = page
        self.core = core
        self.page.bgcolor = COLOR_BG
        
        self.tray_supported = False  # Managed by main.py

    def build_controls(self) -> ft.Container:
        # Pass auto-save callback to tabs
        self.general_tab = GeneralTab(self.core, self._on_settings_change)
        self.sound_tab = SoundTab(self.core, self._on_settings_change)
        self.ai_tab = AITab(self.core, self._on_settings_change)

        self.tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            indicator_color=COLOR_ACCENT,
            label_color=COLOR_ACCENT,
            unselected_label_color=COLOR_TEXT_SECONDARY,
            divider_color=COLOR_BORDER,
            expand=True,
            tabs=[
                ft.Tab(text="General", icon=ft.icons.SETTINGS, content=self.general_tab),
                ft.Tab(text="Sound", icon=ft.icons.VOLUME_UP, content=self.sound_tab),
                ft.Tab(text="AI", icon=ft.icons.AUTO_AWESOME, content=self.ai_tab),
            ]
        )

        header = self._build_header()
        status_footer = self._build_status_footer()

        main_col = ft.Column(
            [
                header,
                ft.Container(content=self.tabs, expand=True),
                status_footer
            ],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        )

        return ft.Container(
            content=main_col, 
            padding=15,
            expand=True,
            bgcolor=COLOR_BG
        )

    def _build_header(self):
        return ft.Container(
            content=ft.Text(
                "Whisper Flash", 
                size=20, 
                weight=ft.FontWeight.BOLD, 
                color=COLOR_TEXT_PRIMARY
            ),
            padding=ft.padding.only(bottom=5, top=5),
            alignment=ft.alignment.center
        )

    def _build_status_footer(self):
        self.status_text = ft.Text("Ready", size=12, color=COLOR_TEXT_SECONDARY)
        return ft.Container(
            content=ft.Row([self.status_text], alignment=ft.MainAxisAlignment.CENTER),
            padding=10
        )

    def _on_settings_change(self, **kwargs):
        """Callback passed to tabs to trigger updates."""
        try:
            logging.debug(f"UI Change: {kwargs}")
            self.core.save_settings(kwargs)
        except Exception as e:
            logging.error(f"Settings update failed: {e}")

    def update_status(self, status: str, tooltip: str):
        if not self.page: return
        
        # Map string status to Enum if needed, or handle raw string
        # Status comes from CoreService callback which sends strings usually
        
        color = COLOR_TEXT_SECONDARY
        if status == AppState.RECORDING.value:
            color = COLOR_ACCENT
        elif status == AppState.ERROR.value:
            color = ft.colors.RED

        try:
            if hasattr(self, 'status_text'):
                self.status_text.value = f"Status: {tooltip}"
                self.status_text.color = color
                self.status_text.update()

            # Update Tray (if attached externally)
            if self.tray_supported:
                pass  # Icon swapping handled by SystemTray
                
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

    def set_exit_callback(self, callback):
        """Pass main exit callback to UI if needed (e.g. for a Close button in Settings)."""
        self.exit_callback = callback

    def refresh_ui_from_settings(self):
        """Forces all tabs to reload from fresh core settings."""
        try:
            if hasattr(self, 'ai_tab'):
                self.ai_tab.update_from_settings()
        except Exception as e:
            logging.error(f"UI Refresh failed: {e}")
