import flet as ft
import logging
from ui.theme import *

class GeneralTab(ft.Container):
    def __init__(self, core, on_change):
        super().__init__()
        self.core = core
        self.on_change = on_change
        self.padding = 10
        self.content = self._build_content()
        self._sync_values()

    def _build_content(self):
        # Fields
        self.hotkey_field = ft.TextField(
            label="Global Hotkey",
            hint_text="ex: f3, ctrl+space",
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            on_blur=lambda e: self.on_change(hotkey=self.hotkey_field.value.strip())
        )
        
        self.mic_dropdown = ft.Dropdown(
            label="Microphone",
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            on_change=lambda e: self.on_change(input_device_index=int(self.mic_dropdown.value) if self.mic_dropdown.value else None)
        )
        
        self.auto_paste_switch = self._create_switch("Auto-paste result", "auto_paste")
        self.mouse_hotkey_switch = self._create_switch("Mouse Shortcut (LMB+RMB)", "mouse_hotkey")
        self.append_space_switch = self._create_switch("Append Space to End", "append_space")

        return ft.Column(
            [
                ft.Container(height=10),
                self.hotkey_field,
                self.mic_dropdown,
                ft.Divider(color=COLOR_BORDER),
                self.auto_paste_switch,
                self.mouse_hotkey_switch,
                self.append_space_switch,
            ],
            spacing=15,
            scroll=ft.ScrollMode.AUTO
        )

    def _create_switch(self, label, setting_key):
        return ft.Switch(
            label=label,
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, size=13),
            on_change=lambda e: self.on_change(**{setting_key: e.control.value})
        )

    def _sync_values(self):
        settings = self.core.settings
        self.hotkey_field.value = settings.hotkey
        
        # Populate Mics
        devices = self.core.get_audio_devices()
        self.mic_dropdown.options = [ft.dropdown.Option(key=str(d['id']), text=d['name']) for d in devices]
        self.mic_dropdown.value = str(settings.input_device_index) if settings.input_device_index is not None else None

        self.auto_paste_switch.value = settings.auto_paste
        self.mouse_hotkey_switch.value = settings.mouse_hotkey
        self.append_space_switch.value = settings.append_space
