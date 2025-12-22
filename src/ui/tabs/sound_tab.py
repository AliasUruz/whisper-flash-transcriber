import flet as ft
from ui.theme import *

class SoundTab(ft.Container):
    def __init__(self, core, on_change):
        super().__init__()
        self.core = core
        self.on_change = on_change
        self.padding = 10
        self.content = self._build_content()
        self._sync_values()

    def _build_content(self):
        self.sound_switch = ft.Switch(
            label="Enable Sound Feedback", 
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, size=14),
            on_change=lambda e: self.on_change(sound_enabled=e.control.value)
        )

        self.volume_slider = self._create_slider(
            "Volume: {value}%", 0, 100, "sound_volume"
        )
        self.start_freq_slider = self._create_slider(
            "Start Tone: {value}Hz", 200, 2000, "sound_freq_start"
        )
        self.stop_freq_slider = self._create_slider(
            "Stop Tone: {value}Hz", 200, 2000, "sound_freq_stop"
        )

        return ft.Column(
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
        )

    def _create_slider(self, label_fmt, vmin, vmax, key):
        return ft.Slider(
            min=vmin, max=vmax, divisions=20,
            label=label_fmt,
            active_color=COLOR_ACCENT,
            on_change_end=lambda e: self.on_change(**{key: int(e.control.value)})
        )

    def _sync_values(self):
        s = self.core.settings
        self.sound_switch.value = s.sound_enabled
        self.volume_slider.value = s.sound_volume
        self.start_freq_slider.value = s.sound_freq_start
        self.stop_freq_slider.value = s.sound_freq_stop
