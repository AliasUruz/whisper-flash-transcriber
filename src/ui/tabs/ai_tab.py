import flet as ft
from ui.theme import *
from settings import VALID_MODELS, DEFAULT_PROMPT

class AITab(ft.Container):
    def __init__(self, core, on_change):
        super().__init__()
        self.core = core
        self.on_change = on_change
        self.padding = 10
        self.content = self._build_content()
        self._sync_values()

    def _build_content(self):
        # API Key
        self.gemini_key_field = ft.TextField(
            label="Gemini API Key",
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
            on_blur=lambda e: self.on_change(gemini_api_key=e.control.value.strip())
        )

        # Model Dropdown
        model_options = [
            ft.dropdown.Option(m, m.replace("-", " ").title()) 
            for m in VALID_MODELS
        ]
        
        self.gemini_model_dropdown = ft.Dropdown(
            label="AI Model",
            options=model_options,
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            label_style=ft.TextStyle(color=COLOR_TEXT_SECONDARY, size=11),
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            on_change=lambda e: self.on_change(gemini_model=e.control.value)
        )
        
        # Prompt Field
        self.gemini_prompt_field = ft.TextField(
            multiline=True,
            min_lines=3, max_lines=5,
            text_size=13,
            color=COLOR_TEXT_PRIMARY,
            border_color=COLOR_BORDER,
            focused_border_color=COLOR_ACCENT,
            border_radius=BORDER_RADIUS,
            dense=True,
            content_padding=12,
            on_blur=lambda e: self.on_change(gemini_prompt=e.control.value.strip())
        )

        prompt_label = ft.Row(
            [
                ft.Text("System Prompt", size=12, color=COLOR_TEXT_SECONDARY),
                ft.IconButton(
                    icon=ft.icons.RESTORE, 
                    icon_size=16, 
                    icon_color=COLOR_ACCENT,
                    tooltip="Reset to Default", 
                    on_click=self._reset_prompt,
                    style=ft.ButtonStyle(padding=0)
                )
            ],
            alignment=ft.MainAxisAlignment.START,
        )

        self.gemini_switch = ft.Switch(
            label="Enable AI Correction",
            active_color=COLOR_ACCENT,
            label_style=ft.TextStyle(color=COLOR_TEXT_PRIMARY, weight=ft.FontWeight.BOLD),
            on_change=self._on_enable_change
        )
        
        self.ai_controls = ft.Column(
            [
                ft.Divider(color=COLOR_BORDER),
                self.gemini_key_field,
                self.gemini_model_dropdown,
                prompt_label,
                self.gemini_prompt_field
            ],
            visible=False,
            spacing=12
        )

        return ft.Column(
            [
                ft.Container(height=10),
                self.gemini_switch,
                self.ai_controls
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO
        )

    def _on_enable_change(self, e):
        enabled = e.control.value
        self.ai_controls.visible = enabled
        self.update()
        self.on_change(gemini_enabled=enabled)

    def _reset_prompt(self, e):
        self.gemini_prompt_field.value = DEFAULT_PROMPT
        self.gemini_prompt_field.update()
        self.on_change(gemini_prompt=DEFAULT_PROMPT)

    def _sync_values(self):
        s = self.core.settings
        self.gemini_switch.value = s.gemini_enabled
        self.ai_controls.visible = s.gemini_enabled
        
        self.gemini_key_field.value = s.gemini_api_key
        self.gemini_model_dropdown.value = s.gemini_model
        self.gemini_prompt_field.value = s.gemini_prompt

    def update_from_settings(self):
        """Update controls from current core settings (called by AppUI)."""
        if hasattr(self, 'gemini_switch'): # Changed from ai_switch to gemini_switch to match init
            self.gemini_switch.value = self.core.settings.gemini_enabled
            # Also update visibility
            self.ai_controls.visible = self.core.settings.gemini_enabled
            
        if hasattr(self, 'gemini_model_dropdown'): # Changed from model_dropdown to match init
            self.gemini_model_dropdown.value = self.core.settings.gemini_model
        
        try:
            self.update()
        except Exception: pass
