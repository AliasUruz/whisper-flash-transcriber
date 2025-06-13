import customtkinter as ctk
from typing import Dict, Any, Callable


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, core_instance):
        super().__init__(master)
        self.core_instance = core_instance
        self.initial_config = core_instance.config
        self.is_dirty = False

        self.title("Whisper Recorder Settings")
        self.geometry("620x600")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.attributes("-topmost", True)
        self.transient(master)
        self.grab_set()

        self.tab_view = ctk.CTkTabview(self, anchor="nw")
        self.tab_view.pack(padx=10, pady=10, fill="both", expand=True)
        self.tab_view.add("Geral")
        self.tab_view.add("Hotkeys")
        self.tab_view.add("Correção com IA")
        self.tab_view.add("Avançado")

        self.general_frame = GeneralSettingsFrame(
            self.tab_view.tab("Geral"),
            self.initial_config,
            self.set_dirty
        )
        self.correction_frame = CorrectionSettingsFrame(
            self.tab_view.tab("Correção com IA"),
            self.initial_config,
            self.set_dirty
        )

        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(padx=10, pady=(0, 10), fill="x", side="bottom")
        self.apply_button = ctk.CTkButton(
            self.action_frame,
            text="Aplicar",
            command=self.apply_and_close,
            state="disabled"
        )
        self.apply_button.pack(side="right", padx=(10, 0))
        self.cancel_button = ctk.CTkButton(
            self.action_frame,
            text="Cancelar",
            command=self.destroy,
            fg_color="#555555",
            hover_color="#6E6E6E"
        )
        self.cancel_button.pack(side="right")

    def set_dirty(self, *args):
        if not self.is_dirty:
            self.is_dirty = True
            self.apply_button.configure(state="normal")

    def apply_and_close(self):
        if not self.is_dirty:
            self.destroy()
            return

        final_settings: Dict[str, Any] = {}
        final_settings.update(self.general_frame.get_settings())
        final_settings.update(self.correction_frame.get_settings())
        self.core_instance.apply_settings_from_external(**final_settings)
        self.destroy()


class GeneralSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, initial_config: Dict[str, Any], set_dirty_callback: Callable):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="both", expand=True, padx=5, pady=5)

        self.auto_paste_var = ctk.BooleanVar(value=initial_config.get("auto_paste"))
        ctk.CTkSwitch(
            self,
            text="Colar texto automaticamente...",
            variable=self.auto_paste_var
        ).pack(anchor="w", pady=(5, 20), padx=5)
        self.auto_paste_var.trace_add("write", set_dirty_callback)

        sound_frame = ctk.CTkFrame(self)
        sound_frame.pack(fill="x", expand=True, anchor="w", padx=5)
        ctk.CTkLabel(
            sound_frame,
            text="Feedback Sonoro",
            font=ctk.CTkFont(weight="bold")
        ).pack(anchor="w", padx=10, pady=(5, 0))

        self.sound_enabled_var = ctk.BooleanVar(value=initial_config.get("sound_enabled"))
        ctk.CTkSwitch(
            sound_frame,
            text="Habilitar sons...",
            variable=self.sound_enabled_var
        ).pack(anchor="w", padx=10, pady=5)
        self.sound_enabled_var.trace_add("write", set_dirty_callback)

        ctk.CTkLabel(sound_frame, text="Volume").pack(anchor="w", padx=10)
        self.sound_volume_var = ctk.DoubleVar(value=initial_config.get("sound_volume"))
        ctk.CTkSlider(
            sound_frame,
            from_=0.0,
            to=1.0,
            variable=self.sound_volume_var
        ).pack(fill="x", padx=10, pady=(0, 10))
        self.sound_volume_var.trace_add("write", set_dirty_callback)

        grid_frame = ctk.CTkFrame(sound_frame, fg_color="transparent")
        grid_frame.pack(fill="x", padx=10)
        grid_frame.columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(grid_frame, text="Frequência (Hz)").grid(row=0, column=0, sticky="w")
        self.sound_frequency_var = ctk.StringVar(value=str(initial_config.get("sound_frequency")))
        ctk.CTkEntry(grid_frame, textvariable=self.sound_frequency_var).grid(row=1, column=0, sticky="ew", padx=(0, 5))
        self.sound_frequency_var.trace_add("write", set_dirty_callback)

        ctk.CTkLabel(grid_frame, text="Duração (s)").grid(row=0, column=1, sticky="w")
        self.sound_duration_var = ctk.StringVar(value=str(initial_config.get("sound_duration")))
        ctk.CTkEntry(grid_frame, textvariable=self.sound_duration_var).grid(row=1, column=1, sticky="ew", padx=(5, 0))
        self.sound_duration_var.trace_add("write", set_dirty_callback)

    def get_settings(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "new_auto_paste": self.auto_paste_var.get(),
            "new_sound_enabled": self.sound_enabled_var.get(),
            "new_sound_volume": self.sound_volume_var.get(),
        }
        try:
            result["new_sound_frequency"] = int(self.sound_frequency_var.get())
        except ValueError:
            result["new_sound_frequency"] = None
        try:
            result["new_sound_duration"] = float(self.sound_duration_var.get())
        except ValueError:
            result["new_sound_duration"] = None
        return result


class CorrectionSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, initial_config: Dict[str, Any], set_dirty_callback: Callable):
        super().__init__(master, fg_color="transparent")
        self.pack(fill="both", expand=True, padx=5, pady=5)

        self.enabled_var = ctk.BooleanVar(value=initial_config.get("text_correction_enabled"))
        self.service_var = ctk.StringVar(value=initial_config.get("text_correction_service"))
        self.gemini_api_key_var = ctk.StringVar(value=initial_config.get("gemini_api_key"))
        self.gemini_model_var = ctk.StringVar(value=initial_config.get("gemini_model"))
        self.gemini_prompt_var = ctk.StringVar(value=initial_config.get("gemini_prompt"))
        self.openrouter_api_key_var = ctk.StringVar(value=initial_config.get("openrouter_api_key"))
        self.openrouter_model_var = ctk.StringVar(value=initial_config.get("openrouter_model"))

        for var in [
            self.enabled_var,
            self.service_var,
            self.gemini_api_key_var,
            self.gemini_model_var,
            self.gemini_prompt_var,
            self.openrouter_api_key_var,
            self.openrouter_model_var,
        ]:
            var.trace_add("write", set_dirty_callback)

        ctk.CTkSwitch(
            self,
            text="Habilitar Correção de Texto com IA",
            variable=self.enabled_var
        ).pack(anchor="w", padx=5, pady=(5, 10))

        self.service_selection_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkRadioButton(
            self.service_selection_frame,
            text="Gemini",
            variable=self.service_var,
            value="gemini"
        ).pack(side="left", padx=5)
        ctk.CTkRadioButton(
            self.service_selection_frame,
            text="OpenRouter",
            variable=self.service_var,
            value="openrouter"
        ).pack(side="left", padx=5)

        self.gemini_frame = self._create_gemini_frame(initial_config)
        self.openrouter_frame = self._create_openrouter_frame(initial_config)

        self.enabled_var.trace_add("write", self._update_visibility)
        self.service_var.trace_add("write", self._update_visibility)
        self._update_visibility()

    def _create_gemini_frame(self, initial_config: Dict[str, Any]):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(frame, text="Chave de API do Gemini").pack(anchor="w", padx=5, pady=(5, 0))
        ctk.CTkEntry(frame, textvariable=self.gemini_api_key_var, show="*").pack(fill="x", padx=5, pady=(0, 5))
        ctk.CTkLabel(frame, text="Modelo do Gemini").pack(anchor="w", padx=5)
        ctk.CTkOptionMenu(
            frame,
            variable=self.gemini_model_var,
            values=initial_config.get("gemini_model_options", [])
        ).pack(fill="x", padx=5, pady=(0, 5))
        ctk.CTkLabel(frame, text="Prompt de Correção").pack(anchor="w", padx=5)
        textbox = ctk.CTkTextbox(frame, height=120)
        textbox.pack(fill="both", padx=5, pady=(0, 5), expand=True)
        textbox.insert("1.0", self.gemini_prompt_var.get())
        textbox.bind(
            "<KeyRelease>",
            lambda event: self.gemini_prompt_var.set(textbox.get("1.0", "end-1c"))
        )
        return frame

    def _create_openrouter_frame(self, initial_config: Dict[str, Any]):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        ctk.CTkLabel(frame, text="Chave de API do OpenRouter").pack(anchor="w", padx=5, pady=(5, 0))
        ctk.CTkEntry(frame, textvariable=self.openrouter_api_key_var, show="*").pack(fill="x", padx=5, pady=(0, 5))
        ctk.CTkLabel(frame, text="Modelo do OpenRouter").pack(anchor="w", padx=5)
        ctk.CTkEntry(frame, textvariable=self.openrouter_model_var).pack(fill="x", padx=5, pady=(0, 5))
        return frame

    def _update_visibility(self, *args):
        self.service_selection_frame.pack_forget()
        self.gemini_frame.pack_forget()
        self.openrouter_frame.pack_forget()

        if self.enabled_var.get():
            self.service_selection_frame.pack(anchor="w", padx=5, pady=(0, 5))
            if self.service_var.get() == "gemini":
                self.gemini_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))
            elif self.service_var.get() == "openrouter":
                self.openrouter_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

    def get_settings(self) -> Dict[str, Any]:
        return {
            "new_text_correction_enabled": self.enabled_var.get(),
            "new_text_correction_service": self.service_var.get(),
            "new_gemini_api_key": self.gemini_api_key_var.get(),
            "new_gemini_model": self.gemini_model_var.get(),
            "new_gemini_prompt": self.gemini_prompt_var.get(),
            "new_openrouter_api_key": self.openrouter_api_key_var.get(),
            "new_openrouter_model": self.openrouter_model_var.get(),
        }

