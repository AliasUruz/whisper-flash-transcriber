from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox


@dataclass(slots=True)
class WizardDownloadRequest:
    """Structured request to trigger an ASR model download."""

    model_id: str
    backend: str
    cache_dir: str
    quant: str | None = None


@dataclass(slots=True)
class WizardResult:
    """Outcome of the onboarding wizard."""

    config_updates: dict[str, Any]
    hotkey_preferences: dict[str, str]
    download_request: WizardDownloadRequest | None = None


class DownloadProgressPanel(ctk.CTkToplevel):
    """Simple indeterminate progress dialog reused across the application."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        title: str = "Download de Modelo",
        message: str = "Preparando download do modelo...",
        cancel_label: str = "Cancelar",
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.withdraw()
        self.title(title)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._handle_close if on_cancel is None else self._handle_cancel)

        self._cancel_callback = on_cancel
        self._finished = False

        self.columnconfigure(0, weight=1)

        self._message_var = tk.StringVar(value=message)
        self._status_var = tk.StringVar(value="")

        header = ctk.CTkLabel(self, textvariable=self._message_var, wraplength=420, justify="left")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(24, 12))

        self._progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self._progress.grid(row=1, column=0, sticky="ew", padx=24)
        self._progress.start()

        status = ctk.CTkLabel(self, textvariable=self._status_var, text_color=("#1f6aa5", "#1f6aa5"))
        status.grid(row=2, column=0, sticky="w", padx=24, pady=(12, 6))

        button_frame = ctk.CTkFrame(self)
        button_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(8, 16))
        button_frame.columnconfigure(0, weight=1)

        self._action_button = ctk.CTkButton(
            button_frame,
            text="Fechar" if on_cancel is None else cancel_label,
            command=self._handle_close if on_cancel is None else self._handle_cancel,
            width=110,
        )
        self._action_button.grid(row=0, column=1, sticky="e", padx=(0, 4))
        if on_cancel is None:
            self._finished = True

    def open(self) -> None:
        self.deiconify()
        self.update_idletasks()
        self.geometry(self._center_geometry(460, 210))
        self.wait_visibility()
        self.grab_set()
        self.focus_force()

    def wait_until_closed(self) -> None:
        self.wait_window()

    def _center_geometry(self, width: int, height: int) -> str:
        try:
            parent = self.master.winfo_toplevel()
            parent.update_idletasks()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            x = px + (pw - width) // 2
            y = py + (ph - height) // 2
        except Exception:
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x = (screen_w - width) // 2
            y = (screen_h - height) // 2
        return f"{width}x{height}+{x}+{y}"

    def update_message(self, message: str) -> None:
        self._message_var.set(message)

    def update_status(self, message: str) -> None:
        self._status_var.set(message)

    def mark_success(self, message: str) -> None:
        self._progress.stop()
        self._status_var.set(message)
        self._finalize_buttons("Fechar")

    def mark_error(self, message: str) -> None:
        self._progress.stop()
        self._status_var.set(message)
        self._finalize_buttons("Fechar")

    def mark_cancelled(self, message: str) -> None:
        self._progress.stop()
        self._status_var.set(message)
        self._finalize_buttons("Fechar")

    def close_after(self, delay_ms: int = 0) -> None:
        if delay_ms <= 0:
            self._handle_close()
        else:
            self.after(delay_ms, self._handle_close)

    def disable_cancel(self) -> None:
        if self._cancel_callback is not None and not self._finished:
            self._action_button.configure(state="disabled")

    def _finalize_buttons(self, label: str) -> None:
        if self._finished:
            self._action_button.configure(text=label, state="normal")
            return
        self._finished = True
        self._action_button.configure(text=label, state="normal", command=self._handle_close)

    def _handle_cancel(self) -> None:
        if self._cancel_callback is None:
            self._handle_close()
            return
        self.disable_cancel()
        try:
            self._cancel_callback()
        except Exception:
            pass
        self._status_var.set("Cancelando download...")

    def _handle_close(self) -> None:
        if self._finished:
            try:
                self.grab_release()
            except Exception:
                pass
            self.destroy()
        else:
            self._handle_cancel()


class FirstRunWizard(ctk.CTkToplevel):
    """Multi-step onboarding wizard for first-time configuration."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        first_run: bool,
        config_snapshot: Mapping[str, Any],
        hotkey_defaults: Mapping[str, str],
        profile_dir: str | None = None,
        recommended_models: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        super().__init__(master)
        self.withdraw()
        self.title("Assistente de Primeira Execução")
        self.minsize(640, 540)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        self._result: WizardResult | None = None
        self._first_run = bool(first_run)
        self._profile_dir = profile_dir
        self._config_snapshot = dict(config_snapshot)
        self._hotkey_snapshot = dict(hotkey_defaults)
        self._recommended_entries = self._normalize_catalog(recommended_models or [])

        self._step_index = 0
        self._steps: list[ctk.CTkFrame] = []
        self._validators: list[callable[[], bool] | None] = []
        self._titles: list[str] = []

        self._build_variables()
        self._build_layout()
        self._build_steps()
        self._show_step(0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> WizardResult | None:
        self.deiconify()
        self.update_idletasks()
        self.geometry(self._center_geometry())
        self.wait_visibility()
        self.grab_set()
        self.focus_force()
        self.wait_window()
        return self._result

    # ------------------------------------------------------------------
    # UI Construction Helpers
    # ------------------------------------------------------------------
    def _center_geometry(self, width: int = 680, height: int = 560) -> str:
        try:
            parent = self.master.winfo_toplevel()
            parent.update_idletasks()
            px = parent.winfo_rootx()
            py = parent.winfo_rooty()
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            x = px + (pw - width) // 2
            y = py + (ph - height) // 2
        except Exception:
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x = (screen_w - width) // 2
            y = (screen_h - height) // 2
        return f"{width}x{height}+{x}+{y}"

    def _build_variables(self) -> None:
        cfg = self._config_snapshot
        hotkeys = self._hotkey_snapshot

        record_key = str(hotkeys.get("record_key", cfg.get("record_key", "f3")))
        agent_key = str(hotkeys.get("agent_key", cfg.get("agent_key", "f4")))
        record_mode = str(hotkeys.get("record_mode", cfg.get("record_mode", "toggle"))).lower()

        self.record_key_var = tk.StringVar(value=record_key)
        self.agent_key_var = tk.StringVar(value=agent_key)
        self.record_mode_var = tk.StringVar(value=record_mode if record_mode in {"toggle", "press"} else "toggle")

        language = str(cfg.get("ui_language", "en-US"))
        self._language_choices = {
            "English (US)": "en-US",
            "Português (Brasil)": "pt-BR",
        }
        default_language_display = next(
            (label for label, code in self._language_choices.items() if code.lower() == language.lower()),
            "English (US)",
        )
        self.ui_language_var = tk.StringVar(value=default_language_display)

        model_id = str(cfg.get("asr_model_id", "openai/whisper-large-v3-turbo"))
        backend = str(cfg.get("asr_backend", "ctranslate2"))
        quant = str(cfg.get("asr_ct2_compute_type", "int8_float16"))

        self.model_id_var = tk.StringVar(value=model_id)
        self.backend_var = tk.StringVar(value=backend)
        self.quant_var = tk.StringVar(value=quant or "int8_float16")
        self.recommended_choice_var = tk.StringVar(value="<Manter configuração atual>")

        storage_root = str(cfg.get("storage_root_dir", ""))
        models_dir = str(cfg.get("models_storage_dir", ""))
        cache_dir = str(cfg.get("asr_cache_dir", ""))
        recordings_dir = str(cfg.get("recordings_dir", ""))

        self.storage_root_var = tk.StringVar(value=storage_root)
        self.models_dir_var = tk.StringVar(value=models_dir)
        self.cache_dir_var = tk.StringVar(value=cache_dir)
        self.recordings_dir_var = tk.StringVar(value=recordings_dir)

        self.download_now_var = tk.BooleanVar(value=self._first_run)
        self.summary_text_var = tk.StringVar(value="")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)

        container = ctk.CTkFrame(self)
        container.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        self._title_var = tk.StringVar(value="Bem-vindo")
        title = ctk.CTkLabel(
            container,
            textvariable=self._title_var,
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew", pady=(6, 12))

        self._content_frame = ctk.CTkFrame(container)
        self._content_frame.grid(row=1, column=0, sticky="nsew")
        self._content_frame.columnconfigure(0, weight=1)
        self._content_frame.rowconfigure(0, weight=1)

        nav_frame = ctk.CTkFrame(container)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        nav_frame.columnconfigure(0, weight=1)

        self._back_button = ctk.CTkButton(nav_frame, text="Voltar", command=self._on_back, width=100)
        self._back_button.grid(row=0, column=0, sticky="w")

        self._next_button = ctk.CTkButton(nav_frame, text="Avançar", command=self._on_next, width=110)
        self._next_button.grid(row=0, column=0)

        self._finish_button = ctk.CTkButton(nav_frame, text="Concluir", command=self._on_finish, width=110)
        self._finish_button.grid(row=0, column=0)

        self._cancel_button = ctk.CTkButton(nav_frame, text="Cancelar", command=self._on_cancel, width=110)
        self._cancel_button.grid(row=0, column=1, sticky="e")

    def _build_steps(self) -> None:
        self._steps.clear()
        self._validators.clear()
        self._titles.clear()

        self._register_step("Introdução", self._build_welcome_step(), None)
        self._register_step("Hotkeys", self._build_hotkey_step(), self._validate_hotkeys)
        self._register_step("Idioma", self._build_language_step(), self._validate_language)
        self._register_step("Modelo ASR", self._build_model_step(), self._validate_model)
        self._register_step("Diretórios", self._build_paths_step(), self._validate_paths)
        self._register_step("Resumo", self._build_summary_step(), None)

    def _register_step(self, title: str, frame: ctk.CTkFrame, validator: callable[[], bool] | None) -> None:
        frame.grid_remove()
        self._steps.append(frame)
        self._titles.append(title)
        self._validators.append(validator)

    def _build_welcome_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_frame)
        frame.grid(row=0, column=0, sticky="nsew")

        message_lines = [
            "Bem-vindo ao Whisper Flash Transcriber!",
            "\n",
            "Este assistente guiará você pelos ajustes essenciais da primeira execução,",
            "garantindo que o aplicativo esteja pronto para gravar, transcrever e baixar o modelo de ASR necessário.",
        ]
        if self._profile_dir:
            message_lines.extend(
                [
                    "\n",
                    f"Os arquivos de configuração serão armazenados em:\n{self._profile_dir}",
                ]
            )
        message_lines.extend(
            [
                "\n",
                "Se você precisar colar resultados em aplicativos que rodam como administrador, "
                "execute o Whisper Flash Transcriber com o mesmo nível de privilégio para evitar bloqueios do Windows.",
            ]
        )
        text = " ".join(message_lines)
        label = ctk.CTkLabel(frame, text=text, wraplength=600, justify="left")
        label.pack(anchor="w", padx=16, pady=20)

        return frame

    def _build_hotkey_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_frame)
        frame.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(frame, text="Defina as hotkeys principais", font=ctk.CTkFont(size=14, weight="bold")) \
            .pack(anchor="w", padx=16, pady=(16, 8))

        form = ctk.CTkFrame(frame)
        form.pack(fill="x", padx=16)
        for i in range(2):
            form.columnconfigure(i, weight=1)

        ctk.CTkLabel(form, text="Tecla para iniciar/parar gravação:").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=8)
        record_entry = ctk.CTkEntry(form, textvariable=self.record_key_var)
        record_entry.grid(row=0, column=1, sticky="ew", pady=8)

        ctk.CTkLabel(form, text="Tecla para o comando agêntico:").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=8)
        agent_entry = ctk.CTkEntry(form, textvariable=self.agent_key_var)
        agent_entry.grid(row=1, column=1, sticky="ew", pady=8)

        ctk.CTkLabel(form, text="Modo de gravação:").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=8)
        mode_menu = ctk.CTkOptionMenu(form, variable=self.record_mode_var, values=["toggle", "press"])
        mode_menu.grid(row=2, column=1, sticky="w", pady=8)

        hint = (
            "Use valores aceitos pela biblioteca de hotkeys (ex.: 'f3', 'ctrl+alt+s').\n"
            "No modo 'press', a gravação fica ativa enquanto a tecla estiver pressionada."
        )
        ctk.CTkLabel(frame, text=hint, justify="left", wraplength=600).pack(anchor="w", padx=16, pady=(8, 16))

        return frame

    def _build_language_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_frame)
        frame.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(frame, text="Escolha o idioma da interface", font=ctk.CTkFont(size=14, weight="bold")) \
            .pack(anchor="w", padx=16, pady=(16, 8))

        menu = ctk.CTkOptionMenu(
            frame,
            variable=self.ui_language_var,
            values=list(self._language_choices.keys()),
            width=260,
        )
        menu.pack(anchor="w", padx=16, pady=12)

        ctk.CTkLabel(
            frame,
            text=(
                "A seleção influencia apenas o idioma utilizado pelo aplicativo. "
                "Você pode alterar este parâmetro posteriormente nas configurações."
            ),
            wraplength=600,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(6, 16))

        return frame

    def _build_model_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_frame)
        frame.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(frame, text="Selecione o backend e o modelo ASR", font=ctk.CTkFont(size=14, weight="bold")) \
            .pack(anchor="w", padx=16, pady=(16, 8))

        if self._recommended_entries:
            recommendation = ctk.CTkOptionMenu(
                frame,
                variable=self.recommended_choice_var,
                values=["<Manter configuração atual>"] + [entry[0] for entry in self._recommended_entries],
                command=self._apply_recommended_model,
                width=340,
            )
            recommendation.pack(anchor="w", padx=16, pady=(4, 12))
            ctk.CTkLabel(
                frame,
                text="Selecione uma sugestão para preencher automaticamente os campos abaixo.",
                wraplength=600,
                justify="left",
            ).pack(anchor="w", padx=16)

        form = ctk.CTkFrame(frame)
        form.pack(fill="x", padx=16, pady=(12, 0))
        for i in range(2):
            form.columnconfigure(i, weight=1)

        ctk.CTkLabel(form, text="Identificador do modelo (Hugging Face):").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=8)
        model_entry = ctk.CTkEntry(form, textvariable=self.model_id_var)
        model_entry.grid(row=0, column=1, sticky="ew", pady=8)

        ctk.CTkLabel(form, text="Backend de inferência:").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=8)
        backend_menu = ctk.CTkOptionMenu(
            form,
            variable=self.backend_var,
            values=["ctranslate2"],
            command=lambda _: self._sync_quant_visibility(),
        )
        backend_menu.grid(row=1, column=1, sticky="w", pady=8)

        ctk.CTkLabel(form, text="Quantização (apenas CTranslate2):").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=8)
        quant_menu = ctk.CTkOptionMenu(
            form,
            variable=self.quant_var,
            values=["int8_float16", "int8", "float16"],
        )
        quant_menu.grid(row=2, column=1, sticky="w", pady=8)
        self._quant_menu = quant_menu
        self._sync_quant_visibility()

        ctk.CTkLabel(
            frame,
            text=(
                "Certifique-se de escolher um backend compatível com sua GPU/CPU. "
                "O download do modelo pode exigir vários gigabytes de espaço em disco."
            ),
            wraplength=600,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(12, 16))

        return frame

    def _build_paths_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_frame)
        frame.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(frame, text="Defina os diretórios principais", font=ctk.CTkFont(size=14, weight="bold")) \
            .pack(anchor="w", padx=16, pady=(16, 8))

        form = ctk.CTkFrame(frame)
        form.pack(fill="x", padx=16)
        form.columnconfigure(1, weight=1)

        self._add_path_field(
            form,
            row=0,
            label="Diretório raiz de armazenamento:",
            variable=self.storage_root_var,
        )
        self._add_path_field(
            form,
            row=1,
            label="Diretório para modelos instalados:",
            variable=self.models_dir_var,
        )
        self._add_path_field(
            form,
            row=2,
            label="Cache de modelos (download):",
            variable=self.cache_dir_var,
        )
        self._add_path_field(
            form,
            row=3,
            label="Diretório das gravações:",
            variable=self.recordings_dir_var,
        )

        ctk.CTkLabel(
            frame,
            text=(
                "Os diretórios serão criados automaticamente caso não existam. "
                "Verifique permissões de escrita antes de continuar."
            ),
            wraplength=600,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(12, 16))

        return frame

    def _build_summary_step(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._content_frame)
        frame.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(frame, text="Revise suas escolhas", font=ctk.CTkFont(size=14, weight="bold")) \
            .pack(anchor="w", padx=16, pady=(16, 8))

        summary = ctk.CTkLabel(frame, textvariable=self.summary_text_var, justify="left", wraplength=600)
        summary.pack(anchor="w", padx=16, pady=(8, 16))
        self._summary_label = summary

        download_checkbox = ctk.CTkCheckBox(
            frame,
            text="Baixar o modelo selecionado agora",
            variable=self.download_now_var,
            command=self._update_summary,
        )
        download_checkbox.pack(anchor="w", padx=16)

        return frame

    def _add_path_field(self, form: ctk.CTkFrame, *, row: int, label: str, variable: tk.StringVar) -> None:
        ctk.CTkLabel(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=8)
        entry = ctk.CTkEntry(form, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=8)

        def choose_directory(var: tk.StringVar = variable) -> None:
            initial_dir = Path(var.get()).expanduser() if var.get() else None
            selected = filedialog.askdirectory(title="Selecionar diretório", initialdir=initial_dir)
            if selected:
                var.set(selected)

        browse = ctk.CTkButton(form, text="Procurar...", width=110, command=choose_directory)
        browse.grid(row=row, column=2, padx=(12, 0), pady=8)

    # ------------------------------------------------------------------
    # Step Handling & Validation
    # ------------------------------------------------------------------
    def _show_step(self, index: int) -> None:
        index = max(0, min(index, len(self._steps) - 1))
        if self._step_index == index and self._steps[index].winfo_ismapped():
            return
        for frame in self._steps:
            frame.grid_remove()
        self._step_index = index
        step = self._steps[index]
        step.grid(row=0, column=0, sticky="nsew")
        self._title_var.set(self._titles[index])
        self._update_nav_buttons()
        if self._titles[index] == "Resumo":
            self._update_summary()

    def _update_nav_buttons(self) -> None:
        is_first = self._step_index == 0
        is_last = self._step_index == len(self._steps) - 1
        self._back_button.configure(state="disabled" if is_first else "normal")
        if is_last:
            self._next_button.grid_remove()
            self._finish_button.grid()
        else:
            self._finish_button.grid_remove()
            self._next_button.grid()

    def _on_back(self) -> None:
        self._show_step(self._step_index - 1)

    def _on_next(self) -> None:
        validator = self._validators[self._step_index]
        if validator and not validator():
            return
        self._show_step(self._step_index + 1)

    def _on_finish(self) -> None:
        validator = self._validators[self._step_index]
        if validator and not validator():
            return
        self._result = WizardResult(
            config_updates=self._collect_config_updates(),
            hotkey_preferences=self._collect_hotkey_preferences(),
            download_request=self._collect_download_request(),
        )
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()

    def _on_cancel(self) -> None:
        if messagebox.askyesno("Assistente", "Deseja cancelar a configuração inicial?", parent=self):
            self._result = None
            try:
                self.grab_release()
            except Exception:
                pass
            self.destroy()

    def _validate_hotkeys(self) -> bool:
        record = self.record_key_var.get().strip()
        agent = self.agent_key_var.get().strip()
        mode = self.record_mode_var.get().strip().lower()
        if not record or not agent:
            messagebox.showerror("Hotkeys", "Informe teclas válidas para gravação e comando agêntico.", parent=self)
            return False
        if mode not in {"toggle", "press"}:
            messagebox.showerror("Hotkeys", "Modo inválido. Utilize 'toggle' ou 'press'.", parent=self)
            return False
        return True

    def _validate_language(self) -> bool:
        choice = self.ui_language_var.get()
        if choice not in self._language_choices:
            messagebox.showerror("Idioma", "Selecione um idioma suportado.", parent=self)
            return False
        return True

    def _validate_model(self) -> bool:
        model_id = self.model_id_var.get().strip()
        backend = self.backend_var.get().strip().lower()
        if not model_id:
            messagebox.showerror("Modelo", "Informe o identificador do modelo.", parent=self)
            return False
        if backend != "ctranslate2":
            messagebox.showerror(
                "Modelo",
                "Backend inválido. Utilize 'ctranslate2'.",
                parent=self,
            )
            return False
        return True

    def _validate_paths(self) -> bool:
        entries = {
            "Diretório raiz": self.storage_root_var.get().strip(),
            "Modelos": self.models_dir_var.get().strip(),
            "Cache": self.cache_dir_var.get().strip(),
            "Gravações": self.recordings_dir_var.get().strip(),
        }
        missing = [label for label, value in entries.items() if not value]
        if missing:
            messagebox.showerror(
                "Diretórios",
                f"Informe um caminho válido para: {', '.join(missing)}.",
                parent=self,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Data collection helpers
    # ------------------------------------------------------------------
    def _collect_config_updates(self) -> dict[str, Any]:
        language_code = self._language_choices.get(self.ui_language_var.get(), "en-US")
        updates = {
            "ui_language": language_code,
            "record_key": self.record_key_var.get().strip(),
            "agent_key": self.agent_key_var.get().strip(),
            "record_mode": self.record_mode_var.get().strip().lower(),
            "asr_model_id": self.model_id_var.get().strip(),
            "asr_backend": self.backend_var.get().strip().lower(),
            "asr_ct2_compute_type": self.quant_var.get().strip(),
            "storage_root_dir": self.storage_root_var.get().strip(),
            "models_storage_dir": self.models_dir_var.get().strip(),
            "asr_cache_dir": self.cache_dir_var.get().strip(),
            "recordings_dir": self.recordings_dir_var.get().strip(),
        }
        return updates

    def _collect_hotkey_preferences(self) -> dict[str, str]:
        return {
            "record_key": self.record_key_var.get().strip(),
            "agent_key": self.agent_key_var.get().strip(),
            "record_mode": self.record_mode_var.get().strip().lower(),
        }

    def _collect_download_request(self) -> WizardDownloadRequest | None:
        if not self.download_now_var.get():
            return None
        backend = self.backend_var.get().strip().lower()
        quant = self.quant_var.get().strip() if backend == "ctranslate2" else None
        return WizardDownloadRequest(
            model_id=self.model_id_var.get().strip(),
            backend=backend,
            cache_dir=self.cache_dir_var.get().strip(),
            quant=quant,
        )

    def _update_summary(self) -> None:
        language_label = self.ui_language_var.get()
        mode = self.record_mode_var.get().strip().lower()
        quant = self.quant_var.get().strip() if self.backend_var.get().strip().lower() == "ctranslate2" else "-"
        lines = [
            f"Hotkey de gravação: {self.record_key_var.get().strip()} (modo {mode})",
            f"Hotkey agêntica: {self.agent_key_var.get().strip()}",
            f"Idioma da interface: {language_label}",
            f"Modelo ASR: {self.model_id_var.get().strip()} (backend {self.backend_var.get().strip().lower()}, quantização {quant})",
            f"Diretório raiz: {self.storage_root_var.get().strip()}",
            f"Modelos instalados: {self.models_dir_var.get().strip()}",
            f"Cache de modelos: {self.cache_dir_var.get().strip()}",
            f"Gravações: {self.recordings_dir_var.get().strip()}",
            "",
            "Download imediato: " + ("Sim" if self.download_now_var.get() else "Não"),
            "",
            (
                "Dica: alinhe o nível de privilégio do aplicativo com o destino da auto-colagem para evitar que "
                "o Windows bloqueie a automação."
            ),
        ]
        self.summary_text_var.set("\n".join(lines))

    def _apply_recommended_model(self, choice: str) -> None:
        if choice not in {entry[0] for entry in self._recommended_entries}:
            return
        for display, model_id, backend in self._recommended_entries:
            if display == choice:
                self.model_id_var.set(model_id)
                self.backend_var.set(backend)
                break
        self._sync_quant_visibility()

    def _sync_quant_visibility(self) -> None:
        backend = self.backend_var.get().strip().lower()
        if backend == "ctranslate2":
            self._quant_menu.configure(state="normal")
        else:
            self._quant_menu.configure(state="disabled")

    def _normalize_catalog(self, entries: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
        normalized: list[tuple[str, str, str]] = []
        for entry in entries:
            try:
                model_id = str(entry.get("id"))
                if not model_id:
                    continue
                backend = str(entry.get("backend", "")).lower() or "ctranslate2"
                display_name = str(entry.get("display_name") or model_id)
                normalized.append((display_name, model_id, backend))
            except Exception:
                continue
        return normalized

