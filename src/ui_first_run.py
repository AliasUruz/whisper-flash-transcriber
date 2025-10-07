"""UI primitives for the first-run ASR model selection dialog."""

from __future__ import annotations

from typing import Any, Sequence

import customtkinter as ctk


_DEFAULT_WRAP_LENGTH = 520


def _format_option_metadata(option: dict[str, Any]) -> str:
    """Build a multi-line description for a curated model option."""

    backend_label = option.get("backend_label") or option.get("backend") or "-"
    quant_label = option.get("variant_label") or option.get("quantization_label") or "Padrão"
    size_label = option.get("formatted_size") or "?"
    file_count = option.get("estimated_file_count")
    if isinstance(file_count, int) and file_count > 0:
        size_label = f"{size_label} ({file_count} arquivos)"

    min_vram = option.get("min_vram_label") or "-"
    download_label = option.get("download_time_label") or "-"
    assumed_speed = option.get("assumed_download_speed_mib_s")
    if download_label != "-" and isinstance(assumed_speed, (int, float)) and assumed_speed > 0:
        download_label = f"{download_label} @ {assumed_speed:.0f} MiB/s"

    notes_parts: list[str] = []
    for candidate in (
        option.get("variant_description"),
        option.get("description"),
        option.get("notes"),
    ):
        text = (candidate or "").strip()
        if text and text not in notes_parts:
            notes_parts.append(text)

    metadata_lines = [
        f"Backend recomendado: {backend_label}",
        f"Quantização: {quant_label}",
        f"Tamanho estimado: {size_label}",
        f"VRAM mínima: {min_vram}",
        f"Download estimado: {download_label}",
    ]

    if notes_parts:
        metadata_lines.append("\n".join(notes_parts))

    return "\n".join(metadata_lines)


class _ModelInstallDialog(ctk.CTkToplevel):
    """Modal dialog that lists curated ASR models for first-run selection."""

    def __init__(
        self,
        master,
        options: Sequence[dict[str, Any]],
        *,
        initial_option_id: str | None = None,
        help_text: str | None = None,
        focus_model_id: str | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Instalar modelo Whisper")
        self.resizable(False, True)
        self.attributes("-topmost", True)
        self.option_map = {opt["option_id"]: dict(opt) for opt in options}
        self.result: dict[str, Any] | None = None

        if not initial_option_id or initial_option_id not in self.option_map:
            initial_option_id = next(iter(self.option_map.keys()))

        self._selected_option = ctk.StringVar(value=initial_option_id)

        self.grid_columnconfigure(0, weight=1)
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        main_frame.grid_columnconfigure(0, weight=1)

        if help_text:
            help_label = ctk.CTkLabel(
                main_frame,
                text=help_text,
                justify="left",
                wraplength=_DEFAULT_WRAP_LENGTH,
            )
            help_label.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        scroll_frame = ctk.CTkScrollableFrame(main_frame, width=_DEFAULT_WRAP_LENGTH + 40)
        scroll_frame.grid(row=1, column=0, sticky="nsew")
        scroll_frame.grid_columnconfigure(0, weight=1)

        for option in options:
            option_id = option["option_id"]
            recommended = bool(option.get("variant_recommended") or option.get("is_recommended"))
            title_suffix = " (Recomendado)" if recommended else ""
            display_label = option.get("display_name", option.get("model_id", option_id))
            variant_label = option.get("variant_label") or option.get("quantization_label")
            if variant_label:
                title_text = f"{display_label} — {variant_label}{title_suffix}"
            else:
                title_text = f"{display_label}{title_suffix}"

            container = ctk.CTkFrame(scroll_frame)
            container.grid_columnconfigure(0, weight=1)
            container.pack(fill="x", pady=6, padx=6)

            radio = ctk.CTkRadioButton(
                container,
                text=title_text,
                value=option_id,
                variable=self._selected_option,
                anchor="w",
                justify="left",
                wraplength=_DEFAULT_WRAP_LENGTH,
            )
            radio.grid(row=0, column=0, sticky="w")

            details = _format_option_metadata(option)
            detail_label = ctk.CTkLabel(
                container,
                text=details,
                justify="left",
                anchor="w",
                wraplength=_DEFAULT_WRAP_LENGTH,
            )
            detail_label.grid(row=1, column=0, sticky="ew", padx=(28, 4), pady=(4, 0))

        button_frame = ctk.CTkFrame(main_frame)
        button_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        button_frame.grid_columnconfigure((0, 1), weight=1)

        cancel_button = ctk.CTkButton(button_frame, text="Cancelar", command=self._on_cancel)
        cancel_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        confirm_button = ctk.CTkButton(button_frame, text="Instalar modelo", command=self._on_confirm)
        confirm_button.grid(row=0, column=1, sticky="ew")

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _event: self._on_cancel())
        self.bind("<Return>", lambda _event: self._on_confirm())

        self.after_idle(self._finalize_layout, focus_model_id)

        self.transient(master)
        self.grab_set()

    def _finalize_layout(self, focus_model_id: str | None) -> None:
        """Finalize layout adjustments and focus the selected option."""

        self.update_idletasks()
        selected_id = self._selected_option.get()
        widget = None
        for child in self.winfo_children():
            if isinstance(child, ctk.CTkFrame):
                for sub_child in child.winfo_children():
                    if isinstance(sub_child, ctk.CTkFrame):
                        for radio in sub_child.winfo_children():
                            if isinstance(radio, ctk.CTkRadioButton) and radio.cget("value") == selected_id:
                                widget = radio
                                break
                    if widget:
                        break
            if widget:
                break

        if widget:
            widget.focus_set()

        if focus_model_id:
            self.title(f"Instalar modelo Whisper — {focus_model_id}")

    def _on_confirm(self) -> None:
        selected = self._selected_option.get()
        self.result = self.option_map.get(selected)
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


def prompt_model_installation(
    master,
    options: Sequence[dict[str, Any]],
    *,
    initial_option_id: str | None = None,
    focus_model_id: str | None = None,
    help_text: str | None = None,
) -> dict[str, Any] | None:
    """Render a modal dialog that lets the user choose a curated ASR model."""

    if not options:
        return None

    dialog = _ModelInstallDialog(
        master,
        options,
        initial_option_id=initial_option_id,
        help_text=help_text,
        focus_model_id=focus_model_id,
    )
    master.wait_window(dialog)
    return dialog.result
