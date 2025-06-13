"""Janela principal do Whisper Transcriber.

Esta classe representa a interface simplificada
que controlará gravação e transcrição de áudio.
O objetivo inicial é migrar a lógica atualmente
espalhada em ``whisper_tkinter.py`` para uma estrutura
orientada a objetos, permitindo testes isolados.
"""

from __future__ import annotations

import customtkinter as ctk


class MainWindow(ctk.CTk):
    """Janela principal com controles básicos."""

    def __init__(self, core, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.core = core
        self.title("Whisper Transcriber")
        self.geometry("600x400")

        self.record_btn = ctk.CTkButton(
            self,
            text="Iniciar Gravação",
            command=self._toggle_recording,
        )
        self.record_btn.pack(padx=20, pady=10)

        self.settings_btn = ctk.CTkButton(
            self,
            text="Configurações",
            command=self._open_settings,
        )
        self.settings_btn.pack(padx=20, pady=10)

    def _toggle_recording(self) -> None:
        if self.core.is_recording:
            self.core.stop_recording()
            self.record_btn.configure(text="Iniciar Gravação")
        else:
            self.core.start_recording()
            self.record_btn.configure(text="Parar Gravação")

    def _open_settings(self) -> None:
        from whisper_tkinter import on_settings_menu_click
        on_settings_menu_click()

