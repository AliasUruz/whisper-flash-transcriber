"""Janela de configurações do Whisper Transcriber."""

from __future__ import annotations

import customtkinter as ctk


class SettingsWindow(ctk.CTkToplevel):
    """Protótipo da janela de configurações."""

    def __init__(self, master: ctk.CTk | None = None, *args, **kwargs) -> None:
        super().__init__(master=master, *args, **kwargs)
        self.title("Configurações")
        self.geometry("500x400")
        self.transient(master)
        label = ctk.CTkLabel(self, text="Configurações em construção")
        label.pack(padx=20, pady=20)

