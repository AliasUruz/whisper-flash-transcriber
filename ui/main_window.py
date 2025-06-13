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
    """Implementação preliminar da janela principal."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.title("Whisper Transcriber")
        self.geometry("600x400")
        # Elementos serão adicionados conforme avanço da refatoração
        label = ctk.CTkLabel(self, text="Janela Principal em construção")
        label.pack(padx=20, pady=20)

