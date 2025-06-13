"""Pacote de interface do usuário para o Whisper Transcriber.

Este módulo inicializa componentes principais e facilita a importação
centralizada de janelas e widgets reutilizáveis. A refatoração
visa separar a lógica de UI da camada de controle,
conforme descrito em ``update_plans/UI_REFACTORING.md``.
"""

from .main_window import MainWindow
from .settings_window import SettingsWindow

__all__ = ["MainWindow", "SettingsWindow"]

