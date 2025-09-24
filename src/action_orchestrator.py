"""Coordena ações pós-processamento como clipboard/paste e limpeza de áudio.

Este módulo encapsula o fluxo compartilhado entre transcrição e modo agente,
permitindo que o ``AppCore`` delegue responsabilidades operacionais (copiar
texto, colar automaticamente, disparar eventos de estado e fechar a UI) a uma
classe dedicada que pode ser testada de forma isolada.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pyperclip

from .config_manager import ConfigManager


class ActionOrchestrator:
    """Implementa as rotinas de entrega de texto e limpeza de recursos."""

    def __init__(
        self,
        config_manager: ConfigManager,
        *,
        status_logger: Callable[..., None],
        state_dispatcher: Callable[..., None],
        paste_callback: Callable[[], None],
        ui_close_callback: Callable[[], None] | None,
        temp_audio_cleaner: Callable[[], None],
    ) -> None:
        self._config_manager = config_manager
        self._log_status = status_logger
        self._dispatch_state = state_dispatcher
        self._paste_callback = paste_callback
        self._ui_close_callback = ui_close_callback
        self._temp_audio_cleaner = temp_audio_cleaner

    def handle_agent_result(self, agent_response_text: str | None, *, state_event: Any) -> None:
        """Entrega o resultado do modo agente respeitando colagem automática."""
        normalized_response = agent_response_text or ""

        try:
            if not normalized_response:
                logging.warning("Comando do agente retornou uma resposta vazia.")
                self._log_status("Comando do agente sem resposta.", error=True)
                return

            try:
                pyperclip.copy(normalized_response)
                logging.info("Agent response copied to clipboard.")
            except Exception as clipboard_error:  # pragma: no cover - integração com SO
                logging.error(
                    "Erro ao copiar resposta do agente para o clipboard: %s",
                    clipboard_error,
                    exc_info=True,
                )
                self._log_status(
                    "Erro ao copiar resposta do agente para o clipboard.",
                    error=True,
                )

            if self._config_manager.get("agent_auto_paste", True):
                self._paste_callback()
                self._log_status("Comando do agente executado e colado.")
            else:
                self._log_status("Comando do agente executado (colagem automática desativada).")

        except Exception as exc:  # pragma: no cover - integrações dependentes do SO
            logging.error("Erro ao manusear o resultado do agente: %s", exc, exc_info=True)
            self._log_status(f"Erro ao manusear o resultado do agente: {exc}", error=True)
        finally:
            response_size = len(normalized_response)
            self._dispatch_state(
                state_event,
                details=f"Agent response delivered ({response_size} chars)",
                source="agent_mode",
            )
            if self._ui_close_callback:
                try:
                    self._ui_close_callback()
                except Exception as exc:  # pragma: no cover - apenas loga
                    logging.debug(
                        "Falha ao agendar fechamento da janela de transcrição ao final do modo agente: %s",
                        exc,
                        exc_info=True,
                    )
            self._temp_audio_cleaner()

__all__ = ["ActionOrchestrator"]
