"""Coordenador de ações entre captura de áudio e transcrição."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import soundfile as sf

from . import state_manager as sm
from .audio_handler import AUDIO_SAMPLE_RATE
from .config_manager import ConfigManager
from .logging_utils import current_correlation_id, get_logger, log_context

LOGGER = get_logger(__name__, component='ActionOrchestrator')


class ActionOrchestrator:
    """Encapsula o fluxo entre áudio, transcrição e pós-processamento."""

    def __init__(
        self,
        *,
        state_manager: sm.StateManager,
        config_manager: ConfigManager,
        transcription_handler: Any | None = None,
        clipboard_module: Any | None = None,
        paste_callback: Callable[[], None] | None = None,
        log_status_callback: Callable[[str, bool], None] | None = None,
        tk_root: Any | None = None,
        close_ui_callback: Callable[[], None] | None = None,
        fallback_text_provider: Callable[[], str] | None = None,
        reset_transcription_buffer: Callable[[], None] | None = None,
        delete_temp_audio_callback: Callable[[], None] | None = None,
    ) -> None:
        self._state_manager = state_manager
        self._config_manager = config_manager
        self._transcription_handler = transcription_handler
        self._clipboard_module = clipboard_module
        self._paste_callback = paste_callback
        self._log_status_callback = log_status_callback
        self._tk_root = tk_root
        self._close_ui_callback = close_ui_callback
        self._fallback_text_provider = fallback_text_provider
        self._reset_transcription_buffer = reset_transcription_buffer
        self._delete_temp_audio_callback = delete_temp_audio_callback

        self._agent_mode_active = False

    def bind_transcription_handler(self, handler: Any) -> None:
        """Associa o ``TranscriptionHandler`` responsável pelas transcrições."""

        self._transcription_handler = handler

    # ------------------------------------------------------------------
    # Agent mode management
    # ------------------------------------------------------------------
    def activate_agent_mode(self) -> None:
        LOGGER.debug(
            "ActionOrchestrator: agent mode activated.",
            extra={"event": "agent_mode_toggle", "status": "enabled"},
        )
        self._agent_mode_active = True

    def deactivate_agent_mode(self) -> None:
        LOGGER.debug(
            "ActionOrchestrator: agent mode deactivated.",
            extra={"event": "agent_mode_toggle", "status": "disabled"},
        )
        self._agent_mode_active = False

    @property
    def is_agent_mode_active(self) -> bool:
        return self._agent_mode_active

    # ------------------------------------------------------------------
    # Audio coordination
    # ------------------------------------------------------------------
    def on_audio_segment_ready(self, audio_source: str | np.ndarray) -> None:
        """Processa um segmento de áudio finalizado."""

        duration_seconds = self._compute_duration_seconds(audio_source)
        min_duration = float(
            self._config_manager.get("min_transcription_duration", 0.0)
        )
        if duration_seconds < min_duration:
            LOGGER.info(
                log_context(
                    "Audio segment discarded due to minimum duration threshold.",
                    event="audio.segment_discarded",
                    duration_seconds=round(duration_seconds, 2),
                    min_duration_seconds=round(min_duration, 2),
                )
            )
            self._state_manager.transition_if(
                (sm.STATE_RECORDING, sm.STATE_IDLE),
                sm.StateEvent.AUDIO_RECORDING_DISCARDED,
                details=(
                    f"Segment shorter than minimum ({duration_seconds:.2f}s < "
                    f"{min_duration:.2f}s)"
                ),
                source="audio_handler",
            )
            return

        agent_mode = self._agent_mode_active

        handler = self._transcription_handler
        if handler is None:
            LOGGER.error(
                log_context(
                    "Transcription handler is not available to receive audio.",
                    event="audio.dispatch_failed",
                    stage="transcription",
                    agent_mode=agent_mode,
                )
            )
            if agent_mode:
                self._log_status(
                    "Agent mode unavailable: wait for the model to finish loading and try again.",
                    error=True,
                )
            self._state_manager.set_state(
                sm.StateEvent.AUDIO_ERROR,
                details="Transcription handler unavailable",
                source="action_orchestrator",
            )
            return

        previous_future = getattr(handler, "transcription_future", None)
        correlation_id = current_correlation_id()
        try:
            handler.transcribe_audio_segment(
                audio_source,
                agent_mode,
                correlation_id=correlation_id,
            )
        except Exception as exc:  # pragma: no cover - defensive guard around handler
            LOGGER.error(
                "Failed to dispatch audio segment for transcription: %s",
                exc,
                exc_info=True,
                extra={"event": "dispatch_failed", "stage": "transcription"},
            )
            if agent_mode:
                self._agent_mode_active = True
                self._log_status(
                    "Failed to engage agent mode. Verify the model load status and try again.",
                    error=True,
                )
            else:
                self._agent_mode_active = False
            return

        new_future = getattr(handler, "transcription_future", None)
        successfully_enqueued = new_future is not None and new_future is not previous_future
        if not successfully_enqueued:
            if agent_mode:
                self._agent_mode_active = True
                LOGGER.warning(
                    log_context(
                        "Agent mode request preserved: transcription handler rejected audio segment (model likely unavailable).",
                        event="agent_mode.wait",
                        stage="transcription",
                    )
                )
                self._log_status(
                    "Agent mode unavailable: the model is not ready to receive commands yet.",
                    error=True,
                )
            else:
                self._agent_mode_active = False
            return

        self._agent_mode_active = False

        LOGGER.info(
            log_context(
                "Dispatching audio segment for transcription.",
                event="audio.segment_dispatched",
                stage="transcription",
                duration_seconds=round(duration_seconds, 2),
                agent_mode=agent_mode,
            )
        )

    # ------------------------------------------------------------------
    # Result handling
    # ------------------------------------------------------------------
    def handle_transcription_result(self, corrected_text: str | None, raw_text: str | None) -> None:
        """Trata o resultado final da transcrição."""

        final_text = (corrected_text or "").strip()
        if not final_text and self._fallback_text_provider:
            final_text = self._fallback_text_provider().strip()
        if not final_text and raw_text:
            final_text = raw_text.strip()

        if self._config_manager.get("display_transcripts_in_terminal", False):
            print("\n=== COMPLETE TRANSCRIPTION ===\n" + final_text + "\n==============================\n")

        self._copy_to_clipboard(final_text)

        if self._config_manager.get("auto_paste", True):
            self._paste_and_log()
        else:
            self._log_status("Transcription complete. Auto-paste disabled.")

        self._state_manager.transition_if(
            (sm.STATE_TRANSCRIBING, sm.STATE_IDLE),
            sm.StateEvent.TRANSCRIPTION_COMPLETED,
            details=f"Transcription finalized ({len(final_text)} chars)",
            source="transcription",
        )
        self._close_live_transcription_ui()
        if self._reset_transcription_buffer:
            self._reset_transcription_buffer()
        if self._delete_temp_audio_callback:
            self._delete_temp_audio_callback()

        LOGGER.info(
            "Transcription ready for consumption.",
            extra={"event": "transcription_ready", "details": f"chars={len(final_text)}"},
        )

    def handle_agent_result(self, agent_response_text: str) -> None:
        """Trata o resultado do modo agente."""

        response = (agent_response_text or "").strip()
        if not response:
            self._log_status("Agent command returned an empty response.", error=True)
            LOGGER.warning(
                "Agent command returned an empty response.",
                extra={"event": "agent_response_empty"},
            )
            return

        self._copy_to_clipboard(response)

        if self._config_manager.get("agent_auto_paste", True):
            self._paste_and_log(success_message="Agent command executed and pasted.")
        else:
            self._log_status("Agent command executed (auto-paste disabled).")

        self._state_manager.transition_if(
            (sm.STATE_TRANSCRIBING, sm.STATE_IDLE),
            sm.StateEvent.AGENT_COMMAND_COMPLETED,
            details=f"Agent response delivered ({len(response)} chars)",
            source="agent_mode",
        )
        self._close_live_transcription_ui()
        if self._delete_temp_audio_callback:
            self._delete_temp_audio_callback()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_duration_seconds(self, audio_source: str | np.ndarray) -> float:
        if isinstance(audio_source, str):
            try:
                with sf.SoundFile(audio_source) as stream:
                    return len(stream) / float(stream.samplerate)
            except Exception as exc:  # pragma: no cover - logging apenas
                LOGGER.warning("Unable to compute audio duration from file '%s': %s", audio_source, exc)
                return 0.0
        array = np.asarray(audio_source)
        if array.size == 0:
            return 0.0

        if array.ndim <= 1:
            samples = array.shape[0]
        else:
            meaningful_dims = [dim for dim in array.shape if dim > 1]
            if meaningful_dims:
                samples = max(meaningful_dims)
            else:
                samples = array.size

        return float(samples) / float(AUDIO_SAMPLE_RATE)

    def _copy_to_clipboard(self, text: str) -> None:
        if not self._clipboard_module:
            return
        try:
            self._clipboard_module.copy(text)
            LOGGER.info(
                "Text copied to clipboard.",
                extra={"event": "clipboard_copy", "details": f"chars={len(text)}"},
            )
        except Exception as exc:  # pragma: no cover - ambiente pode não suportar
            LOGGER.error("Failed to copy text to clipboard: %s", exc, exc_info=True)

    def _paste_and_log(self, success_message: str | None = None) -> None:
        if not self._paste_callback:
            LOGGER.debug(
                "Paste callback not configured; skipping auto-paste.",
                extra={"event": "auto_paste_skipped", "status": "callback_missing"},
            )
            return
        try:
            self._paste_callback()
            if success_message:
                self._log_status(success_message)
            else:
                self._log_status("Text pasted.")
        except Exception as exc:  # pragma: no cover - dependente de ambiente
            LOGGER.error(
                "Failed to simulate paste action: %s",
                exc,
                exc_info=True,
                extra={"event": "auto_paste_failed"},
            )
            self._log_status("Failed to paste content.", error=True)

    def _log_status(self, message: str, *, error: bool = False) -> None:
        callback = self._log_status_callback
        if callback:
            callback(message, error)
        else:
            log_func = LOGGER.error if error else LOGGER.info
            log_func(message)

    def _close_live_transcription_ui(self) -> None:
        if not self._close_ui_callback:
            return
        if self._tk_root:
            self._tk_root.after(0, self._close_ui_callback)
        else:
            try:
                self._close_ui_callback()
            except Exception:  # pragma: no cover - fail-safe
                LOGGER.debug("Close UI callback raised an exception.", exc_info=True)
