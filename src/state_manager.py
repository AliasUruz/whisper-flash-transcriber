"""
This file will contain the StateManager class and related state management logic.
"""
import logging
from threading import RLock
from dataclasses import dataclass
from enum import Enum, auto, unique
from collections.abc import Callable

LOGGER = logging.getLogger('whisper_flash_transcriber.state')

# Estados da aplicação
STATE_IDLE = "IDLE"
STATE_LOADING_MODEL = "LOADING_MODEL"
STATE_RECORDING = "RECORDING"
STATE_TRANSCRIBING = "TRANSCRIBING"
STATE_ERROR_MODEL = "ERROR_MODEL"
STATE_ERROR_AUDIO = "ERROR_AUDIO"
STATE_ERROR_TRANSCRIPTION = "ERROR_TRANSCRIPTION"
STATE_ERROR_SETTINGS = "ERROR_SETTINGS"

LEGACY_STATE_DEFAULT_DETAILS: dict[str, str] = {
    STATE_IDLE: "State transitioned to IDLE",
    STATE_LOADING_MODEL: "State transitioned to LOADING_MODEL",
    STATE_RECORDING: "State transitioned to RECORDING",
    STATE_TRANSCRIBING: "State transitioned to TRANSCRIBING",
    STATE_ERROR_MODEL: "State transitioned to ERROR_MODEL",
    STATE_ERROR_AUDIO: "State transitioned to ERROR_AUDIO",
    STATE_ERROR_TRANSCRIPTION: "State transitioned to ERROR_TRANSCRIPTION",
    STATE_ERROR_SETTINGS: "State transitioned to ERROR_SETTINGS",
}

@unique
class StateEvent(Enum):
    """Eventos normalizados utilizados para transições de estado."""

    MODEL_MISSING = auto()
    MODEL_CACHE_INVALID = auto()
    MODEL_PROMPT_FAILED = auto()
    MODEL_DOWNLOAD_DECLINED = auto()
    MODEL_DOWNLOAD_STARTED = auto()
    MODEL_DOWNLOAD_CANCELLED = auto()
    MODEL_DOWNLOAD_INVALID_CACHE = auto()
    MODEL_DOWNLOAD_FAILED = auto()
    MODEL_CACHE_NOT_CONFIGURED = auto()
    MODEL_CACHE_MISSING = auto()
    MODEL_READY = auto()
    MODEL_LOADING_FAILED = auto()
    AUDIO_RECORDING_STARTED = auto()
    AUDIO_RECORDING_STOPPED = auto()
    AUDIO_RECORDING_DISCARDED = auto()
    AUDIO_ERROR = auto()
    TRANSCRIPTION_STARTED = auto()
    TRANSCRIPTION_COMPLETED = auto()
    AGENT_COMMAND_COMPLETED = auto()
    SETTINGS_MISSING_RECORD_KEY = auto()
    SETTINGS_HOTKEY_START_FAILED = auto()
    SETTINGS_REREGISTER_FAILED = auto()
    SETTINGS_RECOVERED = auto()  # Hotkeys se recuperaram e voltaram a operar


@dataclass(frozen=True)
class StateNotification:
    """Mensagem estruturada propagada para assinantes de mudanças de estado."""

    event: StateEvent | None
    state: str
    previous_state: str | None = None
    details: str | None = None
    source: str | None = None


STATE_FOR_EVENT: dict[StateEvent, str] = {
    StateEvent.MODEL_MISSING: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_INVALID: STATE_ERROR_MODEL,
    StateEvent.MODEL_PROMPT_FAILED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_DECLINED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_STARTED: STATE_LOADING_MODEL,
    StateEvent.MODEL_DOWNLOAD_CANCELLED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_INVALID_CACHE: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_FAILED: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_NOT_CONFIGURED: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_MISSING: STATE_ERROR_MODEL,
    StateEvent.MODEL_READY: STATE_IDLE,
    StateEvent.MODEL_LOADING_FAILED: STATE_ERROR_MODEL,
    StateEvent.AUDIO_RECORDING_STARTED: STATE_RECORDING,
    StateEvent.AUDIO_RECORDING_STOPPED: STATE_IDLE,
    StateEvent.AUDIO_RECORDING_DISCARDED: STATE_IDLE,
    StateEvent.AUDIO_ERROR: STATE_ERROR_AUDIO,
    StateEvent.TRANSCRIPTION_STARTED: STATE_TRANSCRIBING,
    StateEvent.TRANSCRIPTION_COMPLETED: STATE_IDLE,
    StateEvent.AGENT_COMMAND_COMPLETED: STATE_IDLE,
    StateEvent.SETTINGS_MISSING_RECORD_KEY: STATE_ERROR_SETTINGS,
    StateEvent.SETTINGS_HOTKEY_START_FAILED: STATE_ERROR_SETTINGS,
    StateEvent.SETTINGS_REREGISTER_FAILED: STATE_ERROR_SETTINGS,
    StateEvent.SETTINGS_RECOVERED: STATE_IDLE,
}


EVENT_DEFAULT_DETAILS: dict[StateEvent, str] = {
    StateEvent.MODEL_MISSING: "ASR model not found locally",
    StateEvent.MODEL_CACHE_INVALID: "Configured ASR cache directory is invalid",
    StateEvent.MODEL_PROMPT_FAILED: "Failed to prompt user about model download",
    StateEvent.MODEL_DOWNLOAD_DECLINED: "User declined automatic model download",
    StateEvent.MODEL_DOWNLOAD_STARTED: "Starting model download",
    StateEvent.MODEL_DOWNLOAD_CANCELLED: "Model download was cancelled",
    StateEvent.MODEL_DOWNLOAD_INVALID_CACHE: "Model download aborted due to invalid cache directory",
    StateEvent.MODEL_DOWNLOAD_FAILED: "Model download failed",
    StateEvent.MODEL_CACHE_NOT_CONFIGURED: "ASR cache directory not configured",
    StateEvent.MODEL_CACHE_MISSING: "ASR cache directory missing on disk",
    StateEvent.MODEL_READY: "Model loaded successfully",
    StateEvent.MODEL_LOADING_FAILED: "Model failed to load",
    StateEvent.AUDIO_RECORDING_STARTED: "AudioHandler started recording",
    StateEvent.AUDIO_RECORDING_STOPPED: "AudioHandler returned to idle",
    StateEvent.AUDIO_RECORDING_DISCARDED: "Recorded audio discarded before transcription",
    StateEvent.AUDIO_ERROR: "Audio subsystem reported an error",
    StateEvent.TRANSCRIPTION_STARTED: "Transcription pipeline started",
    StateEvent.TRANSCRIPTION_COMPLETED: "Transcription finished",
    StateEvent.AGENT_COMMAND_COMPLETED: "Agent command completed",
    StateEvent.SETTINGS_MISSING_RECORD_KEY: "Record hotkey is not configured",
    StateEvent.SETTINGS_HOTKEY_START_FAILED: "KeyboardHotkeyManager failed to start",
    StateEvent.SETTINGS_REREGISTER_FAILED: "Hotkey re-registration failed",
    StateEvent.SETTINGS_RECOVERED: "Recovered stable hotkey registration",
}


AUDIO_STATE_EVENT_MAP: dict[str, StateEvent] = {
    "RECORDING": StateEvent.AUDIO_RECORDING_STARTED,
    "TRANSCRIBING": StateEvent.TRANSCRIPTION_STARTED,
    "IDLE": StateEvent.AUDIO_RECORDING_STOPPED,
    "ERROR_AUDIO": StateEvent.AUDIO_ERROR,
}

class StateManager:
    def __init__(self, initial_state: str, main_tk_root=None):
        self.main_tk_root = main_tk_root
        self._current_state = initial_state
        self._last_notification: StateNotification | None = None
        self._state_lock = RLock()
        self._subscribers: list[Callable[[StateNotification], None]] = []

    def subscribe(self, callback: Callable[[StateNotification], None]):
        """Adds a subscriber to be notified of state changes."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def _notify_subscribers(self, notification: StateNotification):
        """Notifies all subscribers of a state change."""
        for callback in self._subscribers:
            try:
                if self.main_tk_root:
                    self.main_tk_root.after(0, lambda c=callback, n=notification: c(n))
                else:
                    callback(notification)
            except Exception as e:
                LOGGER.error(f"Error notifying subscriber: {e}", exc_info=True)

    def set_state(self, event: StateEvent | str, *, details: str | None = None, source: str | None = None):
        """Applies a state transition and notifies subscribers."""
        event_obj: StateEvent | None
        mapped_state: str
        message: str | None

        if isinstance(event, StateEvent):
            event_obj = event
            try:
                mapped_state = STATE_FOR_EVENT[event_obj]
            except KeyError as exc:
                raise ValueError(f"No state mapping defined for event {event_obj!r}") from exc
            message = details or EVENT_DEFAULT_DETAILS.get(event_obj)
        elif isinstance(event, str):
            normalized = event.strip().upper()
            if normalized not in LEGACY_STATE_DEFAULT_DETAILS:
                raise ValueError(f"Unsupported state event payload: {event!r}")
            event_obj = None
            mapped_state = normalized
            message = details or LEGACY_STATE_DEFAULT_DETAILS.get(normalized)
        else:
            raise ValueError(f"Unsupported state event payload: {event!r}")

        with self._state_lock:
            previous_state = self._current_state
            last_event = self._last_notification.event if self._last_notification else None
            last_state = self._last_notification.state if self._last_notification else None
            if last_event == event_obj and last_state == mapped_state:
                LOGGER.debug(
                    "Duplicate state event %s suppressed (state=%s, source=%s).",
                    event_obj.name if event_obj else mapped_state,
                    mapped_state,
                    source,
                )
                return

            notification = StateNotification(
                event=event_obj,
                state=mapped_state,
                previous_state=previous_state,
                details=message,
                source=source,
            )
            self._current_state = mapped_state
            self._last_notification = notification

        origin_label = event_obj.name if event_obj else f"STATE:{mapped_state}"
        transition_log = f"State transition via {origin_label}: {previous_state} -> {mapped_state}"
        if message:
            transition_log += f" ({message})"
        if source:
            transition_log += f" [source={source}]"
        LOGGER.info(transition_log)

        self._notify_subscribers(notification)

    def get_current_state(self) -> str:
        """Returns the current state."""
        with self._state_lock:
            return self._current_state

    def is_transcribing(self) -> bool:
        """Indicates if the current state is TRANSCRIBING."""
        with self._state_lock:
            return self._current_state == STATE_TRANSCRIBING
