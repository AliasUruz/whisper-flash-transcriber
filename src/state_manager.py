"""
This file will contain the StateManager class and related state management logic.
"""
from threading import RLock
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum, auto, unique
from collections.abc import Callable, Iterable, Mapping

from .logging_utils import get_logger, log_context, log_duration

LOGGER = get_logger('whisper_flash_transcriber.state', component='StateManager')

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
    MODEL_DOWNLOAD_PROGRESS = auto()
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
    DEPENDENCY_AUDIT_READY = auto()
    TEXT_CORRECTION_BREAKER_TRIPPED = auto()
    TEXT_CORRECTION_BREAKER_RESET = auto()


@dataclass(frozen=True)
class StateNotification:
    """Mensagem estruturada propagada para assinantes de mudanças de estado."""

    event: StateEvent | None
    state: str
    previous_state: str | None = None
    details: object | None = None
    source: str | None = None
    operation_id: str | None = None


STATE_FOR_EVENT: dict[StateEvent, str] = {
    StateEvent.MODEL_MISSING: STATE_ERROR_MODEL,
    StateEvent.MODEL_CACHE_INVALID: STATE_ERROR_MODEL,
    StateEvent.MODEL_PROMPT_FAILED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_DECLINED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_STARTED: STATE_LOADING_MODEL,
    StateEvent.MODEL_DOWNLOAD_CANCELLED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_INVALID_CACHE: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_FAILED: STATE_ERROR_MODEL,
    StateEvent.MODEL_DOWNLOAD_PROGRESS: STATE_LOADING_MODEL,
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
    StateEvent.DEPENDENCY_AUDIT_READY: STATE_LOADING_MODEL,
    StateEvent.TEXT_CORRECTION_BREAKER_TRIPPED: STATE_IDLE,
    StateEvent.TEXT_CORRECTION_BREAKER_RESET: STATE_IDLE,
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
    StateEvent.MODEL_DOWNLOAD_PROGRESS: "Model download progress updated",
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
    StateEvent.DEPENDENCY_AUDIT_READY: "Dependency audit completed",
    StateEvent.TEXT_CORRECTION_BREAKER_TRIPPED: "Text correction provider temporarily disabled",
    StateEvent.TEXT_CORRECTION_BREAKER_RESET: "Text correction provider restored",
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
        self._last_detail_signature: object | None = None
        self._state_lock = RLock()
        self._subscribers: list[Callable[[StateNotification], None]] = []
        self._logger = LOGGER.bind(manager_id=f"state-{id(self):x}")

    @staticmethod
    def _describe_callback(callback: Callable[[StateNotification], None]) -> str:
        if hasattr(callback, "__qualname__"):
            return callback.__qualname__  # type: ignore[return-value]
        if hasattr(callback, "__name__"):
            return callback.__name__  # type: ignore[return-value]
        return repr(callback)

    def subscribe(self, callback: Callable[[StateNotification], None]):
        """Adds a subscriber to be notified of state changes."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
            self._logger.debug(
                log_context(
                    "Subscriber registered for state updates.",
                    event="state.subscriber_registered",
                    total_subscribers=len(self._subscribers),
                    subscriber=self._describe_callback(callback),
                )
            )

    def _notify_subscribers(self, notification: StateNotification):
        """Notifies all subscribers of a state change."""
        event_name = notification.event.name if notification.event else None
        log_details_payload = {
            "state": notification.state,
            "event_name": event_name,
        }
        if notification.operation_id is not None:
            log_details_payload["operation_id"] = notification.operation_id

        with log_duration(
            self._logger,
            "Dispatching state notification.",
            event="state.notification_dispatch",
            details=log_details_payload,
        ) as log_details:
            log_details["subscriber_count"] = len(self._subscribers)
            had_errors = False
            for callback in self._subscribers:
                try:
                    if self.main_tk_root:
                        self.main_tk_root.after(0, lambda c=callback, n=notification: c(n))
                    else:
                        callback(notification)
                except Exception:
                    had_errors = True
                    self._logger.error(
                        log_context(
                            "State subscriber raised an exception.",
                            event="state.subscriber_error",
                            subscriber=self._describe_callback(callback),
                            state=notification.state,
                            event_name=event_name,
                        ),
                        exc_info=True,
                    )
            if had_errors:
                log_details["status"] = "partial"

    def _resolve_transition(
        self,
        event: StateEvent | str,
        detail_payload: object | None,
    ) -> tuple[StateEvent | None, str, str | None, object | None]:
        event_obj: StateEvent | None
        mapped_state: str
        message: str | None

        if isinstance(event, StateEvent):
            event_obj = event
            try:
                mapped_state = STATE_FOR_EVENT[event_obj]
            except KeyError as exc:
                raise ValueError(f"No state mapping defined for event {event_obj!r}") from exc
            if isinstance(detail_payload, Mapping):
                message = detail_payload.get("message") or EVENT_DEFAULT_DETAILS.get(event_obj)
            else:
                message = detail_payload or EVENT_DEFAULT_DETAILS.get(event_obj)
        elif isinstance(event, str):
            normalized = event.strip().upper()
            if normalized not in LEGACY_STATE_DEFAULT_DETAILS:
                raise ValueError(f"Unsupported state event payload: {event!r}")
            event_obj = None
            mapped_state = normalized
            if isinstance(detail_payload, Mapping):
                message = detail_payload.get("message") or LEGACY_STATE_DEFAULT_DETAILS.get(normalized)
            else:
                message = detail_payload or LEGACY_STATE_DEFAULT_DETAILS.get(normalized)
        else:
            raise ValueError(f"Unsupported state event payload: {event!r}")

        return event_obj, mapped_state, message, detail_payload

    def _apply_transition_locked(
        self,
        *,
        event_obj: StateEvent | None,
        mapped_state: str,
        detail_payload: object | None,
        message: str | None,
        source: str | None,
        operation_id: str | None,
    ) -> tuple[StateNotification, str] | None:
        """Apply a transition while ``_state_lock`` is held.

        The method updates the tracked state and returns the generated
        :class:`StateNotification` alongside the previous state. When the
        incoming request would produce the same event, mapped state, resolved
        ``operation_id`` (explicit argument or derived from ``details``) and
        normalized detail payload (including message fallbacks) as the most
        recent notification, it suppresses the transition and returns ``None``
        instead of emitting a duplicate notification.
        """
        previous_state = self._current_state

        resolved_operation_id: str | None = None
        if isinstance(operation_id, str):
            candidate = operation_id.strip()
            if candidate:
                resolved_operation_id = candidate

        if resolved_operation_id is None:
            if isinstance(detail_payload, Mapping):
                raw_operation_id = detail_payload.get("operation_id")
                if isinstance(raw_operation_id, str):
                    candidate = raw_operation_id.strip()
                    if candidate:
                        resolved_operation_id = candidate
            elif hasattr(detail_payload, "operation_id"):
                raw_operation_id = getattr(detail_payload, "operation_id")
                if isinstance(raw_operation_id, str):
                    candidate = raw_operation_id.strip()
                    if candidate:
                        resolved_operation_id = candidate

        last_event = self._last_notification.event if self._last_notification else None
        last_state = self._last_notification.state if self._last_notification else None
        last_operation_id = (
            self._last_notification.operation_id if self._last_notification else None
        )
        detail_value = detail_payload if detail_payload is not None else message
        detail_signature = self._derive_detail_signature(detail_value)
        resolved_operation_id = self._resolve_operation_id(operation_id, detail_payload)
        if (
            last_event == event_obj
            and last_state == mapped_state
            and last_operation_id == resolved_operation_id
            and self._last_detail_signature == detail_signature
        ):
            self._logger.debug(
                log_context(
                    "Duplicate state notification suppressed.",
                    event="state.duplicate_suppressed",
                    state=mapped_state,
                    event_name=event_obj.name if event_obj else None,
                    source=source,
                    operation_id=resolved_operation_id,
                )
            )
            return None

        notification = StateNotification(
            event=event_obj,
            state=mapped_state,
            previous_state=previous_state,
            details=detail_payload if detail_payload is not None else message,
            source=source,
            operation_id=resolved_operation_id,
        )
        self._current_state = mapped_state
        self._last_notification = notification
        self._last_detail_signature = detail_signature
        return notification, previous_state

    @staticmethod
    def _normalize_operation_id(candidate: object | None) -> str | None:
        if isinstance(candidate, str):
            normalized = candidate.strip()
            if normalized:
                return normalized
        return None

    def _resolve_operation_id(
        self, explicit_operation_id: str | None, detail_payload: object | None
    ) -> str | None:
        if isinstance(detail_payload, Mapping):
            detail_resolved = self._normalize_operation_id(
                detail_payload.get("operation_id")
            )
        elif hasattr(detail_payload, "operation_id"):
            detail_resolved = self._normalize_operation_id(
                getattr(detail_payload, "operation_id")
            )
        else:
            detail_resolved = None

        if detail_resolved is not None:
            return detail_resolved

        return self._normalize_operation_id(explicit_operation_id)

    def _derive_detail_signature(self, details: object | None) -> object | None:
        """Build a hashable representation of ``details`` for deduplication."""

        if details is None:
            return None

        if isinstance(details, Mapping):
            return (
                "mapping",
                self._normalize_mapping_signature(details.items()),
            )

        if isinstance(details, (list, tuple)):
            return (
                type(details).__name__,
                tuple(self._derive_detail_signature(value) for value in details),
            )

        if isinstance(details, (set, frozenset)):
            normalized = [self._derive_detail_signature(value) for value in details]
            normalized.sort(key=self._signature_sort_key)
            return (
                type(details).__name__,
                tuple(normalized),
            )

        if is_dataclass(details):
            return (
                "dataclass",
                type(details).__name__,
                self._derive_detail_signature(asdict(details)),
            )

        if isinstance(details, bytes):
            return ("bytes", details)

        if isinstance(details, (str, int, float, bool)):
            return ("scalar", details)

        if hasattr(details, "__dict__"):
            try:
                return (
                    "object",
                    type(details).__name__,
                    self._normalize_mapping_signature(vars(details).items()),
                )
            except TypeError:
                # Objects without a normal ``vars`` mapping fall through to repr.
                pass

        return ("repr", repr(details))

    @staticmethod
    def _signature_sort_key(signature: object) -> str:
        try:
            return repr(signature)
        except Exception:
            return f"<{type(signature).__name__}:{id(signature)}>"

    def _normalize_mapping_signature(
        self, items: Iterable[tuple[object, object]]
    ) -> tuple[tuple[object, object], ...]:
        normalized: list[tuple[object, object]] = []
        for key, value in items:
            key_signature = self._derive_detail_signature(key)
            value_signature = self._derive_detail_signature(value)
            normalized.append((key_signature, value_signature))

        normalized.sort(key=lambda item: self._signature_sort_key(item[0]))
        return tuple(normalized)

    def _emit_transition(
        self,
        notification: StateNotification,
        *,
        previous_state: str,
        event_obj: StateEvent | None,
        mapped_state: str,
        message: str | None,
        source: str | None,
    ) -> None:
        origin_label = event_obj.name if event_obj else f"STATE:{mapped_state}"
        log_fields = {
            "previous_state": previous_state,
            "current_state": mapped_state,
            "origin": origin_label,
            "message": message,
            "source": source,
        }
        if notification.operation_id is not None:
            log_fields["operation_id"] = notification.operation_id
        self._logger.info(
            log_context(
                "Application state transitioned.",
                event="state.transition",
                **log_fields,
            )
        )
        self._notify_subscribers(notification)

    def set_state(
        self,
        event: StateEvent | str,
        *,
        details: object | None = None,
        source: str | None = None,
        operation_id: str | None = None,
    ):
        """Applies a state transition and notifies subscribers.

        When ``operation_id`` is provided it is attached to the resulting
        :class:`StateNotification` and emitted log records.
        """

        event_obj, mapped_state, message, detail_payload = self._resolve_transition(event, details)

        with self._state_lock:
            result = self._apply_transition_locked(
                event_obj=event_obj,
                mapped_state=mapped_state,
                detail_payload=detail_payload,
                message=message,
                source=source,
                operation_id=operation_id,
            )
            if result is None:
                return
            notification, previous_state = result

        self._emit_transition(
            notification,
            previous_state=previous_state,
            event_obj=event_obj,
            mapped_state=mapped_state,
            message=message,
            source=source,
        )

    def _normalize_expected_states(
        self,
        expected_state: str | StateEvent | Iterable[str | StateEvent],
    ) -> set[str]:
        if isinstance(expected_state, StateEvent):
            return {STATE_FOR_EVENT[expected_state]}
        if isinstance(expected_state, str):
            normalized = expected_state.strip().upper()
            if not normalized:
                raise ValueError("expected_state must not be empty")
            return {normalized}

        try:
            candidates = list(expected_state)
        except TypeError as exc:  # pragma: no cover - defensive path
            raise TypeError(
                "expected_state must be a state string, StateEvent, or iterable of them."
            ) from exc

        normalized_states: set[str] = set()
        for candidate in candidates:
            if isinstance(candidate, StateEvent):
                normalized_states.add(STATE_FOR_EVENT[candidate])
            elif isinstance(candidate, str):
                normalized = candidate.strip().upper()
                if normalized:
                    normalized_states.add(normalized)
            elif candidate is None:
                continue
            else:
                text = str(candidate).strip().upper()
                if text:
                    normalized_states.add(text)

        if not normalized_states:
            raise ValueError("expected_state must resolve to at least one state value")

        return normalized_states

    def transition_if(
        self,
        expected_state: str | StateEvent | Iterable[str | StateEvent],
        event: StateEvent | str,
        *,
        details: object | None = None,
        source: str | None = None,
        operation_id: str | None = None,
    ) -> bool:
        """Conditionally apply a transition when the current state matches ``expected_state``.

        This helper should be used by code paths where multiple threads may signal
        competing transitions (for example, hotkey handlers racing against audio or
        transcription workers). By guarding the transition with the expected
        current state we avoid reverting a newer state and reduce race-condition
        windows when coordinating long-running operations.

        When ``operation_id`` is provided it is attached to the resulting
        :class:`StateNotification` and emitted log records.
        """

        event_obj, mapped_state, message, detail_payload = self._resolve_transition(event, details)
        expected_states = self._normalize_expected_states(expected_state)

        with self._state_lock:
            current_state = self._current_state
            if current_state not in expected_states:
                self._logger.debug(
                    log_context(
                        "State transition guard rejected request.",
                        event="state.transition_guard_blocked",
                        expected_states=sorted(expected_states),
                        current_state=current_state,
                        requested_state=mapped_state,
                        requested_event=event_obj.name if event_obj else None,
                        source=source,
                        operation_id=operation_id,
                    )
                )
                return False

            result = self._apply_transition_locked(
                event_obj=event_obj,
                mapped_state=mapped_state,
                detail_payload=detail_payload,
                message=message,
                source=source,
                operation_id=operation_id,
            )
            if result is None:
                return True
            notification, previous_state = result

        self._emit_transition(
            notification,
            previous_state=previous_state,
            event_obj=event_obj,
            mapped_state=mapped_state,
            message=message,
            source=source,
        )
        return True

    def get_current_state(self) -> str:
        """Returns the current state."""
        with self._state_lock:
            return self._current_state

    def is_transcribing(self) -> bool:
        """Indicates if the current state is TRANSCRIBING."""
        with self._state_lock:
            return self._current_state == STATE_TRANSCRIBING
