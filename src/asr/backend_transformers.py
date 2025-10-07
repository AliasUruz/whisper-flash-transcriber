from __future__ import annotations

from typing import Any

from ..logging_utils import get_logger, log_context

LOGGER = get_logger(__name__, component='TransformersBackend')


class TransformersBackend:
    """Legacy placeholder for the deprecated Transformers backend."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: int | str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.pipe = None
        self.sample_rate = 16000

    def load(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - legacy guard
        LOGGER.error(
            log_context(
                "Attempted to use the deprecated Transformers backend.",
                event="asr.transformers_deprecated",
                model=self.model_id,
            )
        )
        raise RuntimeError(
            "The Transformers backend is no longer bundled. Configure the application to use the CTranslate2 runtime."
        )

    def warmup(self) -> None:  # pragma: no cover - legacy guard
        return

    def transcribe(self, audio: Any, **kwargs: Any) -> dict:  # pragma: no cover - legacy guard
        raise RuntimeError(
            "The Transformers backend is no longer available. Configure the application to use the CTranslate2 runtime."
        )

    def unload(self) -> None:  # pragma: no cover - legacy guard
        self.pipe = None
        self.model = None
        self.processor = None
