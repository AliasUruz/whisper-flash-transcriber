from typing import Dict, Type, Protocol


class ASRBackend(Protocol):
    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        ...


class WhisperBackend:
    """Backend padrão utilizando pipeline HuggingFace."""

    def __init__(self, handler):
        self.handler = handler

    def load(self) -> None:
        self.handler._initialize_model_and_processor()

    def unload(self) -> None:
        self.handler.unload()

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        generate_kwargs = {"task": "transcribe", "language": None}
        return self.handler.pipe(
            audio_source,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )


class DummyBackend:
    """Backend de exemplo sem funcionalidade real."""

    def __init__(self, handler):
        self.handler = handler

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def transcribe(self, audio_source, *, chunk_length_s: float, batch_size: int):
        return {"text": ""}


backend_registry: Dict[str, Type[ASRBackend]] = {
    "whisper": WhisperBackend,
    "dummy": DummyBackend,
}
