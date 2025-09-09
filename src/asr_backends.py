from typing import Protocol, Dict, Type

class ASRBackend(Protocol):
    def load(self) -> None:
        ...

    def unload(self) -> None:
        ...

class WhisperBackend:
    """Backend padrão utilizando pipeline HuggingFace."""

    def __init__(self, handler):
        self.handler = handler

    def load(self) -> None:
        self.handler._initialize_model_and_processor()

    def unload(self) -> None:
        self.handler.unload()

class DummyBackend:
    """Backend de exemplo sem funcionalidade real."""

    def __init__(self, handler):
        self.handler = handler

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

backend_registry: Dict[str, Type[ASRBackend]] = {
    "whisper": WhisperBackend,
    "dummy": DummyBackend,
}
