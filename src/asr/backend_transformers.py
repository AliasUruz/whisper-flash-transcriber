from __future__ import annotations

from typing import Any


class TransformersBackend:
    """ASR backend based on Hugging Face Transformers."""

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: int | str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.pipe = None
        self.sample_rate = 16000

    def load(self, *args, **kwargs) -> None:
        """Load model and processor, constructing the inference pipeline.

        Extra positional or keyword arguments are accepted for compatibility
        with the :class:`ASRBackend` protocol but are ignored.
        """
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            processor=self.processor,
            device=self.device,
        )
        try:
            self.sample_rate = int(self.processor.feature_extractor.sampling_rate)  # type: ignore[attr-defined]
        except Exception:
            self.sample_rate = 16000

    def warmup(self) -> None:
        """Run a small inference pass with 15 ms of silence to initialize kernels."""
        if not self.pipe:
            return
        import numpy as np

        samples = int(self.sample_rate * 0.015)
        audio = np.zeros(samples, dtype="float32")
        self.pipe(audio)

    def transcribe(self, audio: Any, **kwargs) -> dict:
        """Transcribe the provided audio and return a dictionary with text."""
        if not self.pipe:
            raise RuntimeError("Backend not loaded")
        result = self.pipe(audio, **kwargs)
        return {"text": result.get("text", "")}

    def unload(self) -> None:
        """Free model resources."""
        self.pipe = None
        self.model = None
        self.processor = None
