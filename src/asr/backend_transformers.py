from __future__ import annotations

from typing import Any


class TransformersBackend:
    """ASR backend based on Hugging Face Transformers."""

    def __init__(self) -> None:
        self.model_id = ""
        self.device: int | str | None = None
        self.processor = None
        self.model = None
        self.pipe = None
        self.sample_rate = 16000

    def load(
        self,
        *,
        model_id: str,
        device: int | str | None = None,
        dtype: str | None = "auto",
        cache_dir: str | None = None,
        attn_implementation: str = "sdpa",
        **_,
    ) -> None:
        """Load model and processor, constructing the inference pipeline."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
        import torch

        self.model_id = model_id
        self.device = device

        device = device if device not in (None, "auto") else (
            "cuda:0" if torch.cuda.is_available() else -1
        )
        torch_dtype = (
            torch.float16
            if (device != -1 and (dtype in (None, "auto", "float16", "fp16")))
            else torch.float32
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id, cache_dir=cache_dir)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=attn_implementation,
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=(0 if device != -1 else -1),
        )
        try:
            self.sample_rate = int(
                self.processor.feature_extractor.sampling_rate
            )  # type: ignore[attr-defined]
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
