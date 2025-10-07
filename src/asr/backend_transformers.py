from __future__ import annotations

from typing import Any

import importlib
import logging

from ..logging_utils import get_logger, log_context

LOGGER = get_logger(__name__, component='TransformersBackend')


class TransformersBackend:
    """ASR backend based on Hugging Face Transformers."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: int | str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self.pipe = None
        self.sample_rate = 16000

    def load(
        self,
        *,
        device: int | str | None = None,
        dtype: str | None = "auto",
        cache_dir: str | None = None,
        attn_implementation: str = "sdpa",
        **kwargs,
    ) -> None:
        """Load model and processor, constructing the inference pipeline."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The Transformers backend requires the 'torch' package. Install it using the optional requirements."
            ) from exc

        model_override = kwargs.pop("model_id", None)
        if model_override:
            self.model_id = model_override

        requested_device = device if device not in (None, "auto") else self.device
        if requested_device in (None, "auto"):
            requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        resolved_device = self._resolve_device(requested_device, torch)
        torch_dtype = self._resolve_dtype(dtype, resolved_device, torch)

        LOGGER.info(
            log_context(
                "Loading Transformers ASR model.",
                event="asr.transformers_load",
                model=self.model_id,
                device=resolved_device,
                dtype=str(torch_dtype),
            )
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

        pipeline_device = self._resolve_pipeline_device(resolved_device)
        self.device = str(resolved_device)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=pipeline_device,
        )
        self.device = resolved_device
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

    @staticmethod
    def _resolve_device(device: int | str | None, torch_module: Any) -> "torch.device":
        """Normalize a user-provided device declaration into a ``torch.device``."""

        if device in (None, "auto"):
            return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")

        if isinstance(device, torch_module.device):
            return device

        if isinstance(device, int):
            return torch_module.device("cpu" if device < 0 else f"cuda:{device}")

        if isinstance(device, str):
            normalized = device.strip().lower()
            if normalized in {"cpu", "-1"}:
                return torch_module.device("cpu")
            if normalized in {"mps", "xpu"}:
                return torch_module.device(normalized)
            if normalized.isdigit():
                return torch_module.device(f"cuda:{normalized}")
            if normalized.startswith("cuda"):
                return torch_module.device(device)

        raise ValueError(f"Unsupported device specification: {device!r}")

    @staticmethod
    def _resolve_dtype(dtype: str | None, device: "torch.device", torch_module: Any) -> "torch.dtype":
        """Determine the torch dtype honoring the execution device selection."""

        if dtype in (None, "auto"):
            return torch_module.float16 if device.type == "cuda" else torch_module.float32

        normalized = dtype.lower()
        aliases = {
            "float16": "float16",
            "fp16": "float16",
            "float32": "float32",
            "fp32": "float32",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
        }
        try:
            target = aliases.get(normalized, normalized)
            return getattr(torch_module, target)
        except AttributeError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported dtype specification: {dtype!r}") from exc

    @staticmethod
    def _resolve_pipeline_device(device: "torch.device") -> object:
        """Map a resolved device into the value expected by ``pipeline``."""

        if device.type in {"cuda", "mps", "xpu"}:
            return device
        return -1
