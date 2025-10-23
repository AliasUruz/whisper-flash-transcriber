from __future__ import annotations

import math
import inspect
from dataclasses import is_dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any, Callable, Iterator


class FasterWhisperBackend:
    """ASR backend powered by faster-whisper."""

    def __init__(
        self,
        model_id: str = "whisper-large-v3",
        device: str | int | None = "auto",
        *,
        device_index: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.device_index = _coerce_device_index(device_index)
        self.model = None
        self._transcribe_signature_info: dict[str, Any] | None = None
        self._last_requested_batch_size: int | None = None

    def load(self, ct2_compute_type: str = "default", cache_dir: str | None = None, **kwargs) -> None:
        """Load the WhisperModel with the given compute type."""
        from faster_whisper import WhisperModel

        if "model_id" in kwargs:
            self.model_id = kwargs.pop("model_id")

        device_override = kwargs.pop("device", None)
        if device_override is not None:
            self.device = device_override

        device_index_override = kwargs.pop("device_index", None)
        coerced_index = _coerce_device_index(device_index_override)
        if coerced_index is not None:
            self.device_index = coerced_index

        cpu_threads_override = kwargs.pop("cpu_threads", None)
        cpu_threads = _coerce_positive_int(cpu_threads_override)
        if cpu_threads is not None:
            kwargs["cpu_threads"] = cpu_threads

        normalized_device, normalized_index = _normalize_device_spec(
            self.device, self.device_index
        )

        if normalized_device == "auto":
            normalized_device = "cuda" if _has_cuda() else "cpu"

        if normalized_device != "cuda":
            normalized_index = None
        elif normalized_index is None and _has_cuda():
            normalized_index = 0

        runtime_device = "cuda" if normalized_device == "cuda" else normalized_device

        self.device = _compose_device_label(normalized_device, normalized_index)
        self.device_index = normalized_index

        if ct2_compute_type == "default":
            ct2_compute_type = "int8_float16" if normalized_device == "cuda" else "int8"

        model_name = self.model_id
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        if model_name.startswith("whisper-"):
            model_name = model_name.replace("whisper-", "", 1)

        model_source: str | None = None
        download_root: str | None = cache_dir or None
        if cache_dir:
            try:
                cache_path = Path(cache_dir).expanduser()
                from .. import model_manager as model_manager_module

                canonical_dir = model_manager_module.get_installation_dir(
                    cache_path,
                    "ctranslate2",
                    self.model_id,
                )
                candidate_dirs = [canonical_dir]
                for alias in ("ctranslate2", "faster-whisper"):
                    candidate = cache_path / alias / self.model_id
                    if candidate not in candidate_dirs:
                        candidate_dirs.append(candidate)

                for candidate in candidate_dirs:
                    if _looks_like_ct2_installation(candidate):
                        model_source = str(candidate)
                        download_root = None
                        break
            except Exception:
                model_source = None

        if download_root is not None:
            download_root = str(download_root)

        model_kwargs: dict[str, Any] = {
            "device": runtime_device,
            "compute_type": ct2_compute_type,
            "download_root": download_root,
        }
        model_kwargs.update(kwargs)

        if normalized_device == "cuda" and normalized_index is not None:
            model_kwargs["device_index"] = normalized_index

        self.model = WhisperModel(
            model_source or model_name,
            **model_kwargs,
        )

    def warmup(self) -> None:
        """Run a dummy inference to initialize the model."""
        if not self.model:
            return
        import numpy as np

        audio = np.zeros(int(16000 * 0.015), dtype="float32")
        self.stream_transcribe(audio)

    def transcribe(
        self,
        audio: Any,
        *,
        on_segment: Callable[[str], None] | None = None,
        cancel_event: Event | None = None,
        **kwargs,
    ) -> dict:
        """Transcribe the provided audio and return just the text."""
        text, _ = self.stream_transcribe(
            audio,
            on_segment=on_segment,
            cancel_event=cancel_event,
            **kwargs,
        )
        return {"text": text}

    def stream_transcribe(
        self,
        audio: Any,
        *,
        on_segment: Callable[..., None] | None = None,
        cancel_event: Event | None = None,
        **kwargs,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Iterate over streaming segments, optionally emitting callbacks."""
        if not self.model:
            raise RuntimeError("Backend not loaded")

        chunk_length = _coerce_chunk_length(kwargs.pop("chunk_length_s", None))
        if chunk_length is not None:
            kwargs["chunk_length"] = chunk_length

        batch_size = kwargs.pop("batch_size", None)
        sanitized_segments = _sanitize_language_detection_segments(
            kwargs.get("language_detection_segments")
        )
        if sanitized_segments is not None:
            kwargs["language_detection_segments"] = sanitized_segments
        else:
            computed_segments = _segments_from_chunk_length(kwargs.get("chunk_length"))
            if computed_segments is not None:
                kwargs["language_detection_segments"] = computed_segments

        if batch_size is not None:
            info = self._ensure_transcribe_signature()
            if info["accepts_batch_size"]:
                kwargs["batch_size"] = batch_size
            elif info["decode_option_param"] is not None:
                option_key = info["decode_option_param"]
                kwargs[option_key] = _inject_batch_size_into_decode_options(
                    kwargs.get(option_key), batch_size
                )
            else:
                self._transcribe_signature_info = info
            self._last_requested_batch_size = batch_size

        segment_iterator, _ = self.model.transcribe(audio, **kwargs)
        iterator: Iterator[Any] = iter(segment_iterator)

        try:
            current = next(iterator)
        except StopIteration:
            return "", []

        aggregated_parts: list[str] = []
        metadata: list[dict[str, Any]] = []

        while True:
            if cancel_event is not None and cancel_event.is_set():
                break

            segment_info = _segment_to_metadata(current)
            aggregated_parts.append(current.text)
            metadata.append(segment_info)

            try:
                next_segment = next(iterator)
                has_next = True
            except StopIteration:
                next_segment = None
                has_next = False

            cancel_requested = cancel_event.is_set() if cancel_event is not None else False
            segment_info["is_final"] = bool(not has_next and not cancel_requested)

            if on_segment is not None:
                callback_payload = segment_info.copy()
                try:
                    on_segment(
                        current.text,
                        metadata=callback_payload,
                        is_final=segment_info["is_final"],
                    )
                except TypeError:
                    on_segment(current.text)

            if not has_next:
                break

            current = next_segment

        text = " ".join(part for part in aggregated_parts if part).strip()
        return text, metadata

    def unload(self) -> None:
        """Release model resources."""
        self.model = None

    def _ensure_transcribe_signature(self) -> dict[str, Any]:
        cached = getattr(self, "_transcribe_signature_info", None)
        if cached is not None:
            return cached
        info = _describe_transcribe_signature(getattr(self.model, "transcribe", None))
        self._transcribe_signature_info = info
        return info


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _coerce_device_index(raw: Any) -> int | None:
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return None
    return index if index >= 0 else None


def _coerce_positive_int(raw: Any) -> int | None:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _normalize_device_spec(device: Any, index: int | None) -> tuple[str, int | None]:
    if isinstance(device, str):
        normalized = device.strip()
        lowered = normalized.lower()
        if lowered in {"", "auto"}:
            return "auto", index
        if lowered in {"cpu", "-1"}:
            return "cpu", None
        if lowered.startswith("cuda"):
            _, _, suffix = lowered.partition(":")
            parsed_index = _coerce_device_index(suffix) if suffix else None
            if parsed_index is not None:
                index = parsed_index
            return "cuda", index
        return normalized, index

    if hasattr(device, "type"):
        dev_type = getattr(device, "type", None)
        dev_index = getattr(device, "index", None)
        if dev_type == "cuda":
            parsed_index = _coerce_device_index(dev_index)
            return "cuda", parsed_index if parsed_index is not None else index
        if dev_type == "cpu":
            return "cpu", None
        if dev_type is not None:
            return str(dev_type), index

    if isinstance(device, int):
        return ("cuda", device) if device >= 0 else ("cpu", None)

    return "auto", index


def _compose_device_label(device: str, index: int | None) -> str:
    if device == "cuda" and index is not None:
        return f"cuda:{index}"
    return device


def _coerce_chunk_length(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        chunk = float(raw)
    except (TypeError, ValueError):
        return None
    if math.isnan(chunk) or math.isinf(chunk):
        return None
    return max(0.0, chunk)


def _segment_to_metadata(segment: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": getattr(segment, "id", None),
        "start": getattr(segment, "start", None),
        "end": getattr(segment, "end", None),
        "text": getattr(segment, "text", ""),
        "avg_logprob": getattr(segment, "avg_logprob", None),
        "no_speech_prob": getattr(segment, "no_speech_prob", None),
        "temperature": getattr(segment, "temperature", None),
        "compression_ratio": getattr(segment, "compression_ratio", None),
        "language": getattr(segment, "language", None),
    }
    tokens = getattr(segment, "tokens", None)
    if tokens is not None:
        try:
            info["tokens"] = list(tokens)
        except TypeError:
            info["tokens"] = tokens
    words = getattr(segment, "words", None)
    if words is not None:
        try:
            info["words"] = [
                {
                    "start": getattr(word, "start", None),
                    "end": getattr(word, "end", None),
                    "word": getattr(word, "word", None),
                    "prob": getattr(word, "probability", getattr(word, "prob", None)),
                }
                for word in words
            ]
        except TypeError:
            info["words"] = words
    return info


def _sanitize_language_detection_segments(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        value = int(round(float(raw)))
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return 1
    return value


def _segments_from_chunk_length(chunk_length: Any) -> int | None:
    if chunk_length is None:
        return None
    try:
        chunk = float(chunk_length)
    except (TypeError, ValueError):
        return None
    if chunk <= 0:
        return 1
    approx_segments = chunk / 30.0
    return _sanitize_language_detection_segments(approx_segments)


def _looks_like_ct2_installation(path: Path) -> bool:
    try:
        if not path.exists() or not path.is_dir():
            return False
    except Exception:
        return False

    has_config = False
    has_weights = False
    try:
        iterator = path.rglob("*")
    except Exception:
        return False

    for candidate in iterator:
        if not candidate.is_file():
            continue
        name = candidate.name.lower()
        if name == "config.json":
            has_config = True
            continue
        if name in {"model.bin", "model.onnx", "model.safetensors"} or (
            name.endswith((".bin", ".onnx", ".safetensors")) and "model" in name
        ):
            has_weights = True
        if has_config and has_weights:
            return True
    return has_config and has_weights


def _describe_transcribe_signature(transcribe: Callable[..., Any] | None) -> dict[str, Any]:
    info = {
        "accepts_batch_size": False,
        "decode_option_param": None,
    }
    if transcribe is None:
        return info
    try:
        signature = inspect.signature(transcribe)
    except (TypeError, ValueError):
        return info

    parameters = signature.parameters
    if "batch_size" in parameters:
        info["accepts_batch_size"] = True

    for candidate in (
        "decode_options",
        "decoding_options",
        "decoder_options",
        "transcription_options",
    ):
        if candidate in parameters:
            info["decode_option_param"] = candidate
            break
    return info


def _inject_batch_size_into_decode_options(options: Any, batch_size: int) -> Any:
    if options is None:
        factory = _resolve_decode_options_factory()
        if factory is None:
            return {"batch_size": batch_size}
        try:
            return factory(batch_size=batch_size)
        except TypeError:
            # Fallback to dictionary if signature mismatch.
            return {"batch_size": batch_size}

    if isinstance(options, dict):
        updated = dict(options)
        updated["batch_size"] = batch_size
        return updated

    if is_dataclass(options):
        try:
            return replace(options, batch_size=batch_size)
        except TypeError:
            pass

    if hasattr(options, "batch_size"):
        try:
            setattr(options, "batch_size", batch_size)
            return options
        except Exception:
            pass

    return options


def _resolve_decode_options_factory() -> Callable[..., Any] | None:
    candidates = (
        "DecodingOptions",
        "DecodeOptions",
        "DecoderOptions",
        "TranscriptionOptions",
    )
    for name in candidates:
        try:
            module = __import__("faster_whisper.transcribe", fromlist=[name])
            factory = getattr(module, name, None)
        except Exception:
            factory = None
        if factory is None:
            continue
        if callable(factory):
            return factory
    return None
