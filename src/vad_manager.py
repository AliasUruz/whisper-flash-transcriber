import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime

# Caminho para o modelo ONNX do Silero VAD
MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"

# Ajuste para execucao via PyInstaller
if hasattr(sys, "_MEIPASS"):
    MODEL_PATH = Path(sys._MEIPASS) / "models" / "silero_vad.onnx"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
FAILURE_LOG_PATH = LOGS_DIR / "vad_failure.jsonl"

logging.info("VAD model path set to '%s'", MODEL_PATH)


class VADManager:
    """Gerencia a deteccao de voz usando o modelo Silero."""

    @staticmethod
    def is_model_available() -> bool:
        """Verifica se o arquivo do modelo Silero VAD esta presente."""
        return MODEL_PATH.exists()

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000, vad_pre_speech_padding_ms: int = 200, vad_post_speech_padding_ms: int = 300):
        """Inicializa o VAD Manager."""

        self.threshold = threshold
        self.sr = sampling_rate
        self.vad_pre_speech_padding_ms = vad_pre_speech_padding_ms
        self.vad_post_speech_padding_ms = vad_post_speech_padding_ms
        self.enabled = False
        self._chunk_counter = 0
        self.session = None
        self._use_energy_fallback = False
        self._fallback_notified = False

        self.pre_speech_buffer = np.array([], dtype=np.float32)
        self.post_speech_cooldown = 0

        if not self.is_model_available():
            logging.error(
                "VAD model file missing at '%s'. VAD feature disabled.",
                MODEL_PATH,
            )
            self._activate_energy_fallback("model ausente")
            return

        try:
            model_path_str = str(MODEL_PATH)
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider"]
                logging.info("CUDAExecutionProvider detected for VAD.")
            else:
                providers = ["CPUExecutionProvider"]
                logging.info("CUDAExecutionProvider unavailable; using CPUExecutionProvider.")

            self.session = onnxruntime.InferenceSession(
                model_path_str,
                providers=providers,
            )
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self.reset_states()
            self.enabled = True
            logging.info(
                "VAD model loaded successfully from '%s'.",
                model_path_str,
            )
        except Exception as exc:
            logging.error(
                "Error loading VAD model from '%s': %s",
                MODEL_PATH,
                exc,
                exc_info=True,
            )
            self.session = None
            self._activate_energy_fallback("model load failure", exc)

    def reset_states(self) -> None:
        """Reseta os estados internos do modelo."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self.pre_speech_buffer = np.array([], dtype=np.float32)
        self.post_speech_cooldown = 0

    def is_speech(self, audio_chunk: np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Retorna ``True`` se o chunk contem fala."""

        if audio_chunk is None:
            logging.debug("VAD received chunk None; assuming speech to keep recording.")
            return True, None

        self._chunk_counter += 1
        raw_array = np.asarray(audio_chunk)
        if raw_array.size == 0:
            logging.debug("VAD received an empty chunk; returning False.")
            return False, None

        prepared, peak = self._prepare_input(raw_array)
        mono_view = prepared.reshape(-1) if prepared.size else np.empty(0, dtype=np.float32)

        if self.session is None or self._use_energy_fallback:
            detected, _, _, _ = self._energy_gate(mono_view, self.threshold)
        else:
            ort_inputs = {
                "input": prepared,
                "state": self._state,
                "sr": np.array([self.sr], dtype=np.int64),
            }
            try:
                outs = self.session.run(None, ort_inputs)
                speech_prob = float(outs[0][0][0])
                self._state = outs[1]
                detected = speech_prob > self.threshold
            except Exception as exc:
                self.reset_states()
                self._log_failure(exc, prepared, {})
                self._activate_energy_fallback("inference failure", exc)
                detected, _, _, _ = self._energy_gate(mono_view, self.threshold)

        if detected:
            self.post_speech_cooldown = int(self.vad_post_speech_padding_ms / 1000 * self.sr)
            if self.pre_speech_buffer.size > 0:
                # Retorna o buffer de pre-speech e o chunk atual
                returning_buffer = np.concatenate([self.pre_speech_buffer, raw_array])
                self.pre_speech_buffer = np.array([], dtype=np.float32)
                return True, returning_buffer
            return True, raw_array
        else:
            if self.post_speech_cooldown > 0:
                self.post_speech_cooldown -= len(raw_array)
                return True, raw_array

            # Adiciona ao buffer de pre-speech
            self.pre_speech_buffer = np.concatenate([self.pre_speech_buffer, raw_array])
            max_buffer_size = int(self.vad_pre_speech_padding_ms / 1000 * self.sr)
            if self.pre_speech_buffer.size > max_buffer_size:
                self.pre_speech_buffer = self.pre_speech_buffer[-max_buffer_size:]
            return False, None


    @staticmethod
    def _energy_gate(mono: np.ndarray, threshold: float) -> tuple[bool, float, float, float]:
        if mono.size == 0:
            adjusted = float(0.005 + np.clip(threshold, 0.0, 1.0) * 0.045)
            return False, 0.0, 0.0, adjusted

        mono_arr = np.asarray(mono, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(mono_arr))) if mono_arr.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(mono_arr)))) if mono_arr.size else 0.0
        clamped_threshold = float(np.clip(threshold, 0.0, 1.0))
        adjusted = float(0.005 + clamped_threshold * 0.045)
        score = max(rms, peak * 0.6)
        detected = score >= adjusted
        return detected, peak, rms, adjusted


    @staticmethod
    def _prepare_input(audio_chunk: np.ndarray) -> tuple[np.ndarray, float]:
        if audio_chunk is None:
            raise ValueError("audio chunk is None")

        arr = np.asarray(audio_chunk)
        if arr.size == 0:
            raise ValueError("audio chunk is empty")

        if arr.ndim == 1:
            flattened = arr
        elif arr.ndim == 2:
            if 1 in arr.shape:
                flattened = arr.reshape(-1)
            else:
                channel_axis = 0 if arr.shape[0] <= arr.shape[1] else 1
                flattened = arr.mean(axis=channel_axis)
        else:
            flattened = arr.reshape(-1)

        mono = np.ascontiguousarray(flattened, dtype=np.float32).reshape(1, -1)
        peak = float(np.max(np.abs(mono)))
        if peak > 1.0:
            mono = mono / peak
        return mono, peak

    def _log_failure(self, exc: Exception, prepared: np.ndarray, raw_meta: dict) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "chunk_index": self._chunk_counter,
            "prepared_shape": list(prepared.shape),
            "prepared_dtype": str(prepared.dtype),
            "peak": raw_meta.get("raw_max_abs"),
            "raw_shape": raw_meta.get("raw_shape"),
            "raw_dtype": raw_meta.get("raw_dtype"),
            "preview": raw_meta.get("preview"),
            "exception": repr(exc),
        }

        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with FAILURE_LOG_PATH.open("a", encoding="utf-8") as fp:
                json.dump(payload, fp)
                fp.write("\n")
        except Exception:
            logging.debug("Unable to record VAD failure.", exc_info=True)

    def _activate_energy_fallback(self, reason: str, exc: Exception | None = None) -> None:
        if self._use_energy_fallback:
            return

        self._use_energy_fallback = True
        self.session = None
        self.enabled = True

        if not self._fallback_notified:
            if exc is None:
                logging.warning("VAD fallback to energy detection enabled (%s).", reason)
            else:
                logging.warning("VAD fallback to energy detection enabled (%s): %s", reason, exc)
            self._fallback_notified = True
