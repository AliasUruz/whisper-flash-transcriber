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

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        """Inicializa o VAD Manager."""

        self.threshold = threshold
        self.sr = sampling_rate
        self.enabled = False
        self._chunk_counter = 0
        self.session = None
        self._use_energy_fallback = False
        self._fallback_notified = False

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
            self._activate_energy_fallback("falha ao carregar modelo", exc)

    def reset_states(self) -> None:
        """Reseta os estados internos do modelo."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Retorna ``True`` se o chunk contem fala."""

        if audio_chunk is None:
            logging.debug("VAD recebeu chunk None; assumindo fala para manter gravacao.")
            return True

        self._chunk_counter += 1
        raw_array = np.asarray(audio_chunk)
        if raw_array.size == 0:
            logging.debug("VAD recebeu chunk vazio; retornando False.")
            return False

        raw_meta = {
            "raw_shape": list(raw_array.shape),
            "raw_dtype": str(raw_array.dtype),
            "raw_max_abs": float(np.max(np.abs(raw_array))),
            "preview": raw_array.flatten()[:16].astype(float).tolist(),
        }

        prepared, peak = self._prepare_input(raw_array)
        mono_view = prepared.reshape(-1) if prepared.size else np.empty(0, dtype=np.float32)

        if self.session is None or self._use_energy_fallback:
            detected, energy_peak, rms, threshold = self._energy_gate(mono_view, self.threshold)
            logging.debug(
                "VAD energy fallback chunk %s -> detected=%s peak=%.4f rms=%.4f threshold=%.4f",
                self._chunk_counter,
                detected,
                energy_peak,
                rms,
                threshold,
            )
            return detected

        ort_inputs = {
            "input": prepared,
            "state": self._state,
            "sr": np.array([self.sr], dtype=np.int64),
        }

        try:
            outs = self.session.run(None, ort_inputs)
        except Exception as exc:
            self.reset_states()
            self._log_failure(exc, prepared, raw_meta)
            self._activate_energy_fallback("falha de inferencia", exc)
            detected, energy_peak, rms, threshold = self._energy_gate(mono_view, self.threshold)
            logging.debug(
                "VAD energy fallback chunk %s (apos falha) -> detected=%s peak=%.4f rms=%.4f threshold=%.4f",
                self._chunk_counter,
                detected,
                energy_peak,
                rms,
                threshold,
            )
            return detected

        speech_prob = float(outs[0][0][0])
        self._state = outs[1]

        logging.debug(
            "VAD chunk %s -> prob=%.4f peak=%.4f shape=%s",
            self._chunk_counter,
            speech_prob,
            peak,
            prepared.shape,
        )
        return speech_prob > self.threshold


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
            logging.debug("Nao foi possivel registrar falha do VAD.", exc_info=True)

    def _activate_energy_fallback(self, reason: str, exc: Exception | None = None) -> None:
        if self._use_energy_fallback:
            return

        self._use_energy_fallback = True
        self.session = None
        self.enabled = True

        if not self._fallback_notified:
            if exc is None:
                logging.warning("VAD fallback para deteccao por energia ativado (%s).", reason)
            else:
                logging.warning("VAD fallback para deteccao por energia ativado (%s): %s", reason, exc)
            self._fallback_notified = True
