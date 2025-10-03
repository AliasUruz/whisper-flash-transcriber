import json
import logging
import sys
from collections import deque
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

MIN_VAD_INPUT_SAMPLES = 512

logging.info("VAD model path set to '%s'", MODEL_PATH)


class VADManager:
    """Gerencia a detecção de voz usando o modelo Silero."""

    @staticmethod
    def is_model_available() -> bool:
        """Verifica se o arquivo do modelo Silero VAD está presente."""
        return MODEL_PATH.exists()

    def __init__(
        self,
        threshold: float | None = None,
        sampling_rate: int = 16000,
        pre_speech_padding_ms: int | None = None,
        post_speech_padding_ms: int | None = None,
        *,
        config_manager=None,
    ):
        """Inicializa o VAD Manager."""

        self.config_manager = config_manager
        self.sr = int(sampling_rate)
        self.threshold = float(
            self._resolve_config_value("vad_threshold", threshold, fallback=0.5)
        )
        self.pre_speech_padding_ms = int(
            max(0, self._resolve_config_value("vad_pre_speech_padding_ms", pre_speech_padding_ms, fallback=150))
        )
        self.post_speech_padding_ms = int(
            max(0, self._resolve_config_value("vad_post_speech_padding_ms", post_speech_padding_ms, fallback=300))
        )
        self.enabled = False
        self._chunk_counter = 0
        self.session = None
        self._use_energy_fallback = False
        self._fallback_notified = False
        self._pre_buffer: deque[np.ndarray] = deque()
        self._pre_buffer_samples = 0
        self._speech_active = False
        self._post_silence_samples = 0
        self._sanitize_padding()
        self._update_padding_samples()

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
        self._pre_buffer.clear()
        self._pre_buffer_samples = 0
        self._speech_active = False
        self._post_silence_samples = 0

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Retorna ``True`` se o chunk contém fala."""

        pre_padding_ms, post_padding_ms = self._ensure_runtime_state()

        if audio_chunk is None:
            logging.debug("VAD received chunk None; assuming speech to keep recording.")
            return True

        self._chunk_counter += 1
        raw_array = np.asarray(audio_chunk)
        if raw_array.size == 0:
            logging.debug("VAD received an empty chunk; returning False.")
            return False

        prepared, _ = self._prepare_input(raw_array)
        mono_view = prepared.reshape(-1) if prepared.size else np.empty(0, dtype=np.float32)

        if self.session is None or self._use_energy_fallback:
            detected, _, _, _ = self._energy_gate(mono_view, self.threshold)
            return detected

        vad_input = prepared
        if vad_input.shape[1] < MIN_VAD_INPUT_SAMPLES:
            pad = MIN_VAD_INPUT_SAMPLES - vad_input.shape[1]
            vad_input = np.pad(vad_input, ((0, 0), (0, pad)), mode="constant")

        ort_inputs = {
            "input": vad_input,
            "state": self._state,
            "sr": np.array([self.sr], dtype=np.int64),
        }
        try:
            outs = self.session.run(None, ort_inputs)
            speech_prob = float(outs[0][0][0])
            self._state = outs[1]
            return speech_prob > self.threshold
        except Exception as exc:
            self.reset_states()
            raw_meta = {
                "raw_max_abs": float(np.max(np.abs(raw_array))) if raw_array.size else 0.0,
                "raw_shape": list(raw_array.shape),
                "raw_dtype": str(raw_array.dtype),
                "preview": raw_array.reshape(-1)[:10].tolist(),
            }
            self._log_failure(exc, prepared, raw_meta)
            self._activate_energy_fallback("inference failure", exc)
            detected, _, _, _ = self._energy_gate(mono_view, self.threshold)
            return detected

    def configure(
        self,
        *,
        threshold: float | None = None,
        pre_padding_ms: int | None = None,
        post_padding_ms: int | None = None,
    ) -> None:
        """Atualiza parâmetros do VAD em tempo de execução."""

        if threshold is not None:
            self.threshold = float(threshold)
        elif self.config_manager is not None:
            try:
                self.threshold = float(self.config_manager.get("vad_threshold", self.threshold))
            except Exception:
                logging.debug("Failed to read VAD threshold from config manager.", exc_info=True)

        if pre_padding_ms is not None:
            self.pre_speech_padding_ms = max(0, int(pre_padding_ms))
        elif self.config_manager is not None:
            try:
                value = self.config_manager.get("vad_pre_speech_padding_ms", self.pre_speech_padding_ms)
                self.pre_speech_padding_ms = max(0, int(value))
            except Exception:
                logging.debug("Failed to read VAD pre padding from config manager.", exc_info=True)

        if post_padding_ms is not None:
            self.post_speech_padding_ms = max(0, int(post_padding_ms))
        elif self.config_manager is not None:
            try:
                value = self.config_manager.get("vad_post_speech_padding_ms", self.post_speech_padding_ms)
                self.post_speech_padding_ms = max(0, int(value))
            except Exception:
                logging.debug("Failed to read VAD post padding from config manager.", exc_info=True)
        self._update_padding_samples()

    def process_chunk(self, chunk: np.ndarray) -> tuple[bool, list[np.ndarray]]:
        """Avalia um chunk e retorna (``is_speech``, ``frames``).

        ``is_speech`` indica se o chunk está dentro de uma janela ativa de fala
        (incluindo o período de pós-fala). ``frames`` contém os buffers que devem
        ser persistidos imediatamente. Quando não há fala detectada o método
        retorna ``(False, [])`` e o chunk é mantido no buffer de pré-fala.
        """

        frames: list[np.ndarray] = []
        speech_detected = self.is_speech(chunk)
        chunk_array = np.asarray(chunk, dtype=np.float32).copy()

        if speech_detected:
            if not self._speech_active:
                frames.extend(self._drain_pre_buffer())
                self._speech_active = True
            self._post_silence_samples = 0
            frames.append(chunk_array)
            return True, frames

        if self._speech_active:
            self._post_silence_samples += len(chunk_array)
            frames.append(chunk_array)
            if self._post_silence_samples >= self._post_padding_samples:
                self._speech_active = False
                self._post_silence_samples = 0
            return True, frames

        self._append_to_pre_buffer(chunk_array)
        return False, frames

    def enable_energy_fallback(self, reason: str, exc: Exception | None = None) -> None:
        """Expõe o modo de fallback por energia para o pipeline externo."""

        self._activate_energy_fallback(reason, exc)

    def _resolve_config_value(self, key: str, override, *, fallback):
        if override is not None:
            return override
        if self.config_manager is None:
            return fallback
        try:
            return self.config_manager.get(key, fallback)
        except Exception:
            logging.debug(
                "Failed to read '%s' from config manager. Using fallback %s.",
                key,
                fallback,
                exc_info=True,
            )
            return fallback


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

        mono_array = np.ascontiguousarray(flattened, dtype=np.float32).reshape(1, -1)
        peak = float(np.max(np.abs(mono_array))) if mono_array.size else 0.0
        if peak > 1.0:
            mono_array = mono_array / peak
        return mono_array, peak

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

    def _append_to_pre_buffer(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        self._pre_buffer.append(chunk.copy())
        self._pre_buffer_samples += len(chunk)
        while self._pre_buffer_samples > self._pre_padding_samples and self._pre_buffer:
            removed = self._pre_buffer.popleft()
            self._pre_buffer_samples -= len(removed)

    def _drain_pre_buffer(self) -> list[np.ndarray]:
        if not self._pre_buffer:
            return []
        drained = list(self._pre_buffer)
        self._pre_buffer.clear()
        self._pre_buffer_samples = 0
        return [frame.copy() for frame in drained]

    def _update_padding_samples(self) -> None:
        sr = int(getattr(self, "sr", 16000) or 16000)
        pre_ms = max(0, int(getattr(self, "pre_speech_padding_ms", 0) or 0))
        post_ms = max(0, int(getattr(self, "post_speech_padding_ms", 0) or 0))
        self._pre_padding_samples = int(sr * (pre_ms / 1000.0))
        self._post_padding_samples = int(sr * (post_ms / 1000.0))

    def _sanitize_padding(self) -> tuple[int, int]:
        """Normaliza os valores de padding vindos da configuração ou de ajustes dinâmicos."""

        def _coerce(raw: int | float | None) -> int:
            try:
                value = int(float(raw))
            except (TypeError, ValueError):
                return 0
            return max(0, value)

        pre_raw = getattr(self, "pre_speech_padding_ms", None)
        if pre_raw is None:
            pre_raw = getattr(self, "vad_pre_speech_padding_ms", None)
        post_raw = getattr(self, "post_speech_padding_ms", None)
        if post_raw is None:
            post_raw = getattr(self, "vad_post_speech_padding_ms", None)

        pre_ms = _coerce(pre_raw)
        post_ms = _coerce(post_raw)

        self.pre_speech_padding_ms = pre_ms
        self.post_speech_padding_ms = post_ms
        self.vad_pre_speech_padding_ms = pre_ms
        self.vad_post_speech_padding_ms = post_ms
        return pre_ms, post_ms

    def _ensure_runtime_state(self) -> tuple[int, int]:
        """Garante que atributos críticos existam mesmo sem ``__init__``."""

        pre_ms, post_ms = self._sanitize_padding()
        if not hasattr(self, "pre_speech_buffer"):
            self.pre_speech_buffer = np.array([], dtype=np.float32)
        if not hasattr(self, "post_speech_cooldown"):
            self.post_speech_cooldown = 0
        if not hasattr(self, "_speech_active"):
            self._speech_active = False
        if not hasattr(self, "_post_silence_samples"):
            self._post_silence_samples = 0
        if not hasattr(self, "_pre_buffer"):
            self._pre_buffer = deque()
            self._pre_buffer_samples = 0
        elif not hasattr(self, "_pre_buffer_samples"):
            self._pre_buffer_samples = sum(len(frame) for frame in self._pre_buffer)
        self._update_padding_samples()
        return pre_ms, post_ms
