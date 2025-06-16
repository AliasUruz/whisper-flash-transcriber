import logging
import numpy as np
import onnxruntime
import torch

class VADManager:
    """Gerencia a detecção de voz usando o modelo Silero."""

    def __init__(self, model_path: str = "silero_vad.onnx", threshold: float = 0.5, sampling_rate: int = 16000):
        try:
            self.session = onnxruntime.InferenceSession(model_path)
            self.threshold = threshold
            self.sr = sampling_rate
            self.reset_states()
            logging.info(f"Modelo VAD carregado de '{model_path}'.")
        except Exception as exc:
            logging.error(f"Erro ao carregar o modelo VAD '{model_path}': {exc}", exc_info=True)
            raise

    def reset_states(self) -> None:
        """Reseta os estados internos do modelo."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Retorna True se o chunk contém fala."""
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError("audio_chunk deve ser um np.ndarray")
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        ort_inputs = {
            "input": torch.from_numpy(audio_chunk).unsqueeze(0).numpy(),
            "h": self._h,
            "c": self._c,
            "sr": np.array([self.sr], dtype=np.int64),
        }
        outs = self.session.run(None, ort_inputs)
        speech_prob = float(outs[0][0][0])
        self._h, self._c = outs[1], outs[2]
        return speech_prob > self.threshold

