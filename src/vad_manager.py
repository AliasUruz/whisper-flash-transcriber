import logging
import numpy as np
import onnxruntime
from pathlib import Path

# Construir um caminho absoluto para o modelo ONNX, relativo a este arquivo.
# Isso garante que o modelo seja encontrado independentemente do diretório de trabalho.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"

class VADManager:
    """Gerencia a detecção de voz usando o modelo Silero."""

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        """
        Inicializa o VAD Manager. O caminho do modelo é determinado internamente
        para garantir robustez.
        """
        try:
            # Garante que o caminho seja uma string para a sessão ONNX
            model_path_str = str(MODEL_PATH)
            self.session = onnxruntime.InferenceSession(model_path_str)
            self.threshold = threshold
            self.sr = sampling_rate
            self.reset_states()
            logging.info(f"Modelo VAD carregado com sucesso de '{model_path_str}'.")
        except FileNotFoundError:
            logging.error(f"Arquivo do modelo VAD não encontrado no caminho esperado: {MODEL_PATH}")
            logging.error("Certifique-se de que o arquivo 'silero_vad.onnx' existe em 'src/models/'.")
            raise
        except Exception as exc:
            logging.error(f"Erro ao carregar o modelo VAD de '{MODEL_PATH}': {exc}", exc_info=True)
            raise

    def reset_states(self) -> None:
        """Reseta os estados internos do modelo."""
        # O estado é um tensor único para esta versão do modelo
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Retorna True se o chunk contém fala."""
        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError("audio_chunk deve ser um np.ndarray")
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        ort_inputs = {
            "input": np.expand_dims(audio_chunk, 0),
            "state": self._state,
            "sr": np.array([self.sr], dtype=np.int64),
        }
        outs = self.session.run(None, ort_inputs)
        speech_prob = float(outs[0][0][0])
        self._state = outs[1] # Atualiza o estado com a saída do modelo
        return speech_prob > self.threshold
