import logging
import numpy as np
import onnxruntime
import torch
from pathlib import Path

# Construir um caminho absoluto para o modelo ONNX, relativo a este arquivo.
# Isso garante que o modelo seja encontrado independentemente do diretório de trabalho.
MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"

class VADManager:
    """Gerencia a detecção de voz usando o modelo Silero."""

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        """Inicializa o VAD Manager."""

        self.threshold = threshold
        self.sr = sampling_rate
        # Flag que indica se o VAD está pronto para uso
        self.enabled = False

        if not MODEL_PATH.exists():
            logging.error(
                "Arquivo do modelo VAD ausente em '%s'. Recurso VAD desabilitado.",
                MODEL_PATH,
            )
            self.session = None
            return

        try:
            model_path_str = str(MODEL_PATH)
            # Seleciona automaticamente o provider, priorizando CUDA se disponível
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider"]
                logging.info("CUDAExecutionProvider detectado para o VAD.")
            else:
                providers = ["CPUExecutionProvider"]
                logging.info("CUDAExecutionProvider indisponível; usando CPUExecutionProvider.")

            self.session = onnxruntime.InferenceSession(
                model_path_str,
                providers=providers,
            )
            # Se a inicialização foi bem-sucedida, habilita o VAD
            self.enabled = True
            self.threshold = threshold
            self.sr = sampling_rate
            self.reset_states()
            logging.info(
                "Modelo VAD carregado com sucesso de '%s'.", model_path_str
            )
        except Exception as exc:
            logging.error(
                "Erro ao carregar o modelo VAD de '%s': %s", MODEL_PATH, exc, exc_info=True
            )
            self.session = None

    def reset_states(self) -> None:
        """Reseta os estados internos do modelo."""
        # O estado é um tensor único para esta versão do modelo
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Retorna ``True`` se o chunk contém fala."""

        if self.session is None:
            # VAD desabilitado – assume sempre haver fala para não cortar áudio
            return True

        if not isinstance(audio_chunk, np.ndarray):
            raise TypeError("audio_chunk deve ser um np.ndarray")
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        ort_inputs = {
            "input": torch.from_numpy(audio_chunk).unsqueeze(0).numpy(),
            "state": self._state,
            "sr": np.array([self.sr], dtype=np.int64),
        }
        outs = self.session.run(None, ort_inputs)
        speech_prob = float(outs[0][0][0])
        self._state = outs[1] # Atualiza o estado com a saída do modelo
        return speech_prob > self.threshold
