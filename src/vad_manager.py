import logging
import sys
import numpy as np
try:
    import onnxruntime
except ImportError:  # pragma: no cover - handled in tests
    logging.warning("onnxruntime n\u00e3o encontrado. VAD desativado.")
    onnxruntime = None
import torch
from pathlib import Path

# Detecta se o aplicativo está rodando em um diretório temporário (PyInstaller)
if hasattr(sys, "_MEIPASS"):
    base_dir = Path(sys._MEIPASS)
else:
    base_dir = Path(__file__).resolve().parent

# Construir o caminho absoluto para o modelo ONNX
MODEL_PATH = base_dir / "models" / "silero_vad.onnx"
logging.info("Caminho do modelo VAD definido como '%s'", MODEL_PATH)

class VADManager:
    """Gerencia a detecção de voz usando o modelo Silero."""

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        """Inicializa o VAD Manager."""

        self.threshold = threshold
        self.sr = sampling_rate
        # Flag que indica se o VAD está pronto para uso
        self.enabled = False

        if onnxruntime is None:
            logging.warning("onnxruntime indispon\u00edvel. VAD desativado.")
            self.session = None
            self.enabled = False
            return

        if not MODEL_PATH.exists():
            logging.error(
                "Arquivo do modelo VAD ausente em '%s'. Recurso VAD desabilitado.",
                MODEL_PATH,
            )
            self.session = None
            self.enabled = False
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
            # Entradas inválidas não devem gerar exceção, apenas retornar False
            return False
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)

        ort_inputs = {
            "input": audio_chunk.reshape(1, -1),
            "state": self._state,
            "sr": np.array([self.sr], dtype=np.int64),
        }
        outs = self.session.run(None, ort_inputs)
        speech_prob = float(outs[0][0][0])
        self._state = outs[1] # Atualiza o estado com a saída do modelo
        return speech_prob > self.threshold
