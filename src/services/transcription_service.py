import logging
import threading
from typing import Optional
from faster_whisper import WhisperModel

class TranscriptionService:
    def __init__(self, model_path: str = ""):
        self.model: Optional[WhisperModel] = None
        self.current_device = "cpu"
        self.model_path = model_path if model_path.strip() else None
        self.is_loading = False
        self._lock = threading.Lock()

    def load_model_async(self, callback_success, callback_fail, preferred_strategy: Optional[str] = None):
        """Loads the model in a background thread."""
        if self.is_loading: 
            logging.warning("Model load already in progress.")
            return

        self.is_loading = True
        threading.Thread(
            target=self._load_worker, 
            args=(callback_success, callback_fail, preferred_strategy), 
            daemon=True
        ).start()

    def _load_worker(self, on_success, on_fail, preferred_strategy):
        strategies = [
            {"device": "cuda", "compute_type": "float16", "desc": "GPU (Float16)", "id": "cuda:float16"},
            {"device": "cuda", "compute_type": "int8", "desc": "GPU (Int8)", "id": "cuda:int8"},
            {"device": "cpu", "compute_type": "int8", "desc": "CPU (Int8)", "id": "cpu:int8"}
        ]

        # Optimize order
        if preferred_strategy:
            strategies.sort(key=lambda x: x["id"] == preferred_strategy, reverse=True)

        model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
        
        for strategy in strategies:
            try:
                logging.info(f"Attempting load: {strategy['desc']}...")
                model = WhisperModel(
                    model_name,
                    device=strategy["device"],
                    compute_type=strategy["compute_type"],
                    download_root=self.model_path
                )
                
                with self._lock:
                    self.model = model
                    self.current_device = strategy["device"]
                
                logging.info(f"Success! Model loaded on {strategy['desc']}.")
                self.is_loading = False
                if on_success: on_success(strategy["id"], strategy["desc"])
                return

            except Exception as e:
                logging.warning(f"Load failed for {strategy['desc']}: {e}")
                continue

        logging.error("Fatal: All model loading strategies failed.")
        self.is_loading = False
        if on_fail: on_fail("Failed to load model on any device.")

    def transcribe(self, audio_data) -> str:
        """Runs transcription on the provided audio data (numpy array or path)."""
        if not self.model:
            logging.error("Transcribe called but model is not loaded.")
            return ""

        try:
            # VAD filter helps reduce hallucination on silence
            segments, _ = self.model.transcribe(
                audio_data, 
                beam_size=5, 
                vad_filter=True
            )
            # Generator to string
            result = " ".join([s.text.strip() for s in segments])
            return result.strip()
        except Exception as e:
            logging.error(f"Transcription engine error: {e}")
            return ""

    def unload(self):
        with self._lock:
            self.model = None
            import gc
            gc.collect()
