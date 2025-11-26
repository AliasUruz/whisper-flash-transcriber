import json
import time
import threading
import logging
import tempfile
import os
from pathlib import Path
from hotkeys import HotkeyManager
from mouse_handler import MouseHandler
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyperclip
from pynput.keyboard import Controller, Key
from faster_whisper import WhisperModel

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CoreService:
    def __init__(self):
        logging.info("Initializing CoreService (Final Edition)...")
        self.settings = self._load_or_create_settings()
        self.model = None
        self.device_used = "cpu"
        self.state = "loading"
        
        # UI Callbacks
        self.ui_update_callback = None
        self.error_popup_callback = None
        self.hotkey_manager = None
        self.mouse_handler = MouseHandler(self)

        # Audio buffers and control
        self.audio_stream = None
        self.audio_frames = []
        
        # Temp File Control
        self.temp_file_path = None
        self.temp_file_writer = None
        
        self.sample_rate = 16000
        
        # Buffer Control
        self.current_ram_duration = 0.0 
        
        # Hardware and IO Control
        self.keyboard = None # Lazy init
        self._lock = threading.Lock()
        self.RAM_FLUSH_THRESHOLD_SECONDS = 30

    def _load_or_create_settings(self) -> dict:
        self.config_path = Path.home() / ".whisper_flash_transcriber" / "config.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            logging.info(f"Loading settings from {self.config_path}")
            with open(self.config_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logging.error("Config file corrupted, creating new.")
        
        logging.info(f"Creating default settings file at {self.config_path}")
        default_settings = {
            "hotkey": "f3",
            "mouse_hotkey": False,
            "auto_paste": True,
            "input_device_index": None,
            "model_path": "",
            "first_run": True
        }
        with open(self.config_path, "w") as f:
            json.dump(default_settings, f, indent=2)
        return default_settings

    def get_audio_devices(self):
        """Returns list of available microphones."""
        devices = []
        try:
            info = sd.query_devices()
            for i, dev in enumerate(info):
                if dev['max_input_channels'] > 0:
                    # Filter input devices only
                    devices.append({"id": i, "name": f"{dev['name']} ({dev['hostapi']})"})
        except Exception as e:
            logging.error(f"Error listing devices: {e}")
        return devices

    def _get_temp_dir(self):
        """Returns temp directory, respecting model_path."""
        model_path = self.settings.get("model_path")
        if model_path and os.path.isdir(model_path):
            # Create tmp_audio folder inside model path
            tmp_dir = os.path.join(model_path, "tmp_audio")
            os.makedirs(tmp_dir, exist_ok=True)
            return tmp_dir
        return None # Use system default tempfile

    def load_model_async(self):
        model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
        logging.info(f"Starting asynchronous model load: {model_name}")
        
        if self.ui_update_callback:
            self.ui_update_callback("transcribing", "Loading/Downloading Model...")

        try:
            logging.info("Attempting to load on GPU (CUDA)...")
            
            download_root = self.settings.get("model_path")
            if download_root and not download_root.strip(): download_root = None
            
            self.model = WhisperModel(model_name, device="cuda", compute_type="float16", download_root=download_root)
            self.device_used = "cuda"
            logging.info("Success! Model loaded on GPU (VRAM).")
        except Exception as e_gpu:
            logging.warning(f"GPU (float16) Load failed: {e_gpu}")
            logging.info("Attempting GPU (int8) fallback...")
            try:
                download_root = self.settings.get("model_path")
                if download_root and not download_root.strip(): download_root = None
                
                self.model = WhisperModel(model_name, device="cuda", compute_type="int8", download_root=download_root)
                self.device_used = "cuda"
                logging.info("Success! Model loaded on GPU (VRAM) with INT8 quantization.")
            except Exception as e_gpu_int8:
                logging.warning(f"GPU (int8) Load failed: {e_gpu_int8}")
                logging.info("Falling back to CPU (Int8)...")
                try:
                    download_root = self.settings.get("model_path")
                    if download_root and not download_root.strip(): download_root = None

                    self.model = WhisperModel(model_name, device="cpu", compute_type="int8", download_root=download_root)
                    self.device_used = "cpu"
                    logging.info("Success! Model loaded on CPU.")
                except Exception as e_cpu:
                    msg = f"Fatal: Failed to load model.\nGPU: {e_gpu}\nCPU: {e_cpu}"
                    logging.error(msg)
                    self.state = "error"
                    if self.ui_update_callback:
                        self.ui_update_callback("error", "Model Load Failed")
                    if self.error_popup_callback:
                        self.error_popup_callback("Critical Error", msg)
                    return

        if self.state != "shutdown":
            self.state = "idle"
            if self.ui_update_callback:
                device_label = "GPU" if self.device_used == "cuda" else "CPU"
                self.ui_update_callback("idle", f"Ready ({device_label})")

    def set_ui_update_callback(self, callback):
        self.ui_update_callback = callback

    def set_error_popup_callback(self, callback):
        self.error_popup_callback = callback

    def set_hotkey_manager(self, manager):
        self.hotkey_manager = manager

    def shutdown(self):
        logging.info("Core shutting down...")
        self.state = "shutdown"
        
        # Force cleanup audio stream
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception: pass
            self.audio_stream = None

        self.stop_recording()
        self.model = None
        
        if self.mouse_handler:
            self.mouse_handler.stop_listening()

    def toggle_recording(self):
        if self.state == "shutdown": return

        if not self.model:
            if self.state == "loading":
                if self.ui_update_callback:
                    self.ui_update_callback("transcribing", "Wait: Loading...") 
                    threading.Timer(1.5, lambda: self.ui_update_callback("idle", "Wait...") if self.state == "loading" else None).start()
            return

        if self.state == "recording":
            self.stop_recording()
        elif self.state == "idle":
            self.start_recording()

    def start_recording(self):
        if self.state != "idle": return

        logging.info("Starting audio recording...")
        self.state = "recording"
        
        with self._lock:
            self.audio_frames = []
            self.current_ram_duration = 0.0 
            self.temp_file_path = None
            self.temp_file_writer = None

        if self.ui_update_callback:
            self.ui_update_callback("recording", "Listening...")

        try:
            # Device selection
            device_index = self.settings.get("input_device_index")
            # If device_index is None, sounddevice uses OS default
            
            self.audio_stream = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device_index 
            )
            self.audio_stream.start()
        except Exception as e:
            msg = f"Microphone init failed: {e}"
            logging.error(msg)
            self.state = "error"
            self._reset_to_idle()
            if self.ui_update_callback:
                self.ui_update_callback("error", "Mic Error")
            if self.error_popup_callback:
                self.error_popup_callback("Audio Error", msg)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Audio status: {status}")

        with self._lock:
            if self.state != "recording": return

            if self.temp_file_writer:
                try:
                    self.temp_file_writer.write(indata)
                except Exception as e:
                    logging.error(f"Disk write failed: {e}")
                    self.state = "error"
                    if self.ui_update_callback: self.ui_update_callback("error", "Disk Error")
                return

            self.audio_frames.append(indata.copy())
            chunk_duration = len(indata) / self.sample_rate
            self.current_ram_duration += chunk_duration
            
            if self.current_ram_duration > self.RAM_FLUSH_THRESHOLD_SECONDS:
                logging.info(f"RAM full. Flushing to disk.")
                try:
                    custom_temp_dir = self._get_temp_dir()
                    tf = tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=".wav", 
                        mode="w+b",
                        dir=custom_temp_dir
                    )
                    self.temp_file_path = tf.name
                    tf.close() 

                    self.temp_file_writer = sf.SoundFile(
                        self.temp_file_path,
                        mode="w",
                        samplerate=self.sample_rate,
                        channels=1,
                    )
                    
                    ram_buffer = np.concatenate(self.audio_frames)
                    self.temp_file_writer.write(ram_buffer)
                    
                    self.audio_frames = [] 
                    self.current_ram_duration = 0.0 
                except Exception as e:
                    logging.error(f"Flush failed: {e}")
                    self.state = "error"
                    if self.ui_update_callback: self.ui_update_callback("error", "IO Error")

    def stop_recording(self):
        if self.state != "recording": return

        logging.info("Stopping recording sequence...")
        
        # Change state immediately to prevent race in audio callback
        self.state = "transcribing"

        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                logging.error(f"Stream stop error: {e}")
            self.audio_stream = None

        with self._lock:
            if self.temp_file_writer:
                try:
                    self.temp_file_writer.flush()
                    self.temp_file_writer.close()
                except Exception as e:
                    logging.error(f"Writer close error: {e}")
                self.temp_file_writer = None
        if self.ui_update_callback:
            mode = "GPU" if self.device_used == "cuda" else "CPU"
            self.ui_update_callback("transcribing", f"Processing ({mode})...")

        threading.Thread(target=self._process_audio, daemon=True).start()

    def _process_audio(self):
        audio_source = None
        processing_path = None

        with self._lock:
            if self.temp_file_path and os.path.exists(self.temp_file_path):
                processing_path = self.temp_file_path
                audio_source = processing_path
                self.temp_file_path = None 
            elif self.audio_frames:
                audio_source = np.concatenate(self.audio_frames, axis=0).flatten()
            else:
                logging.warning("Empty buffer.")
                self._reset_to_idle()
                return

        try:
            if self.state == "shutdown": return
            text = self._run_transcription(audio_source)
            if self.state == "shutdown": return
            self._handle_result(text)
        finally:
            if processing_path:
                try:
                    os.remove(processing_path)
                    logging.info(f"Deleted temp file: {processing_path}")
                except Exception as e:
                    logging.error(f"Cleanup error: {e}")
        
        if self.state not in ["error", "shutdown"]:
            self._reset_to_idle()

    def _reset_to_idle(self):
        if self.state == "shutdown": return
        self.state = "idle"
        if self.ui_update_callback:
            mode = "GPU" if self.device_used == "cuda" else "CPU"
            self.ui_update_callback("idle", f"Ready ({mode}).")

    def _run_transcription(self, audio_source) -> str:
        if not self.model: return ""
        try:
            segments, _ = self.model.transcribe(
                audio_source, beam_size=5, vad_filter=True
            )
            result = " ".join([s.text.strip() for s in segments])
            logging.info(f"Result: {result}")
            return result.strip()
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            self.state = "error"
            if self.ui_update_callback:
                self.ui_update_callback("error", "Engine Error")
            return ""

    def _handle_result(self, text: str):
        if not text: return

        try:
            pyperclip.copy(text)
        except Exception as e:
            logging.error(f"Copy failed: {e}")
            return

        if self.settings.get("auto_paste", True):
            logging.info("Auto-pasting...")
            time.sleep(0.4) 
            try:
                # Safe Lazy Init
                if self.keyboard is None:
                    self.keyboard = Controller()

                self.keyboard.release(Key.alt)
                self.keyboard.release(Key.ctrl)
                
                with self.keyboard.pressed(Key.ctrl):
                    self.keyboard.press("v")
                    self.keyboard.release("v")
            except Exception as e:
                logging.error(f"Paste failed: {e}")

    def save_settings(self, settings: dict = None):
        logging.info("Saving settings...")
        if settings is None:
            settings = self.settings

        old_hotkey = self.settings.get("hotkey")
        new_hotkey = settings.get("hotkey")
        
        self.settings = settings
        with open(self.config_path, "w") as f:
            json.dump(self.settings, f, indent=2)

        # Restart hotkey in background to avoid UI freeze
        if old_hotkey != new_hotkey and self.hotkey_manager:
            threading.Thread(target=self.hotkey_manager.restart_listening, daemon=True).start()

        # Update Mouse Handler
        if self.mouse_handler:
            if settings.get("mouse_hotkey", False):
                self.mouse_handler.start_listening()
            else:
                self.mouse_handler.stop_listening()
