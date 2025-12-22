import queue
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import logging
from collections import deque
import tempfile
from typing import Optional, List, Tuple

class AudioEngine:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.queue = queue.SimpleQueue()
        self.is_recording = False
        self.producer_stream = None
        self.consumer_thread = None
        
        # Buffers - Deque is faster for appends than list
        self.ram_buffer = deque()
        
        # Temp File Control
        self._temp_file_path: Optional[str] = None
        self._temp_file_writer: Optional[sf.SoundFile] = None
        self._file_lock = threading.Lock()
        
        # Thresholds
        self.RAM_FLUSH_THRESHOLD_SAMPLES = sample_rate * 30 # 30 seconds
        self._samples_accumulated = 0
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Producer: Pushes audio data to queue immediately."""
        if status:
            logging.warning(f"Audio status: {status}")
        try:
            # We must copy indata because it is reused by sounddevice
            self.queue.put(indata.copy())
        except Exception as e:
            logging.error(f"Producer error: {e}")

    def _consumer_worker(self):
        """Consumer: Processes queue items and manages RAM/Disk storage."""
        
        while True:
            try:
                item = self.queue.get()
                if item is None: # Sentinel to stop
                    break
                    
                chunk_len = len(item)
                
                with self._file_lock:
                    if self._temp_file_writer:
                        # Direct to disk mode
                        self._temp_file_writer.write(item)
                    else:
                        # RAM mode
                        self.ram_buffer.append(item)
                        self._samples_accumulated += chunk_len
                        
                        if self._samples_accumulated > self.RAM_FLUSH_THRESHOLD_SAMPLES:
                            self._flush_to_disk()
                            
            except Exception as e:
                logging.error(f"Consumer loop error: {e}")

    def _flush_to_disk(self):
        """Moves RAM buffer to a temp file and switches mode."""
        try:
            logging.info("AudioEngine: Flushing RAM to disk (Buffer Full)...")
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode="w+b")
            self._temp_file_path = tf.name
            tf.close()
            
            self._temp_file_writer = sf.SoundFile(
                self._temp_file_path, mode="w", samplerate=self.sample_rate, channels=1
            )
            
            # Write existing buffer
            # Convert deque to single array only once
            if self.ram_buffer:
                full_buffer = np.concatenate(list(self.ram_buffer))
                self._temp_file_writer.write(full_buffer)
            
            # Clear RAM
            self.ram_buffer.clear()
            self._samples_accumulated = 0
            
        except Exception as e:
            logging.error(f"Flush failed: {e}")

    def start_capture(self, device_index: Optional[int] = None):
        """Initializes and starts both producer stream and consumer thread."""
        with self._file_lock: 
            self.ram_buffer.clear()
            self._samples_accumulated = 0
            self._temp_file_path = None
            self._temp_file_writer = None
            
        # Drain queue safe
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except queue.Empty: break
            
        self.is_recording = True
        
        # Start Consumer Thread
        self.consumer_thread = threading.Thread(target=self._consumer_worker, daemon=True)
        self.consumer_thread.start()
        
        # Start Producer Stream
        try:
            self.producer_stream = sd.InputStream(
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=device_index
            )
            self.producer_stream.start()
            logging.info(f"AudioEngine started on device {device_index}")
        except Exception as e:
            logging.error(f"Failed to start audio stream: {e}")
            self.is_recording = False
            self.queue.put(None)
            raise e

    def stop_capture(self) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Stops capture and returns (file_path, ram_data). Only one will be set."""
        if not self.is_recording:
            return None, None
            
        logging.info("AudioEngine stopping...")
        self.is_recording = False
        
        # 1. Stop Stream
        if self.producer_stream:
            try:
                self.producer_stream.stop()
                self.producer_stream.close()
            except Exception: pass
            self.producer_stream = None
            
        # 2. Signal Consumer
        self.queue.put(None)
        
        # 3. Wait Consumer
        if self.consumer_thread:
            self.consumer_thread.join(timeout=2.0)
            self.consumer_thread = None
            
        # 4. Finalize
        with self._file_lock:
            if self._temp_file_writer:
                try:
                    self._temp_file_writer.flush()
                    self._temp_file_writer.close()
                except Exception: pass
                
                path = self._temp_file_path
                self._temp_file_writer = None
                return path, None
            else:
                if not self.ram_buffer: return None, None
                
                # Retrieve list of chunks (each is usually (N, 1) or (N,))
                chunks = list(self.ram_buffer)
                
                # Concatenate
                combined = np.concatenate(chunks)
                
                # Flatten -> Ensures 1D array (N,)
                if combined.ndim > 1:
                    combined = combined.flatten()
                    
                return None, combined
