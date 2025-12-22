import time
import threading
import logging
import os
import pyperclip
from pynput.keyboard import Controller, Key

from audio_engine import AudioEngine
from ai_corrector import AICorrector
from native_mouse import NativeMouseHook
from settings import SettingsManager, AppState, AppSettings
from services.transcription_service import TranscriptionService
from services.sound_service import SoundService

class CoreService:
    def __init__(self):
        logging.info("Initializing CoreService (Orchestrator Edition)...")
        self.settings_manager = SettingsManager()
        self.settings: AppSettings = self.settings_manager.get_settings()
        
        # State Management
        self.state = AppState.LOADING
        
        # Services
        self.sound_service = SoundService()
        self.transcription_service = TranscriptionService(model_path=self.settings.model_path)
        self.audio_engine = AudioEngine()
        self.ai_corrector = AICorrector()
        
        # Input Hooks
        self.mouse_hook = NativeMouseHook(self)
        self.hotkey_manager = None # Injected later
        self.keyboard = None # Lazy init

        # Callbacks
        self.ui_update_callback = None
        self.error_popup_callback = None

    def load_model_async(self):
        """Delegates model loading to service."""
        if self.ui_update_callback:
            self.ui_update_callback(AppState.TRANSCRIBING.value, "Loading Model...")

        def on_success(strategy_id, strategy_desc):
            logging.info(f"Core: Model loaded via {strategy_id}")
            self.settings_manager.update(last_device_strategy=strategy_id)
            self._set_state(AppState.IDLE, f"Ready ({strategy_desc})")

        def on_fail(error_msg):
            logging.error(f"Core: Model load failed - {error_msg}")
            self._set_state(AppState.ERROR, "Load Failed")
            if self.error_popup_callback:
                self.error_popup_callback("Model Error", error_msg)

        self.transcription_service.load_model_async(
            callback_success=on_success,
            callback_fail=on_fail,
            preferred_strategy=self.settings.last_device_strategy
        )

    def _set_state(self, new_state: AppState, status_text: str = None):
        """Centralized state mutation."""
        if self.state == AppState.SHUTDOWN: return
        
        self.state = new_state
        if self.ui_update_callback:
            # UI expects string values currently
            txt = status_text if status_text else new_state.value.title()
            self.ui_update_callback(new_state.value, txt)

    def toggle_recording(self):
        if self.state == AppState.SHUTDOWN: return

        if self.transcription_service.is_loading:
            self.ui_update_callback(AppState.TRANSCRIBING.value, "Wait: Loading...")
            return

        if self.state == AppState.RECORDING:
            self.stop_recording()
        elif self.state == AppState.IDLE:
            self.start_recording()

    def start_recording(self):
        if self.state != AppState.IDLE: return

        logging.info("Core: Starting recording...")
        self._set_state(AppState.RECORDING, "Listening...")
        self.sound_service.play_tone(
            self.settings.sound_freq_start, 
            volume=self.settings.sound_volume, 
            enabled=self.settings.sound_enabled
        )

        try:
            self.audio_engine.start_capture(self.settings.input_device_index)
        except Exception as e:
            self._set_state(AppState.ERROR, "Mic Error")
            logging.error(f"Mic fail: {e}")
            if self.error_popup_callback:
                self.error_popup_callback("Audio Error", str(e))
            self._reset_to_idle_delayed()

    def stop_recording(self):
        if self.state != AppState.RECORDING: return

        logging.info("Core: Stopping recording...")
        self._set_state(AppState.TRANSCRIBING, "Processing...")
        
        # Stop capture
        try:
            file_path, ram_data = self.audio_engine.stop_capture()
        except Exception as e:
            logging.error(f"Stop capture error: {e}")
            self._set_state(AppState.ERROR, "Capture Error")
            self._reset_to_idle_delayed()
            return

        self.sound_service.play_tone(
            self.settings.sound_freq_stop, 
            volume=self.settings.sound_volume, 
            enabled=self.settings.sound_enabled
        )

        # Process in thread
        threading.Thread(
            target=self._process_pipeline, 
            args=(ram_data, file_path), 
            daemon=True
        ).start()

    def _process_pipeline(self, ram_data, file_path):
        try:
            if self.state == AppState.SHUTDOWN: return

            # 1. Transcribe
            raw_text = self.transcription_service.transcribe(ram_data if ram_data is not None else file_path)
            
            # 2. AI Correction (Optional)
            final_text = raw_text
            if final_text and self.settings.gemini_enabled and self.settings.gemini_api_key:
                if self.ui_update_callback:
                    self.ui_update_callback(AppState.TRANSCRIBING.value, "AI Correcting...")
                
                corrected = self.ai_corrector.correct_text(
                    final_text,
                    self.settings.gemini_api_key,
                    self.settings.gemini_prompt,
                    self.settings.gemini_model
                )
                if corrected:
                    final_text = corrected

            # 3. Post-Process & Output
            if final_text:
                if self.settings.append_space:
                    final_text += " "
                
                self._handle_output(final_text)

        except Exception as e:
            logging.error(f"Pipeline error: {e}")
            self._set_state(AppState.ERROR, "Error")
        finally:
            # Cleanup temp file if exists
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception: pass
            
            self._reset_to_idle()

    def _handle_output(self, text: str):
        try:
            pyperclip.copy(text)
            if self.settings.auto_paste:
                self._perform_paste()
        except Exception as e:
            logging.error(f"Output error: {e}")

    def _perform_paste(self):
        logging.info("Auto-pasting...")
        time.sleep(0.3)
        try:
            if not self.keyboard: self.keyboard = Controller()
            with self.keyboard.pressed(Key.ctrl):
                self.keyboard.press('v')
                self.keyboard.release('v')
        except Exception as e:
            logging.error(f"Paste error: {e}")

    def _reset_to_idle(self):
        if self.state == AppState.SHUTDOWN: return
        self._set_state(AppState.IDLE, "Ready")

    def _reset_to_idle_delayed(self):
        threading.Timer(2.0, self._reset_to_idle).start()

    # --- External Control ---
    def set_ui_update_callback(self, cb): self.ui_update_callback = cb
    def set_error_popup_callback(self, cb): self.error_popup_callback = cb
    def set_hotkey_manager(self, mgr): self.hotkey_manager = mgr

    def get_audio_devices(self):
        # Pass-through or move logic here later if needed. 
        # Keeping minimal logic inside core directly for now or use sounddevice
        import sounddevice as sd
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev['max_input_channels'] > 0:
                    devices.append({"id": i, "name": f"{dev['name']} ({dev['hostapi']})"})
        except Exception: pass
        return devices

    def save_settings(self, updates: dict):
        """Update settings and trigger side effects."""
        old_hotkey = self.settings.hotkey
        
        # Commit to manager
        self.settings_manager.update(**updates)
        
        # Side Effects
        if "hotkey" in updates and updates["hotkey"] != old_hotkey:
            if self.hotkey_manager:
                threading.Thread(target=self.hotkey_manager.restart_listening, daemon=True).start()

        if "mouse_hotkey" in updates:
            if updates["mouse_hotkey"]: self.mouse_hook.start()
            else: self.mouse_hook.stop()

    def shutdown(self):
        self.state = AppState.SHUTDOWN
        if self.audio_engine: self.audio_engine.stop_capture()
        if self.mouse_hook: self.mouse_hook.stop()
        if self.hotkey_manager: self.hotkey_manager.stop_listening()
        self.transcription_service.unload()
