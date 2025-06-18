import logging
import threading
import time
from threading import RLock
import atexit
try:
    import pyautogui  # Ainda necessário para _do_paste
except ImportError as exc:
    raise SystemExit(
        "Erro: a biblioteca 'pyautogui' não está instalada. "
        "Execute 'pip install -r requirements.txt' antes de executar o aplicativo."
    ) from exc
import pyperclip # Ainda necessário para _handle_transcription_result
import numpy as np # Adicionado para np.ndarray no callback
from tkinter import messagebox # Adicionado para messagebox no _on_model_load_failed

# Importar os novos módulos
from .config_manager import (
    ConfigManager,
    REREGISTER_INTERVAL_SECONDS,
    HOTKEY_HEALTH_CHECK_INTERVAL,
    DISPLAY_TRANSCRIPTS_KEY,
)
from .audio_handler import AudioHandler, AUDIO_SAMPLE_RATE # AUDIO_SAMPLE_RATE ainda é usado em _handle_transcription_result
from .transcription_handler import TranscriptionHandler
from .keyboard_hotkey_manager import KeyboardHotkeyManager # Assumindo que está na raiz
from .gemini_api import GeminiAPI # Adicionado para correção de texto

# Estados da aplicação (movidos de global)
STATE_IDLE = "IDLE"
STATE_LOADING_MODEL = "LOADING_MODEL"
STATE_RECORDING = "RECORDING"
STATE_TRANSCRIBING = "TRANSCRIBING"
STATE_ERROR_MODEL = "ERROR_MODEL"
STATE_ERROR_AUDIO = "ERROR_AUDIO"
STATE_ERROR_TRANSCRIPTION = "ERROR_TRANSCRIPTION"
STATE_ERROR_SETTINGS = "ERROR_SETTINGS"

class AppCore:
    def __init__(self, main_tk_root):
        self.main_tk_root = main_tk_root # Referência para a raiz Tkinter

        # --- Locks ---
        self.hotkey_lock = RLock()
        self.recording_lock = RLock()
        self.transcription_lock = RLock()
        self.state_lock = RLock()
        self.keyboard_lock = RLock()

        # --- Callbacks para UI (definidos externamente pelo UIManager) ---
        self.state_update_callback = None
        self.on_segment_transcribed = None # Callback para UI ao vivo

        # --- Módulos ---
        self.config_manager = ConfigManager()
        self.audio_handler = AudioHandler(
            self.config_manager,
            on_audio_segment_ready_callback=self._on_audio_segment_ready,
            on_recording_state_change_callback=self._set_state
        )
        self.gemini_api = GeminiAPI(self.config_manager) # Instancia o GeminiAPI
        self.transcription_handler = TranscriptionHandler(
            config_manager=self.config_manager,
            gemini_api_client=self.gemini_api,  # Injeta a instância da API
            on_model_ready_callback=self._on_model_loaded,
            on_model_error_callback=self._on_model_load_failed,
            on_transcription_result_callback=self._handle_transcription_result,
            on_agent_result_callback=self._handle_agent_result_final, # Usa o novo callback
            on_segment_transcribed_callback=self._on_segment_transcribed_for_ui
        )
        self.ui_manager = None # Será setado externamente pelo main.py

        # --- Estado da Aplicação ---
        self.current_state = STATE_LOADING_MODEL
        self.shutting_down = False
        self.full_transcription = "" # Acumula transcrição completa
        self.agent_mode_active = False # Adicionado para controle do modo agente
        self.key_detection_active = False # Flag para controle da detecção de tecla

        # --- Hotkey Manager ---
        self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
        self.ahk_running = False
        self.last_key_press_time = 0.0
        self.reregister_timer_thread = None
        self.stop_reregister_event = threading.Event()
        self.health_check_thread = None
        self.stop_health_check_event = threading.Event()
        self.key_detection_callback = None # Callback para atualizar a UI com a tecla detectada

        # Carregar configurações iniciais e iniciar carregamento do modelo
        self._apply_initial_config_to_core_attributes()
        self.transcription_handler.start_model_loading()
        self._cleanup_old_audio_files_on_startup()
        atexit.register(self.shutdown)

    def _apply_initial_config_to_core_attributes(self):
        # Mover a atribuição de self.record_key, self.record_mode, etc.
        # para cá, usando self.config_manager.get()
        self.record_key = self.config_manager.get("record_key")
        self.record_mode = self.config_manager.get("record_mode")
        self.auto_paste = self.config_manager.get("auto_paste")
        self.agent_key = self.config_manager.get("agent_key")
        self.hotkey_stability_service_enabled = self.config_manager.get("hotkey_stability_service_enabled") # Nova configuração unificada
        self.keyboard_library = self.config_manager.get("keyboard_library")
        self.min_record_duration = self.config_manager.get("min_record_duration")
        self.display_transcripts_in_terminal = self.config_manager.get(DISPLAY_TRANSCRIPTS_KEY)
        # ... e outras configurações que AppCore precisa diretamente

    # --- Callbacks de Módulos ---
    def set_state_update_callback(self, callback):
        self.state_update_callback = callback

    def set_segment_callback(self, callback):
        self.on_segment_transcribed = callback

    def set_key_detection_callback(self, callback):
        """Define o callback para atualizar a UI com a tecla detectada."""
        self.key_detection_callback = callback

    def _on_audio_segment_ready(self, audio_segment: np.ndarray):
        """Callback do AudioHandler quando um segmento de áudio está pronto para transcrição."""
        duration_seconds = len(audio_segment) / AUDIO_SAMPLE_RATE
        min_duration = self.config_manager.get('min_transcription_duration')
        
        if duration_seconds < min_duration:
            logging.info(f"Segmento de áudio ({duration_seconds:.2f}s) é mais curto que o mínimo configurado ({min_duration}s). Ignorando.")
            self._set_state(STATE_IDLE) # Volta para o estado IDLE
            return # Interrompe o processamento

        # Captura o estado do modo agente ANTES de qualquer coisa.
        is_agent_mode = self.agent_mode_active
        
        # Reseta o estado imediatamente após capturá-lo para a próxima gravação.
        if is_agent_mode:
            self.agent_mode_active = False

        logging.info(f"AppCore: Segmento de áudio pronto ({duration_seconds:.2f}s). Enviando para TranscriptionHandler (Modo Agente: {is_agent_mode}).")
        
        # Passa o estado capturado para o handler de transcrição.
        self.transcription_handler.transcribe_audio_segment(audio_segment, agent_mode=is_agent_mode)

    def _on_model_loaded(self):
        """Callback do TranscriptionHandler quando o modelo é carregado com sucesso."""
        logging.info("AppCore: Modelo carregado com sucesso.")
        self._set_state(STATE_IDLE)
        self._start_autohotkey()
        
        # Iniciar serviços de estabilidade de hotkey se habilitados
        if self.hotkey_stability_service_enabled:
            # Iniciar thread de re-registro periódico
            if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                self.stop_reregister_event.clear()
                self.reregister_timer_thread = threading.Thread(
                    target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister"
                )
                self.reregister_timer_thread.start()
                logging.info("Periodic hotkey re-registration thread started.")
            
            # Iniciar thread de verificação de saúde
            if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                self.stop_health_check_event.clear()
                self.health_check_thread = threading.Thread(
                    target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread"
                )
                self.health_check_thread.start()
                logging.info("Hotkey health monitoring thread launched.")
        else:
            logging.info("Hotkey stability services are disabled by configuration.")

    def _on_model_load_failed(self, error_msg):
        """Callback do TranscriptionHandler quando o modelo falha ao carregar."""
        logging.error(f"AppCore: Falha ao carregar o modelo: {error_msg}")
        self._set_state(STATE_ERROR_MODEL)
        self._log_status(f"Erro: Falha ao carregar o modelo. {error_msg}", error=True)
        # Exibir messagebox via UI Manager se disponível
        if self.ui_manager:
            self.main_tk_root.after(0, lambda: messagebox.showerror("Erro de Carregamento do Modelo", f"Falha ao carregar o modelo Whisper:\n{error_msg}\n\nPor favor, verifique sua conexão com a internet, o nome do modelo nas configurações ou a memória da sua GPU."))

    def _on_segment_transcribed_for_ui(self, text):
        """Callback para enviar texto de segmento para a UI ao vivo."""
        if self.on_segment_transcribed:
            self.on_segment_transcribed(text)
        self.full_transcription += text + " " # Acumula a transcrição completa

    def _handle_transcription_result(self, corrected_text, raw_text):
        """Lida com o texto final de transcrição, priorizando a versão corrigida."""
        logging.info("AppCore: Lidando com o resultado final da transcrição.")
        # O texto corrigido tem prioridade; se vazio, usa o acumulado durante a gravação
        text_to_display = corrected_text
        final_text = text_to_display.strip() if text_to_display else self.full_transcription.strip()

        if self.display_transcripts_in_terminal:
            print("\n=== COMPLETE TRANSCRIPTION ===\n" + final_text + "\n==============================\n")

        if pyperclip:
            try:
                pyperclip.copy(final_text)
                logging.info("Transcrição copiada para o clipboard.")
            except Exception as e:
                logging.error(f"Erro ao copiar para o clipboard: {e}")
        
        if self.auto_paste:
            self._do_paste()
        else:
            self._log_status("Transcription complete. Auto-paste disabled.")
        
        self._set_state(STATE_IDLE)
        if self.ui_manager:
            self.ui_manager.close_live_transcription_window()
        logging.info(f"Texto final corrigido para copiar/colar: {final_text}")
        self.full_transcription = ""  # Reset para a próxima gravação

    def _handle_agent_result_final(self, agent_response_text: str):
        """
        Lida com o resultado final do modo agente (copia, cola e reseta o estado).
        Esta função é chamada pelo TranscriptionHandler após a API Gemini ser consultada.
        """
        try:
            if not agent_response_text:
                logging.warning("Comando do agente retornou uma resposta vazia.")
                self._log_status("Comando do agente sem resposta.", error=True)
                return

            if pyperclip:
                pyperclip.copy(agent_response_text)
                logging.info("Resposta do agente copiada para a área de transferência.")

            if self.config_manager.get("agent_auto_paste", True): # Usa agent_auto_paste
                self._do_paste()
                self._log_status("Comando do agente executado e colado.")
            else:
                self._log_status("Comando do agente executado (colagem automática desativada).")

        except Exception as e:
            logging.error(f"Erro ao manusear o resultado do agente: {e}", exc_info=True)
            self._log_status(f"Erro ao manusear o resultado do agente: {e}", error=True)
        finally:
            self._set_state(STATE_IDLE)
            if self.ui_manager:
                self.ui_manager.close_live_transcription_window()

    def _do_paste(self):
        # Lógica movida de WhisperCore._do_paste
        try:
            pyautogui.hotkey('ctrl', 'v')
            logging.info("Texto colado.")
            self._log_status("Texto colado.")
        except Exception as e:
            logging.error(f"Erro ao colar: {e}")
            self._log_status("Erro ao colar.", error=True)

    def start_key_detection_thread(self):
        """Inicia uma thread para detectar uma única tecla e atualizar a UI."""
        if self.key_detection_active:
            logging.info("Detecção de tecla já está ativa.")
            return

        self.key_detection_active = True
        logging.info("Iniciando detecção de tecla...")
        
        def detect_key_task():
            try:
                # Temporariamente desativar hotkeys existentes para evitar conflitos
                self._cleanup_hotkeys()
                time.sleep(0.1) # Pequena pausa para garantir que os hooks sejam liberados

                detected_key = self.ahk_manager.detect_single_key()
                if detected_key:
                    logging.info(f"Tecla detectada: {detected_key}")
                    if self.key_detection_callback:
                        self.main_tk_root.after(0, lambda: self.key_detection_callback(detected_key.upper()))
                else:
                    logging.warning("Nenhuma tecla detectada ou detecção cancelada.")
                    if self.key_detection_callback:
                        self.main_tk_root.after(0, lambda: self.key_detection_callback("N/A")) # Ou algum valor padrão
            except Exception as e:
                logging.error(f"Erro durante a detecção de tecla: {e}", exc_info=True)
                if self.key_detection_callback:
                    self.main_tk_root.after(0, lambda: self.key_detection_callback("ERRO"))
            finally:
                self.key_detection_active = False
                # Re-registrar hotkeys após a detecção
                self.register_hotkeys()
                logging.info("Detecção de tecla finalizada. Hotkeys re-registradas.")

        threading.Thread(target=detect_key_task, daemon=True, name="KeyDetectionThread").start()

    # --- Gerenciamento de Estado e Logs ---
    def _set_state(self, new_state):
        with self.state_lock:
            if self.current_state == new_state:
                logging.debug(f"State already {new_state}, not changing.")
                return
            self.current_state = new_state
            logging.info(f"State changed to: {new_state}")
            callback_to_call = self.state_update_callback
            current_state_for_callback = new_state
        if callback_to_call:
            try:
                callback_to_call(current_state_for_callback)
            except Exception as e:
                logging.error(f"Error calling state update callback for state {current_state_for_callback}: {e}")

    def _log_status(self, text, error=False):
        if error: logging.error(text)
        else: logging.info(text)

    # --- Hotkey Logic (movida de WhisperCore) ---
    def _start_autohotkey(self):
        with self.hotkey_lock:
            if self.ahk_running: return True
            self.ahk_manager.update_config(
                record_key=self.record_key, agent_key=self.agent_key, record_mode=self.record_mode
            )
            self.ahk_manager.set_callbacks(
                toggle=self.toggle_recording, start=self.start_recording,
                stop=self.stop_recording_if_needed, agent=self.start_agent_command
            )
            success = self.ahk_manager.start()
            if success:
                self.ahk_running = True
                self._log_status(f"Hotkey registrada: {self.record_key.upper()} (modo: {self.record_mode})")
            else:
                self._set_state(STATE_ERROR_SETTINGS)
                self._log_status("Erro: Falha ao iniciar KeyboardHotkeyManager.", error=True)
            return success

    def register_hotkeys(self):
        self._cleanup_hotkeys()
        time.sleep(0.2)
        if not self.record_key:
            self._set_state(STATE_ERROR_SETTINGS)
            self._log_status("Error: No record key set!", error=True)
            return False
        success = self._start_autohotkey()
        if success:
            self._log_status(f"Global hotkey registered: {self.record_key.upper()} (mode: {self.record_mode})")
            if self.current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
                self._set_state(STATE_IDLE)
        else:
            self._set_state(STATE_ERROR_SETTINGS)
            self._log_status("Error: Hotkey registration failed.", error=True)
        return success

    def _cleanup_hotkeys(self):
        with self.keyboard_lock:
            try:
                if self.ahk_running:
                    if hasattr(self.ahk_manager, 'hotkey_handlers'):
                        self.ahk_manager.hotkey_handlers.clear()
                    self.ahk_manager.stop()
                    self.ahk_running = False
                    time.sleep(0.2)
            except Exception as e:
                logging.error(f"Error stopping KeyboardHotkeyManager: {e}")

    def _reload_keyboard_and_suppress(self):
        with self.keyboard_lock:
            max_attempts = 3; attempt = 0; last_error = None
            self._cleanup_hotkeys(); time.sleep(0.3)
            while attempt < max_attempts:
                attempt += 1
                try:
                    if self.ahk_running: self.ahk_manager.stop(); self.ahk_running = False; time.sleep(0.2)
                    self.ahk_manager = KeyboardHotkeyManager(config_file="hotkey_config.json")
                    logging.info("Recarregamento do KeyboardHotkeyManager concluído com sucesso.")
                    break
                except Exception as e: last_error = e; logging.error(f"Erro na tentativa {attempt} de recarregamento: {e}"); time.sleep(1)
            if attempt >= max_attempts and last_error is not None:
                logging.error(f"Falha após {max_attempts} tentativas de recarregamento. Último erro: {last_error}")
                return False
            return self.register_hotkeys()

    def _periodic_reregister_task(self):
        while not self.stop_reregister_event.wait(REREGISTER_INTERVAL_SECONDS):
            with self.state_lock: current_state = self.current_state
            if current_state == STATE_IDLE: # Re-registrar hotkeys apenas quando ocioso
                logging.info(f"Periodic check: State is {current_state}. Attempting hotkey re-registration.")
                try:
                    success = self._reload_keyboard_and_suppress()
                    if success:
                        logging.info("Periodic hotkey re-registration attempt finished successfully.")
                        with self.state_lock:
                            if self.current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
                                self._set_state(STATE_IDLE)
                    else:
                        logging.warning("Periodic hotkey re-registration attempt failed.")
                        self._set_state(STATE_ERROR_SETTINGS)
                except Exception as e:
                    logging.error(f"Error during periodic hotkey re-registration: {e}", exc_info=True)
                    self._set_state(STATE_ERROR_SETTINGS)
            else:
                logging.debug(f"Periodic check: State is {current_state}. Skipping hotkey re-registration.")
        logging.info("Periodic hotkey re-registration thread stopped.")

    def force_reregister_hotkeys(self):
        with self.state_lock: current_state = self.current_state
        if current_state not in [STATE_RECORDING, STATE_LOADING_MODEL]:
            logging.info(f"Manual trigger: State is {current_state}. Attempting hotkey re-registration.")
            with self.hotkey_lock:
                try:
                    if self.ahk_running: self.ahk_manager.stop(); self.ahk_running = False; time.sleep(0.5)
                    self.ahk_manager.update_config(record_key=self.record_key, agent_key=self.agent_key, record_mode=self.record_mode)
                    self.ahk_manager.set_callbacks(toggle=self.toggle_recording, start=self.start_recording, stop=self.stop_recording_if_needed, agent=self.start_agent_command)
                    success = self.ahk_manager.start()
                    if success:
                        self.ahk_running = True
                        if current_state.startswith("ERROR"): self._set_state(STATE_IDLE)
                        self._log_status("Recarregamento do KeyboardHotkeyManager concluído.", error=False)
                        return True
                    else:
                        self._log_status("Falha ao recarregar KeyboardHotkeyManager.", error=True)
                        self._set_state(STATE_ERROR_SETTINGS)
                        return False
                except Exception as e:
                    self.ahk_running = False
                    logging.error(f"Exception during manual KeyboardHotkeyManager re-registration: {e}", exc_info=True)
                    self._log_status(f"Erro ao recarregar KeyboardHotkeyManager: {e}", error=True)
                    self._set_state(STATE_ERROR_SETTINGS)
                    return False
        else:
            logging.warning(f"Manual trigger: Cannot re-register hotkeys. Current state is {current_state}.")
            self._log_status(f"Não é possível recarregar agora (Estado: {current_state}).", error=True)
            return False

    def _hotkey_health_check_task(self):
        while not self.stop_health_check_event.wait(HOTKEY_HEALTH_CHECK_INTERVAL):
            with self.state_lock: current_state = self.current_state
            if current_state in [STATE_IDLE, STATE_TRANSCRIBING]:
                if not self.ahk_running:
                    logging.warning("KeyboardHotkeyManager não está em execução. Tentando reiniciar.")
                    self.force_reregister_hotkeys()
                    self._log_status("Tentativa de reiniciar KeyboardHotkeyManager.", error=False)
                else:
                    logging.debug("KeyboardHotkeyManager está funcionando corretamente.")
            # Se o serviço de estabilidade estiver desativado, esta thread não deveria estar rodando.
            # Se estiver rodando, significa que o estado mudou ou houve um erro.
            # Não é necessário logar "Pulando verificação" se o serviço está desativado.
        logging.info("Hotkey health monitoring thread stopped.")

    # --- Recording Control (delegando para AudioHandler) ---
    def start_recording(self):
        with self.recording_lock:
            if self.audio_handler.is_recording: return
            with self.transcription_lock:
                if self.transcription_handler.transcription_in_progress:
                    self._log_status("Cannot record: Transcription running.", error=True); return
            with self.state_lock:
                if self.transcription_handler.pipe is None or self.current_state == STATE_LOADING_MODEL:
                    self._log_status("Cannot record: Model not loaded.", error=True); return
                if self.current_state.startswith("ERROR"):
                    self._log_status(f"Cannot record: App in error state ({self.current_state}).", error=True); return
        
        # if self.ui_manager:
        #     self.ui_manager.show_live_transcription_window()
        self.audio_handler.start_recording()
        self.full_transcription = "" # Reset full transcription on new recording

    def stop_recording(self, agent_mode=False):
        with self.recording_lock:
            if not self.audio_handler.is_recording: return
        
        self.audio_handler.stop_recording()
        
        # A janela de UI ao vivo será fechada pelo _handle_transcription_result

    def stop_recording_if_needed(self):
        with self.recording_lock:
            if not self.audio_handler.is_recording: return
        self.stop_recording()

    def toggle_recording(self):
        with self.recording_lock: rec = self.audio_handler.is_recording
        with self.transcription_lock: transcribing = self.transcription_handler.transcription_in_progress
        if rec: self.stop_recording()
        elif transcribing: self._log_status("Cannot start recording, transcription in progress.", error=True)
        else: self.start_recording()

    def start_agent_command(self):
        with self.recording_lock:
            if self.audio_handler.is_recording and self.agent_mode_active:
                self.stop_recording(agent_mode=True); self.agent_mode_active = False; return
            elif self.audio_handler.is_recording: return
            with self.transcription_lock:
                if self.transcription_handler.transcription_in_progress: return
            with self.state_lock:
                if self.transcription_handler.pipe is None or self.current_state == STATE_LOADING_MODEL:
                    self._log_status("Model not loaded.", error=True); return
                if self.current_state.startswith("ERROR"):
                    self._log_status(f"Cannot start command: state {self.current_state}", error=True); return
        self.agent_mode_active = True
        self.start_recording()

    # --- Settings Application Logic (delegando para ConfigManager e outros) ---
    def apply_settings_from_external(self, **kwargs):
        logging.info("AppCore: Applying new configuration from external source.")
        config_changed = False

        # Atualizar ConfigManager e verificar se houve mudanças
        for key, value in kwargs.items():
            # Mapear nomes de kwargs para chaves de config_manager se necessário
            config_key_map = {
                "new_key": "record_key", "new_mode": "record_mode", "new_auto_paste": "auto_paste",
                "new_sound_enabled": "sound_enabled", "new_sound_frequency": "sound_frequency",
                "new_sound_duration": "sound_duration", "new_sound_volume": "sound_volume",
                "new_agent_key": "agent_key", "new_text_correction_enabled": "text_correction_enabled",
                "new_text_correction_service": "text_correction_service",
                "new_openrouter_api_key": "openrouter_api_key", "new_openrouter_model": "openrouter_model",
                "new_gemini_api_key": "gemini_api_key", "new_gemini_model": "gemini_model",
                "new_agent_model": "gemini_agent_model",
                "new_gemini_prompt": "gemini_prompt",
                "new_prompt_agentico": "prompt_agentico",
                "new_batch_size": "batch_size", "new_gpu_index": "gpu_index",
                "new_hotkey_stability_service_enabled": "hotkey_stability_service_enabled", # Nova configuração unificada
                "new_min_transcription_duration": "min_transcription_duration",
                "new_save_audio_for_debug": "save_audio_for_debug",
                "new_gemini_model_options": "gemini_model_options",
                "new_use_vad": "use_vad",
                "new_vad_threshold": "vad_threshold",
                "new_vad_silence_duration": "vad_silence_duration",
                "new_display_transcripts_in_terminal": "display_transcripts_in_terminal"
            }
            mapped_key = config_key_map.get(key, key) # Usa o nome original se não mapeado

            current_value = self.config_manager.get(mapped_key)
            if current_value != value:
                self.config_manager.set(mapped_key, value)
                config_changed = True
                logging.info(f"Configuração '{mapped_key}' alterada para: {value}")
        
        # Lógica para unificar auto_paste: se new_auto_paste foi passado, ele se aplica a ambos
        if "new_auto_paste" in kwargs:
            new_auto_paste_value = kwargs["new_auto_paste"]
            if self.config_manager.get("auto_paste") != new_auto_paste_value:
                self.config_manager.set("auto_paste", new_auto_paste_value)
                config_changed = True
                logging.info(f"Configuração 'auto_paste' alterada para: {new_auto_paste_value}")
            # Garantir que agent_auto_paste seja sempre igual a auto_paste
            if self.config_manager.get("agent_auto_paste") != new_auto_paste_value:
                self.config_manager.set("agent_auto_paste", new_auto_paste_value)
                config_changed = True
                logging.info(f"Configuração 'agent_auto_paste' (unificada) alterada para: {new_auto_paste_value}")
        
        if config_changed:
            self.config_manager.save_config()
            self._apply_initial_config_to_core_attributes() # Re-aplicar configs ao AppCore
            self.audio_handler.config_manager = self.config_manager # Atualizar referência
            self.transcription_handler.config_manager = self.config_manager # Atualizar referência
            if any(key in kwargs for key in ["new_use_vad", "new_vad_threshold", "new_vad_silence_duration"]):
                self.audio_handler.update_config()
            self.transcription_handler.update_config() # Chamar para recarregar configs específicas do handler
            # Re-inicializar clientes API existentes em vez de recriá-los
            self.gemini_api.reinitialize_client() # Re-inicializar cliente principal
            if self.transcription_handler.gemini_client:
                self.transcription_handler.gemini_client.reinitialize_client() # Re-inicializar cliente Gemini do TranscriptionHandler
            if self.transcription_handler.openrouter_client:
                self.transcription_handler.openrouter_client.reinitialize_client(
                    api_key=self.config_manager.get("openrouter_api_key"),
                    model_id=self.config_manager.get("openrouter_model")
                ) # Re-inicializar cliente OpenRouter do TranscriptionHandler
            
            # Re-registrar hotkeys se as chaves ou modo mudaram
            if kwargs.get("new_key") is not None or kwargs.get("new_mode") is not None or kwargs.get("new_agent_key") is not None:
                self.register_hotkeys()
            
            # Reiniciar/parar serviços de estabilidade de hotkey se a configuração mudou
            if kwargs.get("new_hotkey_stability_service_enabled") is not None:
                if kwargs["new_hotkey_stability_service_enabled"]:
                    # Iniciar thread de re-registro periódico
                    if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                        self.stop_reregister_event.clear()
                        self.reregister_timer_thread = threading.Thread(target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister")
                        self.reregister_timer_thread.start()
                        logging.info("Periodic hotkey re-registration thread launched via settings update.")
                    
                    # Iniciar thread de verificação de saúde
                    if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                        self.stop_health_check_event.clear()
                        self.health_check_thread = threading.Thread(target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread")
                        self.health_check_thread.start()
                        logging.info("Hotkey health monitoring thread launched via settings update.")
                else:
                    self.stop_reregister_event.set()
                    self.stop_health_check_event.set()
                    logging.info("Hotkey stability services stopped via settings update.")

            # Atualizar min_transcription_duration
            if kwargs.get('new_min_transcription_duration') is not None:
                if self.config_manager.get('min_transcription_duration') != kwargs['new_min_transcription_duration']:
                    self.config_manager.set('min_transcription_duration', kwargs['new_min_transcription_duration'])
                    logging.info(f"Configuração 'min_transcription_duration' alterada para: {kwargs['new_min_transcription_duration']}")

            self._log_status("Configurações atualizadas.")
        else:
            logging.info("Nenhuma configuração alterada.")

    def update_setting(self, key: str, value):
        """
        Atualiza uma única configuração e propaga a mudança para os módulos relevantes.
        Usado para atualizações de configuração individuais, como do menu da bandeja.
        """
        old_value = self.config_manager.get(key)
        if old_value == value:
            logging.info(f"Configuração '{key}' já possui o valor '{value}'. Nenhuma alteração necessária.")
            return

        self.config_manager.set(key, value)
        self.config_manager.save_config()
        logging.info(f"Configuração '{key}' alterada para: {value}")

        # Re-aplicar configurações aos atributos do AppCore
        self._apply_initial_config_to_core_attributes()

        # Propagar para TranscriptionHandler se for uma configuração relevante
        if key in ["batch_size_mode", "manual_batch_size", "gpu_index", "min_transcription_duration"]:
            self.transcription_handler.config_manager = self.config_manager # Garantir que a referência esteja atualizada
            self.transcription_handler.update_config()
            logging.info(f"TranscriptionHandler: Configurações de transcrição atualizadas via update_setting para '{key}'.")

        # Re-inicializar clientes API se a chave ou modelo mudou
        if key in ["gemini_api_key", "gemini_model", "gemini_agent_model", "openrouter_api_key", "openrouter_model"]:
            self.gemini_api.reinitialize_client()
            if self.transcription_handler.gemini_client:
                self.transcription_handler.gemini_client.reinitialize_client()
            if self.transcription_handler.openrouter_client:
                self.transcription_handler.openrouter_client.reinitialize_client(
                    api_key=self.config_manager.get("openrouter_api_key"),
                    model_id=self.config_manager.get("openrouter_model")
                )
            logging.info(f"Clientes API re-inicializados via update_setting para '{key}'.")

        # Re-registrar hotkeys se as chaves ou modo mudaram
        if key in ["record_key", "agent_key", "record_mode"]:
            self.register_hotkeys()
            logging.info(f"Hotkeys re-registradas via update_setting para '{key}'.")
        
        # Iniciar/parar serviços de estabilidade de hotkey se a configuração mudou
        if key == "hotkey_stability_service_enabled":
            if value:
                if not self.reregister_timer_thread or not self.reregister_timer_thread.is_alive():
                    self.stop_reregister_event.clear()
                    self.reregister_timer_thread = threading.Thread(target=self._periodic_reregister_task, daemon=True, name="PeriodicHotkeyReregister")
                    self.reregister_timer_thread.start()
                    logging.info("Periodic hotkey re-registration thread launched via update_setting.")
                
                if self.ahk_running and (not self.health_check_thread or not self.health_check_thread.is_alive()):
                    self.stop_health_check_event.clear()
                    self.health_check_thread = threading.Thread(target=self._hotkey_health_check_task, daemon=True, name="HotkeyHealthThread")
                    self.health_check_thread.start()
                    logging.info("Hotkey health monitoring thread launched via update_setting.")
            else:
                self.stop_reregister_event.set()
                self.stop_health_check_event.set()
                logging.info("Hotkey stability services stopped via update_setting.")
        
        logging.info(f"Configuração '{key}' atualizada e propagada com sucesso.")

    # --- Cleanup ---
    def _cleanup_old_audio_files_on_startup(self):
        # Lógica movida de WhisperCore._cleanup_old_audio_files_on_startup
        # ...
        import glob
        import os
        removed_count = 0
        logging.info("Running startup audio file cleanup...")
        try:
            files_to_check = glob.glob("temp_recording_*.wav") + glob.glob("recording_*.wav")
            for f in files_to_check:
                try:
                    os.remove(f)
                    logging.info(f"Deleted old audio file: {f}")
                    removed_count += 1
                except OSError as e:
                    logging.warning(f"Could not delete old audio file '{f}': {e}")
            if removed_count > 0:
                logging.info(f"Cleanup (startup): {removed_count} old audio file(s) removed.")
            else:
                logging.debug("Cleanup (startup): No old audio files found.")
        except Exception as e:
            logging.error(f"Error during startup audio file cleanup: {e}")

    def shutdown(self):
        if self.shutting_down: return
        self.shutting_down = True
        logging.info("Shutdown sequence initiated.")

        self.stop_reregister_event.set()
        self.stop_health_check_event.set()

        try:
            logging.info("Stopping KeyboardHotkeyManager...")
            self._cleanup_hotkeys()
        except Exception as e:
            logging.error(f"Error during hotkey cleanup in shutdown: {e}")

        # Sinaliza para o AudioHandler parar a gravação e processamento
        if self.audio_handler.is_recording:
            logging.warning("Recording active during shutdown. Forcing stop...")
            self.audio_handler.is_recording = False # Sinaliza para a thread de gravação parar
            # try:
            #     self.audio_handler.audio_queue.put_nowait(None) # Sinaliza para a thread de processamento parar
            # except queue.Full:
            #     pass
            if self.audio_handler.audio_stream:
                try:
                    if self.audio_handler.audio_stream.active:
                        self.audio_handler.audio_stream.stop()
                        self.audio_handler.audio_stream.close()
                        logging.info("Audio stream stopped and closed during shutdown.")
                except Exception as e:
                    logging.error(f"Error stopping audio stream on close: {e}")
            self.audio_handler.recording_data.clear()

        with self.transcription_lock:
            if self.transcription_handler.transcription_in_progress:
                logging.warning("Shutting down while transcription is in progress. Transcription may not complete.")

        if self.reregister_timer_thread and self.reregister_timer_thread.is_alive():
            self.reregister_timer_thread.join(timeout=1.5)
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=1.5)

        logging.info("Core shutdown sequence complete.")
