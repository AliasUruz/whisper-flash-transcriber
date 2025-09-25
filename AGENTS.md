# AGENTS.md - Master Project Context Document

**Purpose**: This document is the single source of truth for any human or AI agent that needs to understand, operate, maintain, or extend the **Whisper Flash Transcriber** project. Read it thoroughly before making any modifications.

---

## 1. Core Directives & User Preferences

This section outlines the fundamental rules and user-mandated instructions that all agents must follow.

### Agent Operational Mandates
- **Detailed Planning**: Always provide comprehensive plans, step-by-step guides, and detailed checklists when proposing tasks or changes.
- **Plan Replication**: When asked to create a plan file, generate a context-rich, well-developed, and well-written plan, including checklists. Be sure that it is very well developed and contextualized.
- **Minimalist Dependencies**: Install only the absolute minimum required dependencies for any given task. Avoid bloating the environment with unnecessary packages.
- **Full Compliance**: Strictly and completely adhere to all user requests without deviation unless clarification is sought and granted.

### User Communication Protocol
- **Primary Language**: All communication with the user **must be in Brazilian Portuguese (pt-BR)**. This is a strict requirement. The initial greeting and all subsequent interactions should respect this rule.

---

## 2. Quick Technical Sheet

- **Target System**: Windows 10+ (Desktop). Partial compatibility via WSL2 for CLI-only usage.
- **Primary Language**: Python 3.11+ (tested up to 3.13.2).
- **UI Framework**: `tkinter` (for the settings window) + `pystray` (for the system tray icon).
- **Input/Output**: Audio capture via `sounddevice`; automatic text pasting using `pyautogui` and `pyperclip`.
- **Primary ASR Engine**: Hugging Face Whisper models, accessed through `transformers`, `faster-whisper`, or `ctranslate2` backends.
- **Optional External Services**: Google Gemini (for text correction/agent mode) and OpenRouter (for text correction).
- **Execution Model**: System tray application driven by global hotkeys, managed by the `keyboard` library.

### Essential Directory Structure
| Path | Contents | Notes |
| --- | --- | --- |
| `src/` | Main application source code. | See component map in the architecture section. |
| `src/asr/` | Specific ASR backend adapters (e.g., faster-whisper, transformers). | Contains the pipeline adapters for different ASR libraries. |
| `src/utils/` | Shared utilities for memory, autostart, UI helpers, etc. | Reused by core components and the UI. |
| `plans/` | Tactical plans and agent-generated artifacts. | Any agent must register new plans and strategies here. |
| `docs/` | Supplementary documentation (workflows, UI variables, changelog). | Use as a reference for deeper understanding. |
| `tests/` | The current `pytest` test suite. | Currently focused on the VAD pipeline. |
| `config.json` / `secrets.json` | Persisted user settings and API keys. | Automatically created on the first run. |

---

## 3. High-Level Architecture

The application is orchestrated by a central `AppCore` class, which acts as the main coordinator.

```
main.py (Tk bootstrap) -> AppCore (Orchestrator)
    |- ConfigManager (handles config.json + secrets.json)
    |- StateManager (serves as an event bus)
    |- AudioHandler (manages audio capture + VAD)
    |- TranscriptionHandler (runs ASR + post-processing)
    |- ActionOrchestrator (pastes text, manages clipboard, runs agent mode)
    |- KeyboardHotkeyManager (listens for global hotkeys)
    |- GeminiAPI / OpenRouterAPI (optional external services)
    |- ModelManager (manages ASR model catalog and downloads)
UI Thread -> UIManager (manages tray icon + settings window)
```

### Component Responsibilities
| Component | File | Key Functions |
| --- | --- | --- |
| `AppCore` | `src/core.py` | The application's heart. Manages global state, dispatches events, synchronizes the UI, enables/disables hotkeys, and controls model downloads. |
| `StateManager` | `src/state_manager.py` | Normalizes and propagates state events, prevents duplicates, and ensures thread-safe delivery of state changes to listeners. |
| `AudioHandler` | `src/audio_handler.py` | Manages all aspects of audio capture, switches between RAM and disk storage, applies VAD, and sends prepared audio segments to the core. |
| `TranscriptionHandler` | `src/transcription_handler.py` | Loads the ASR backend, executes transcriptions in a `ThreadPoolExecutor` to prevent UI freezing, and integrates AI corrections from Gemini/OpenRouter. |
| `ActionOrchestrator` | `src/action_orchestrator.py` | Determines the final workflow for the transcribed text (clipboard, auto-paste, agent mode) and cleans up temporary files. |
| `KeyboardHotkeyManager` | `src/keyboard_hotkey_manager.py` | Registers global hotkeys using the `keyboard` library, provides key detection, and handles automatic re-registering for stability. |
| `UIManager` | `src/ui_manager.py` | Generates and manages the system tray icon, its context menus, and the `customtkinter` settings window. Applies configuration changes via `ConfigManager`. |
| `ModelManager` | `src/model_manager.py` | Provides a curated catalog of ASR models, synchronizes the local cache, and handles model downloads with cancellation and timeout logic. |
| `GeminiAPI` / `OpenRouterAPI` | `src/gemini_api.py`, `src/openrouter_api.py` | Encapsulate API calls for text correction and agent mode, handling retries, timeouts, and error fallback gracefully. |

### Threading and Concurrency
- **Main Thread**: Dedicated to the UI, running `Tk.mainloop()` and the `pystray` icon loop. All blocking operations must occur elsewhere.
- **`AudioHandler`**: Uses a dedicated `AudioRecordThread` for capture and a separate thread for processing the audio queue.
- **`TranscriptionHandler`**: Manages a `ThreadPoolExecutor` with one worker for transcription and a `ModelLoadThread` for loading models in the background.
- **Hotkey Stability Services**: Conditional threads (`HotkeyReregisterThread`, `HotkeyHealthThread`) that ensure hotkeys remain active.
- **Core Locks**: `recording_lock`, `transcription_lock`, `state_lock`, `keyboard_lock`, `agent_mode_lock`. You **must** respect these locks when extending functionality to prevent race conditions.

---

## 4. Operational Workflows

### 4.1 Application Initialization
1.  `main.py` sets up environment variables, logging, and CUDA diagnostics.
2.  It creates a hidden root Tk window, registers an `atexit` handler, and patches `tk.Variable.__del__` to prevent shutdown errors.
3.  It instantiates `AppCore` and `UIManager`, linking them via `app_core.ui_manager`.
4.  `AppCore` synchronizes settings, checks the model cache, and triggers a model download or background loading via `TranscriptionHandler`.
5.  `UIManager` builds the system tray icon, its dynamic menus, and associated event listeners.

### 4.2 Capture and Transcription Flow
1.  A global hotkey press is detected by `KeyboardHotkeyManager`, which calls `AppCore.toggle_recording()`.
2.  `AudioHandler` starts capturing audio, deciding between RAM or disk storage, and applying VAD if enabled.
3.  When recording stops, the audio segment is passed to `ActionOrchestrator`, which triggers `TranscriptionHandler.transcribe_audio_segment()`.
4.  `TranscriptionHandler` executes the ASR backend, emitting partial results via callbacks and a final result upon completion.
5.  **Optional AI Pipeline**: The result can be routed to Agent Mode (Gemini) or Text Correction (Gemini/OpenRouter). The system gracefully falls back to the raw text if the API call fails.
6.  `ActionOrchestrator` copies the final text to the clipboard, performs an auto-paste if enabled, and signals the final state `TRANSCRIPTION_COMPLETED`.

### 4.3 State and UI Synchronization
- Every significant event flows through `StateManager.set_state(event, details, source)`.
- A `StateEvent` maps to a specific application state (e.g., `IDLE`, `RECORDING`, `TRANSCRIBING`, `ERROR_*`). See Section 6 for the full map.
- `UIManager` is a passive listener; it reacts to state notifications to update the tray icon color, tooltips, and windows, but does not initiate core logic.

### 4.4 Graceful Shutdown
1.  `AppCore.shutdown()` sets the `shutting_down` flag, cancels any ongoing downloads, and signals the hotkey and health threads to terminate.
2.  It gracefully shuts down `KeyboardHotkeyManager`, `TranscriptionHandler` (clearing the executor and GPU cache), and `AudioHandler`.
3.  Finally, it stops the system tray icon and terminates the `Tk.mainloop()`.

---

## 5. Configuration, Secrets, and Variables

### 5.1 Configuration Sources
- **`config.json`**: Contains non-sensitive user preferences like hotkeys, ASR backend choice, and feature flags.
- **`secrets.json`**: Stores sensitive API keys (`gemini_api_key`, `openrouter_api_key`). This file is in `.gitignore`.
- **`ConfigManager`**: The sole authority for accessing and modifying settings. It uses `config_schema.AppConfig` for validation.

### 5.2 Key Configuration Options
| Key | Type | Impact |
| --- | --- | --- |
| `record_key`, `record_mode` | string | Defines the capture hotkeys (toggle vs. press-to-record). Directly used by `KeyboardHotkeyManager`. |
| `auto_paste`, `agent_auto_paste` | bool | Controls automatic pasting behavior in `ActionOrchestrator`. |
| `asr_model_id`, `asr_backend`, `asr_compute_device` | string | Defines the active ASR model, backend library, and compute device (e.g., 'cuda', 'cpu'). |
| `text_correction_enabled`, `text_correction_service` | bool/string | Enables the AI text correction pipeline and selects the provider (Gemini/OpenRouter). |
| `gemini_model`, `gemini_agent_model` | string | Specifies the exact models to be used for Gemini API requests. |
| `record_storage_mode`, `max_memory_seconds_mode` | string | Controls whether `AudioHandler` uses RAM or disk for storing recordings. |
| `hotkey_stability_service_enabled` | bool | Enables background threads to ensure hotkeys remain registered and functional. |
| `launch_at_startup` | bool | Uses `utils.autostart` to create or remove a shortcut in the Windows startup folder. |

### 5.3 Writing to Configuration
- **Always** use `ConfigManager` methods (`set_*`, `save_config`) to mutate and persist settings.
- The `save_config` method automatically separates secrets into `secrets.json`.

### 5.4 UI Variables
- Refer to `docs/ui_vars.md` for a complete mapping of `ctk.*Var` instances to their corresponding UI widgets and configuration keys.
- When adding new controls, align them with `ConfigManager` and update this documentation.

---

## 6. State Management (`StateManager`)

### 6.1 Primary States
`IDLE`, `LOADING_MODEL`, `RECORDING`, `TRANSCRIBING`, `ERROR_MODEL`, `ERROR_AUDIO`, `ERROR_TRANSCRIPTION`, `ERROR_SETTINGS`.

### 6.2 Common Events and Sources
| Event (`StateEvent`) | Emitter | Resulting State | Notes |
| --- | --- | --- | --- |
| `MODEL_MISSING`, `MODEL_CACHE_INVALID` | `AppCore` | `ERROR_MODEL` | Triggered by cache validation failures or a missing model. |
| `MODEL_DOWNLOAD_STARTED` | `AppCore` | `LOADING_MODEL` | Indicates a model download is in progress. |
| `MODEL_DOWNLOAD_FAILED` | `AppCore` | `ERROR_MODEL` | An exception occurred in `ModelManager.ensure_download`. |
| `MODEL_READY` | `TranscriptionHandler` -> `AppCore` | `IDLE` | The model is loaded into memory, and hotkeys are activated. |
| `AUDIO_RECORDING_STARTED/STOPPED` | `AudioHandler` | `RECORDING`/`IDLE` | Reflects the audio capture status. |
| `TRANSCRIPTION_STARTED/COMPLETED` | `TranscriptionHandler` | `TRANSCRIBING`/`IDLE` | Indicates ASR processing is active. |
| `AUDIO_ERROR`, `MODEL_LOADING_FAILED`, `SETTINGS_*` | Various Components | Corresponding `ERROR_*` states. | |

### 6.3 Best Practices
- Always use `StateManager.set_state` (or a `StateEvent`) to signal a status change that should be visible to the user.
- Avoid manipulating the UI directly from core logic; let `UIManager` react to state changes.
- When adding new events, update the `STATE_FOR_EVENT` and `EVENT_DEFAULT_DETAILS` maps in `StateManager`.

---

## 7. Dependencies and Environment

### 7.1 `requirements*.txt` Files
| File | Usage | Notes |
| --- | --- | --- |
| `requirements.txt` | Core application dependencies (CPU execution). | Version adjustments should follow the plan in `plans/2025-09-24-dependency-remediation.md`. |
| `requirements-optional.txt` | Optional features (GPU acceleration, optimizations). | Place specific libraries like `bitsandbytes` or CUDA-enabled torch here. |
| `requirements-test.txt` | Development and testing. | Includes `pytest`, `flake8`, and other developer tools. |

### 7.2 General Rules
- Always use isolated `venv` environments to prevent conflicts.
- Update dependencies cautiously and always run the `pytest` suite after any change.
- For GPU-enabled Torch, document the need for the extra index URL (`https://download.pytorch.org/whl/torch_stable.html`) in the main `README.md`.

---

## 8. Daily Operations

### 8.1 Basic Commands
```powershell
# Activate the virtual environment
.\.venv\Scripts\activate

# Run the application
python src/main.py

# Run the test suite
pytest
```

### 8.2 Logs and Metrics
- Logging is configured via `logging_utils.setup_logging` to output to both a file and stdout.
- Look for common prefixes like `[METRIC] stage=*` for performance and status messages.
- The logging level can be adjusted via the `LOGGER_LEVEL` environment variable (see `logging_utils`).

### 8.3 Artifact Locations
| Item | Location |
| --- | --- |
| Rotating log files | `logs/` directory (configuration is in the logger setup). |
| Temporary audio files | Project root (`temp_recording_*.wav`, `recording_*.wav`). |
| Agent action plans | `plans/` directory. |

---

## 9. Quick Troubleshooting Guide

| Symptom | Likely Cause | Suggested Action |
| --- | --- | --- |
| Hotkey doesn't start recording | `KeyboardHotkeyManager` failed to register, or there are conflicts with the `keyboard` library. | Check for `StateEvent.SETTINGS_HOTKEY_START_FAILED`, restart hotkeys via the settings UI, and validate admin privileges. |
| Stuck in `ERROR_MODEL` state | Invalid cache or a canceled/failed download. | Check the `asr_cache_dir` path, execute the dependency remediation plan, and trigger a model re-install from the UI. |
| AI corrections are ignored | Missing API keys or the service is unavailable. | Validate `secrets.json`, check the logs for Gemini/OpenRouter API errors. The system should fall back to raw text. |
| Audio errors (overflow) | High system latency or insufficient resources. | Adjust `chunk_length_sec` in settings, enable disk storage mode, and inspect logs for `Audio input overflow` messages. |
| Crash on exit | Resources not cleaned up properly. | Ensure `AppCore.shutdown()` is called (via Tray Icon -> Exit), and review for any lingering threads. |

---

## 10. Extending and Customizing

### 10.1 Adding a New ASR Backend
1.  Create a new adapter module in `src/asr/`. Follow the class and method signature of `backend_faster_whisper.py`.
2.  Update `asr_backends.py` and `model_manager.CURATED` with the new backend ID and its associated models.
3.  Update `config_schema.AppConfig` to accept the new backend identifier as a valid value.
4.  Add tests for the new backend and document its existence in `AGENTS.md` and `README.md`.

### 10.2 Adding a New AI Correction Provider
1.  Create a new API wrapper class (e.g., `src/<provider>_api.py`).
2.  Integrate it into `TranscriptionHandler`'s `_process_ai_pipeline` method.
3.  Expose its configuration options in `ConfigManager`, `config_schema`, and the UI if applicable.
4.  Update `StateManager` and `ActionOrchestrator` as needed to handle new states or outcomes.

### 10.3 Adding New Hotkeys
1.  Extend `KeyboardHotkeyManager` with a new callback method.
2.  Update `AppCore` to handle the event triggered by the new hotkey.
3.  Expose the hotkey configuration in the UI and save it via `ConfigManager`.

---

## 11. Essential Checklists

### 11.1 Pre-Commit Checklist
- [ ] Run `python -m compileall src` to check for syntax errors.
- [ ] Execute `pytest` and ensure all tests pass.
- [ ] Update relevant documentation (`AGENTS.md`, `README.md`, `docs/`).
- [ ] If the change is significant, create a plan or note in the `plans/` directory.
- [ ] Verify that `config.json` and `secrets.json` have not been accidentally staged for commit.

### 11.2 Pre-Release Checklist
- [ ] Run `pip install -r requirements.txt` in a clean environment.
- [ ] Test the full end-to-end flow: model download, recording, transcription, and AI correction (if enabled).
- [ ] Verify hotkey integration on a target Windows system (check for permissions issues, antivirus conflicts).
- [ ] Confirm that the system tray icon correctly responds to all application states (`IDLE`, `RECORDING`, `TRANSCRIBING`, `ERROR_*`).

---

## 12. Appendices

### 12.1 Helper Scripts
| Script | Location | Description |
| --- | --- | --- |
| `processa_fila.sh` | Root | Legacy helper script for batch processing on Linux/WSL. Review before use. |
| `Run Whisper.bat` | `src/` | A Windows batch shortcut to start the application (maintained for compatibility). |

### 12.2 Relevant External Documents
- `docs/model-loading-flow.md`: Detailed breakdown of the model download and loading lifecycle.
- `docs/ui_vars.md`: Complete mapping of UI variables to config keys.
- `docs/changelog.md`: A history of significant changes.
- `plans/2025-09-24-dependency-remediation.md`: The current plan for dependency adjustments.

### 12.3 Quick Glossary
| Term | Meaning |
| --- | --- |
| **ASR** | Automatic Speech Recognition. |
| **VAD** | Voice Activity Detection. |
| **Agent Mode** | A workflow where the transcribed text is sent to the Gemini API to execute commands or generate responses, rather than just being pasted. |
| **Hotkey Stability Service** | Periodic background threads that ensure the global hotkeys remain registered with the OS. |

---

Maintain this document whenever new features, workflows, or dependencies are introduced. Anything not formalized here is likely to be lost or misinterpreted by automated agents.