# Build Manual: Whisper Flash Transcriber v2

**Document Version:** 4.0
**Date:** 2025-11-05

## 1. Project Mandates

This document outlines the definitive technical specifications for building the application. All development must adhere to the following mandates:

- **Language:** All source code, variables, class names, comments, UI text, and terminal logs MUST be in English.
- **Operating System:** The primary and sole target for the MVP is Windows. No cross-platform compatibility code (e.g., for macOS) should be included.
- **Dependencies:** The dependency list is final. No additional third-party libraries may be introduced without updating this plan.

## 2. Project Objectives

- Build a desktop audio transcription application from scratch.
- Prioritize simplicity, performance, and maintainability.
- Focus exclusively on the hotkey-driven transcription flow for the initial MVP.
- Utilize a modern and minimal technology stack.

## 3. Technology Stack Specification

| Component | Technology | Specific Model/Version |
| :--- | :--- | :--- |
| **Language** | Python | 3.11+ |
| **UI Framework** | Flet | Latest stable release |
| **Transcription Engine**| `faster-whisper` | `openai/whisper-large-v3-turbo` |
| **Global Hotkeys** | `pynput` | Latest stable release |
| **Audio Handling**| `sounddevice`, `numpy`, `soundfile` | Latest stable releases |
| **OS Interaction** | `pyperclip` | For clipboard access |

## 4. Configuration Specification

Application configuration will be stored in a `config.json` file located in the user's data directory.

### 4.1. File Structure

```json
{
  "hotkey": "f3",
  "recording_mode": "toggle",
  "auto_paste": true
}
```
- `hotkey`: The key combination to trigger recording.
- `recording_mode`: The operational mode. Valid values: `"toggle"` or `"press"`.
- `auto_paste`: A boolean (`true`/`false`) determining if the text is automatically pasted.

## 5. First-Run Behavior

- On the first application launch, the system will check for the existence of `config.json`.
- If the file does not exist, it will be **created silently** with the default values specified in section 4.1.
- The application will not prompt the user for initial setup. The settings window can be accessed via the tray icon.

## 6. UI Behavior and User Feedback

UI must provide clear, immediate feedback via the system tray icon.

### 6.1. UI State Matrix

| System State | Tray Icon Appearance | Tooltip Text |
| :--- | :--- | :--- |
| `idle` | Green icon | "Ready to record. Press the hotkey." |
| `recording` | Red icon | "Recording... Press the hotkey to stop." |
| `transcribing`| Blue icon | "Transcribing... Please wait." |
| `error` | Orange icon | "Error: Check logs or hover for details." |

### 6.2. Error Notification

- In addition to the `error` state in the tray, critical, user-facing errors (e.g., "microphone not found") **must trigger a native OS pop-up (dialog box)** displaying a clear error message.

## 7. Architecture and Module Specifications

### 7.1. Directory Structure
```
/
├── src/
│   ├── main.py
│   ├── ui.py
│   ├── core.py
│   └── hotkeys.py
└── requirements.txt
```

### 7.2. Module Specifications with Internal Logic

#### `src/main.py`
- **Responsibility:** Application entry point. Initializes and connects all components.
- **Logic:**
  ```python
  # main.py
  import flet as ft
  import threading
  from ui import AppUI
  from core import CoreService
  from hotkeys import HotkeyManager

  def main(page: ft.Page):
      # 1. Initialize CoreService. It will load or create config.
      core = CoreService()

      # 2. Initialize AppUI, passing page and core references.
      ui = AppUI(page, core)

      # 3. Set the UI update callback in the core service.
      core.set_ui_update_callback(ui.update_status)

      # 4. Add the UI controls to the Flet page.
      page.add(ui.build_controls())

      # 5. Initialize HotkeyManager and start it in a daemon thread.
      hotkey_manager = HotkeyManager(core)
      threading.Thread(target=hotkey_manager.start_listening, daemon=True).start()

      # 6. Set initial window state (hidden).
      page.window_visible = False
      page.update()
  ```

#### `src/ui.py`
- **Responsibility:** Manages all visual elements (settings window and tray icon).
- **Logic:**
  ```python
  # ui.py
  import flet as ft
  from core import CoreService

  class AppUI:
      def __init__(self, page: ft.Page, core: CoreService):
          # 1. Store page and core references.
          # 2. Define all Flet controls (TextField for hotkey, Switch for auto-paste, Buttons).
          # 3. Define the ft.TrayIcon with its menu items ('Settings', 'Exit').
          # 4. Append the TrayIcon to the page overlay.

      def build_controls(self) -> ft.Column:
          # Returns the Flet Column containing all settings controls.
          # Values are pre-filled from `self.core.get_setting('key')`.

      def update_status(self, status: str, tooltip: str):
          # 1. Update `self.tray_icon.tooltip`.
          # 2. Select the icon path based on `status` (e.g., 'assets/icon_{status}.ico').
          # 3. Update `self.tray_icon.icon`.
          # 4. Call `self.page.update()` to refresh the UI.

      def _save_settings(self, e):
          # 1. Get values from the UI controls (`hotkey_field.value`, etc.).
          # 2. Create a settings dictionary.
          # 3. Call `self.core.save_settings(new_settings)`.
          # 4. Hide the window: `self.page.window_visible = False`.
          # 5. Call `self.page.update()`.

      def show_error_popup(self, title: str, message: str):
          # Create and show a Flet dialog/alert with the given title and message.
          pass
  ```

#### `src/core.py`
- **Responsibility:** Orchestrates recording, transcription, and result actions.
- **Recording Logic:** Implements the **hybrid recording strategy**. Recording starts in RAM. If duration exceeds 30 seconds or memory usage hits 50MB, RAM content is flushed to a temporary file on disk, and subsequent audio is appended to that file.
- **Logic:**
  ```python
  # core.py
  import numpy as np
  from faster_whisper import WhisperModel
  import sounddevice as sd
  import pyperclip
  from pynput.keyboard import Controller, Key
  import tempfile
  import json

  class CoreService:
      def __init__(self):
          # 1. Load settings from config.json or create it with defaults if not found.
          # 2. Load the `faster-whisper` model. Handle potential errors and update state.
          # 3. Initialize state variables: `self.state = "idle"`, etc.

      def set_ui_update_callback(self, callback):
          # Store the UI update function reference.

      def toggle_recording(self):
          # 1. If self.state is "recording", call `self._stop_recording()`.
          # 2. Else if self.state is "idle", call `self._start_recording()`.
          # 3. Otherwise, log that the action is ignored.

      def _start_recording(self):
          # 1. Set `self.state = "recording"`.
          # 2. Call `self.ui_update_callback("recording", "Recording...")`.
          # 3. Clear `self.audio_frames` and `self.temp_file`.
          # 4. Start the `sounddevice.InputStream` with `self._audio_callback`.

      def _stop_recording(self):
          # 1. Stop the `sounddevice.InputStream`.
          # 2. Set `self.state = "transcribing"`.
          # 3. Call `self.ui_update_callback("transcribing", "Transcribing...")`.
          # 4. Start `self._process_audio()` in a new thread to avoid blocking.

      def _audio_callback(self, indata, frames, time, status):
          # The core of the hybrid recording logic.
          # 1. Append `indata.copy()` to `self.audio_frames`.
          # 2. Check if total frames > (30 * sample_rate) or memory size > 50MB.
          # 3. If so, and if `self.temp_file` is None, create a temp wave file.
          # 4. Write the contents of `self.audio_frames` to the file and clear the list.
          # 5. If `self.temp_file` already exists, just append the new `indata` to it.

      def _process_audio(self):
          # 1. Determine the audio source: if `self.temp_file` exists, use its path.
          # 2. Otherwise, `np.concatenate(self.audio_frames)`.
          # 3. Call `self._run_transcription(audio_source)`.
          # 4. Call `self._handle_result(transcribed_text)`.
          # 5. Clean up temp file if it exists.
          # 6. Set `self.state = "idle"`.
          # 7. Call `self.ui_update_callback("idle", "Ready to record.")`.

      def _run_transcription(self, audio_input) -> str: ...

      def _handle_result(self, text: str):
          # 1. `pyperclip.copy(text)`.
          # 2. If `self.settings['auto_paste']` is true:
          #    a. Create a `pynput.keyboard.Controller`.
          #    b. Simulate a `Ctrl+V` press and release.

      def save_settings(self, settings: dict):
          # 1. Update `self.settings`.
          # 2. Save the full settings dictionary to `config.json`.
          # 3. Notify the `HotkeyManager` to re-register the hotkey if it changed.
  ```

#### `src/hotkeys.py`
- **Responsibility:** Manages the global hotkey listener in a dedicated thread.
- **Logic:**
  ```python
  # hotkeys.py
  from pynput import keyboard
  from core import CoreService

  class HotkeyManager:
      def __init__(self, core: CoreService): ...

      def _on_press(self, key):
          # 1. Check if the pressed key matches the registered hotkey combination.
          # 2. If it matches, call `self.core.toggle_recording()`.

      def start_listening(self):
          # 1. Create and start `pynput.keyboard.Listener`.
          # 2. The listener runs in its own thread, calling `_on_press`.
  ```

## 8. Logging and Diagnostics

- Logging must be **highly detailed** and printed to the standard terminal (`stdout`).
- Every significant action must be logged with a timestamp, log level (`INFO`, `ERROR`), and a descriptive message.
- **Examples of events to log:** Application start, model loading (start and end), settings loaded, hotkey listener started, hotkey pressed, recording started/stopped, transcription started/finished, text copied to clipboard, errors encountered.

## 9. Packaging and Distribution

- The application will be packaged into a single executable for Windows using `flet pack`.
- A directory named `assets` will be created in the project root to store `icon.ico` and related icon files.
- The final pack command will be:
  `flet pack src/main.py --name WhisperFlashTranscriber --icon assets/icon.ico --add-data "path/to/whisper/model:model" --add-data "assets:assets"`

## 10. Granular Implementation Plan

(This section remains the same as the previous version, as it is already granular.)
