# Architecture & Developer Guide: Whisper Flash Transcriber

**Document Version:** 1.0
**Date:** 2025-11-18

## 1. Overview

This document describes the technical architecture and current state of the Whisper Flash Transcriber application. It serves as a reference for developers and contributors.

## 2. Project Mandates

- **Language:** Source code, comments, and logs are in English.
- **Operating System:** Developed and tested for Windows.
- **Dependencies:** Defined in `requirements.txt`.

## 3. Technology Stack

| Component | Technology | Specific Model/Version |
| :--- | :--- | :--- |
| **Language** | Python | 3.11+ |
| **UI Framework** | Flet | (See `requirements.txt`) |
| **Transcription Engine**| `faster-whisper` | `deepdml/faster-whisper-large-v3-turbo-ct2` |
| **Global Hotkeys** | `pynput` | Keyboard and Mouse listeners |
| **Audio Handling**| `sounddevice`, `numpy`, `soundfile` | (See `requirements.txt`) |
| **OS Interaction** | `pyperclip` | For clipboard access |
| **System Tray** | `pystray` | For background operation |
| **AI Correction** | `google-generativeai` | Gemini 1.5 Flash API |

## 4. Configuration Specification

Configuration is stored in `config.json` in the user's home directory: `~/.whisper_flash_transcriber/`.

```json
{
  "hotkey": "f3",
  "auto_paste": true,
  "input_device_index": null,
  "model_path": "",
  "input_device_index": null,
  "model_path": "",
  "gemini_enabled": false,
  "gemini_api_key": "",
  "gemini_prompt": "...",
  "first_run": false
}
```

## 5. UI Behavior and Feedback

The UI provides feedback primarily through the System Tray icon.

### 5.1. UI State Matrix

| System State | Icon Appearance | Tooltip Text |
| :--- | :--- | :--- |
| `idle` | White/Grey Icon | "WF: Ready (GPU/CPU)" |
| `recording` | Red Dot Icon | "WF: Listening..." |
| `transcribing`| Purple Dot Icon | "WF: Processing..." |
| `error` | (State handled internally) | "WF: Error" |

## 6. Module Specifications

### 6.1. Directory Structure
```
/
├── src/
│   ├── main.py       # Entry point
│   ├── ui.py         # Settings Window (Flet)
│   ├── core.py       # Business Logic (Audio, AI)
│   ├── tray.py       # System Tray (Pystray)
│   ├── hotkeys.py    # Global Keyboard Listener
│   ├── hotkeys.py    # Global Keyboard Listener
│   ├── mouse_handler.py # Global Mouse Listener (LMB+RMB)
│   ├── ai_corrector.py # Gemini AI Integration
│   └── icons.py      # Dynamic Icon Generation
└── requirements.txt
```

### 6.2. Module Logic

#### `src/main.py`
- **Responsibility:** Entry point. Initializes Core, UI, and Tray.
- **Logic:**
  - Checks `first_run` to decide whether to show the window or start minimized.
  - Handles clean shutdown via `cleanup_and_exit`.

#### `src/core.py`
- **Responsibility:** The brain. Manages audio recording, Whisper model, and threading.
- **Key Features:**
  - **Hybrid Recording:** Records to RAM by default. If buffer exceeds 30s, flushes to disk (`tmp_audio`) to save memory.
  - **Smart Model Loading:** Tries GPU (Float16) -> GPU (Int8) -> CPU (Int8).
  - **VAD Filter:** Uses Voice Activity Detection to clean up silence.

#### `src/tray.py`
- **Responsibility:** Manages the system tray icon and menu.
- **Logic:**
  - **Dynamic Menu:** Changes "Start Recording" to "Stop & Transcribe" based on state.
  - **Click Action:** Clicking the icon triggers the default action (Start/Stop).

#### `src/ui.py`
- **Responsibility:** The Settings Window.
- **Logic:**
  - **Auto-Save:** Settings are saved immediately upon change.
  - **Minimalist:** Only essential controls are shown.

#### `src/mouse_handler.py`
- **Responsibility:** Monitors mouse clicks for the "Chord" pattern (Hold Left + Click Right).
- **Logic:**
  - Runs in a separate thread via `pynput`.
  - Only active if enabled in settings.
  - Only active if enabled in settings.
  - Triggers `core.toggle_recording()`.

#### `src/ai_corrector.py`
- **Responsibility:** Handles communication with Google Gemini API.
- **Logic:**
  - **Timeout:** Enforces a 5-second timeout to ensure "Flash" speed.
  - **Cleaning:** Strips markdown and quotes from AI response.
  - **Fallback:** Returns original text if API fails or times out.

## 7. Logging

Logging is printed to `stdout` with timestamps and log levels (`INFO`, `ERROR`). It covers all major events (Startup, Model Load, Recording, Transcription).
