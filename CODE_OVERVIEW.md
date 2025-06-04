# Code Overview

This document provides a quick reference to the main modules in this repository. It was generated to help new contributors understand the project structure.

## Main Scripts

| File | Purpose |
| ---- | ------- |
| `whisper_tkinter.py` | Main application logic and GUI using Tkinter. Handles audio recording, transcription with Whisper, text correction via OpenRouter or Gemini APIs, and system tray management. |
| `autohotkey_manager.py` | Manages communication with AutoHotkey scripts for capturing keyboard hotkeys. |
| `keyboard_hotkey_manager.py` | Alternative hotkey handler using the `keyboard` Python library. Allows hotkey configuration and detection on Windows. |
| `win32_hotkey_manager.py` | Hotkey management using the Win32 API with AutoHotkey integration. Provides reliable hotkey detection on Windows 11. |
| `openrouter_api.py` | Client for the OpenRouter API, used to send transcribed text for punctuation and capitalization correction. |
| `gemini_api.py` | Client for Google Gemini API used for more advanced text correction. |
| `fix_imports.js` | Utility script to append `.js` extensions to relative imports in a TypeScript project. Unrelated to the main Python application. |

## Configuration Files

- `config.json` – generated at runtime to store application settings such as hotkeys and API keys.
- `hotkey_config.json` – stores the currently configured hotkeys for the keyboard managers.
- `requirements.txt` – Python dependencies required to run the application.

## Assets

The repository also contains image files (`icon.png`, `image_1.png`, `icon.ico`) used for the system tray icon and README documentation.

## How to Use This Index

Refer to the table above when navigating the code base. Each Python module contains detailed docstrings and logging to further explain its functionality. This overview should make it easier to locate the part of the code you need to modify.
