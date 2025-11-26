# Whisper Flash Transcriber ⚡

An ultra-fast, lightweight audio transcription utility for Windows, powered by the `faster-whisper` model (Large V3 Turbo). Designed for productivity, it allows you to record, transcribe, and automatically paste text into any application with a single hotkey.

## ✨ Features

- **Global Hotkey:** Press `F3` (default) anywhere to toggle recording.
- **Mouse Hotkey:** Optional `LMB + RMB` chord to toggle recording.
- **AI Text Correction:** Optional integration with Google Gemini AI to correct grammar and punctuation automatically.
- **Auto-Paste:** Automatically types the transcribed text into the active window.
- **Local Processing:** Uses `faster-whisper` for high-performance local transcription.
- **GPU Acceleration:** Auto-detects CUDA for blazing fast performance.

*   **Mouse Chord Trigger** (New!): Start recording by holding the **Left Mouse Button** and clicking the **Right Mouse Button**. Perfect for when your hand is already on the mouse.
*   **Smart System Tray**: Runs silently in the system tray. The icon changes color to indicate status (Recording/Processing).
*   **Minimalist Interface**: A clean settings window that stays out of your way when you don't need it.
*   **Resource Management**: Records to RAM for speed but automatically switches to disk for long recordings, protecting your memory.

## 🚀 Installation

### Prerequisites
*   Python 3.11 or higher.
*   (Optional) NVIDIA CUDA drivers installed for GPU acceleration.

### Step-by-Step
1.  Clone this repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/whisper-flash-transcriber.git
    cd whisper-flash-transcriber
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  (Optional) Install GPU support for `faster-whisper` (CTranslate2):
    *   Refer to the [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html) to ensure you have the correct cuDNN and cuBLAS libraries if you want to use the GPU.

## 🛠️ How to Use

1.  Run the main file:
    ```bash
    python src/main.py
    ```
2.  **First Run**: The settings window will open.
    *   **Global Hotkey**: Choose your hotkey (e.g., F3, F12).
    *   **Microphone**: Select your preferred microphone.
    *   **Custom Model Path** (Recommended): Choose a folder on a secondary drive to save the model (approx. 3GB), saving space on your main drive.
3.  **Daily Use**:
    *   The app starts minimized in the tray (near the clock).
    *   **Click the icon** or press the **Hotkey** to start recording.
    *   Speak whatever you want.
    *   Click/Press again to stop.
    *   Wait a few seconds (icon turns purple) and the text will be typed automatically!

## ⚙️ Advanced Configuration

*   **Mouse Shortcut**: Enable the "LMB + RMB" trigger in settings. Disabled by default to avoid conflicts with games/apps.
*   **Auto-paste**: If enabled, the app simulates `Ctrl+V` to paste the text. If disabled, the text is copied to the clipboard only.
*   **Temporary Files**: Long recordings are temporarily saved in `tmp_audio` within the model folder to save RAM.

## 📦 Project Structure

*   `src/main.py`: Application entry point.
*   `src/core.py`: System brain (Audio management, Whisper, Threads).
*   `src/ui.py`: GUI built with Flet.
*   `src/tray.py`: System tray icon and menu management.
*   `src/hotkeys.py`: Global keyboard listener.

## 🤝 Contribution

Contributions are welcome! Feel free to open Issues or Pull Requests for improvements.

## 📄 License

This project is under the MIT license. See the `LICENSE` file for more details.
