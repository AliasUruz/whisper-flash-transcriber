# Whisper Transcription App

A desktop application for real-time audio transcription using OpenAI's Whisper model, with optional text correction via OpenRouter or Google Gemini APIs.

## Features

- Real-time audio transcription using the Whisper Large v3 model.
- Activation via a configurable hotkey (default: F3).
- Toggle recording mode (start/stop with the same key).
- Automatic pasting of transcribed text to the active application.
- Configurable sound feedback.
- Optional text correction via OpenRouter API or Google Gemini API (improves punctuation, capitalization, and formatting).
- Graphical user interface (GUI) for easy configuration.

## Requirements

To run this application, you need:

- Python 3.8 or higher.
- The dependencies listed in `requirements.txt`.

## Installation

Follow these steps to set up the application:

1.  **Clone the repository:**
    If you haven't already, clone the project repository from GitHub:
    ```bash
    git clone https://github.com/AliasUruz/Whisper-local-app.git
    cd Whisper-local-app
    ```

2.  **Set up a virtual environment (Recommended):**
    It's highly recommended to use a virtual environment to avoid conflicts with other Python projects.
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    With the virtual environment activated, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary packages, including `torch`, `transformers`, `sounddevice`, etc.

4.  **Install PyTorch (if not installed by requirements.txt):**
    Depending on your system and `requirements.txt`, you might need to install PyTorch separately with CUDA support for GPU acceleration. Visit the official PyTorch website ([https://pytorch.org/](https://pytorch.org/)) for specific installation instructions based on your operating system and CUDA version.
    Example (for CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    If you don't have a compatible GPU or prefer to use your CPU, the basic `pip install -r requirements.txt` should suffice for the CPU version of PyTorch.

5.  **Run the application:**
    After installing dependencies, you can run the main application script:
    ```bash
    python whisper_tkinter.py
    ```

## Configuration

Upon the first execution, a `config.json` file will be automatically created in the application directory with default settings.

To configure the application, including hotkeys and API settings, run `whisper_tkinter.py` and use the graphical interface.

**Configuring Text Correction (OpenRouter or Gemini):**

1.  Open the application's settings via the GUI.
2.  Enable the "Enable Text Correction" option.
3.  Select your desired service (OpenRouter or Gemini).
4.  Configure the selected service as instructed below:

    *   **OpenRouter Configuration:**
        1.  Obtain an API key from [OpenRouter](https://openrouter.ai).
        2.  Enter your API key in the corresponding field in the application settings.
        3.  The default model is "deepseek/deepseek-chat-v3-0324:free". You can change this if needed.

    *   **Google Gemini Configuration:**
        1.  Obtain an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
        2.  Enter your API key in the corresponding field in the application settings.
        3.  The default model is "gemini-2.0-flash-001". You can change this if needed.

**Important Note on API Keys:** Your API keys are stored locally in the `config.json` file. This file is included in the `.gitignore` and will **not** be uploaded to GitHub, ensuring your privacy.

## Usage

1.  Run the application (`python whisper_tkinter.py`).
2.  The application will appear as an icon in your system tray.
3.  Press the configured hotkey (default: F3) to start recording audio. The icon might change or a notification might appear (depending on implementation) to indicate recording is active.
4.  Speak the text you want to transcribe.
5.  Press the same hotkey again to stop the recording.
6.  The transcribed text will be processed. If text correction is enabled, it will be sent to the selected API.
7.  The final transcribed (and corrected, if applicable) text will be automatically copied to your clipboard and pasted into the application that was active when you stopped recording.

## Known Issues and Solutions

### Keyboard Library Bug on Windows 11

On some Windows 11 systems, the `keyboard` library might stop responding after the first use of the hotkey. To address this, the application includes workarounds:

1.  **Reload Hotkey (Default: F4):** Press this dedicated hotkey to attempt to reload the keyboard library and restore hotkey functionality.
2.  **System Tray Context Menu Option:** Right-click the application's system tray icon and select a "Reload Keyboard/Hotkey" or similar option.
3.  **Periodic Automatic Reload:** The application attempts to automatically reload hotkeys periodically in the background to mitigate this issue.

If your hotkeys stop working, try one of these methods to restore them.

## License

This project is licensed under the MIT License. See the `LICENSE` file for full details.

## Contributing

(Optional section - can be added later if you plan to accept contributions)

## Support

(Optional section - can be added later with information on how to get help)
