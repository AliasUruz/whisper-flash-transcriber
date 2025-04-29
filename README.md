# Whisper Transcription App

Welcome to the Whisper Transcription App! This is a user-friendly desktop application for Windows that allows you to quickly and accurately transcribe spoken audio into text in real-time. It leverages the power of OpenAI's advanced Whisper model and offers optional integration with OpenRouter or Google Gemini APIs for enhanced text correction. Whether you're a student, professional, or anyone who needs to convert speech to text efficiently, this app is designed to streamline your workflow.

## What Does It Do?

In simple terms, this application listens to your microphone when you press a specific key, converts what you say into written text using artificial intelligence, and then automatically types that text for you wherever your cursor is (like in a document, email, or search bar). You can also connect it to other AI services to automatically fix grammar and punctuation in the transcribed text.

## Key Features

*   **Real-time Audio Transcription:** Captures audio from your microphone and instantly converts it into text using the highly accurate Whisper Large v3 model.
*   **Customizable Hotkey Activation:** Start and stop recording effortlessly with a single press of a keyboard shortcut that you can choose yourself (default is F3).
*   **Toggle Recording:** The hotkey works as a simple on/off switch for recording.
*   **Automatic Text Pasting:** The transcribed text is automatically copied to your computer's clipboard and then pasted directly into the application window that was active when you finished speaking.
*   **Auditory Feedback:** Optional sound cues play when recording starts and stops, so you know exactly when the app is listening.
*   **Intelligent Text Correction (Optional):** Integrate with OpenRouter or Google Gemini APIs to automatically improve the transcribed text's punctuation, capitalization, and overall flow.
*   **Easy Configuration GUI:** A straightforward graphical interface makes it simple to customize hotkeys, select transcription options, and manage API settings without editing code files.

## Getting Started: A Detailed Installation Guide

Follow these steps carefully to get the Whisper Transcription App up and running on your Windows computer.

### Step 1: Install Prerequisites

You need to install two essential tools before setting up the application:

1.  **Install Python 3.8 or higher:**
    *   Python is the programming language the application is built with.
    *   Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
    *   Download the latest version of Python 3.8 or newer for Windows.
    *   Run the downloaded installer.
    *   **VERY IMPORTANT:** On the first screen of the installer, make sure to check the box that says **"Add Python to PATH"**. This step is crucial! If you miss this, you won't be able to run Python commands easily from your terminal. If you forget, you might need to uninstall and reinstall Python.
    *   Follow the rest of the installer prompts (usually clicking "Next" or "Install").
    *   **Verification:** To check if Python was installed correctly and added to PATH, open your Command Prompt (search for `cmd` in the Windows search bar) and type:
        ```bash
        python --version
        ```
        Press Enter. You should see the Python version number printed. If you get an error like " 'python' is not recognized...", Python was not added to PATH correctly.

2.  **Install Git:**
    *   Git is a version control system used to download the project files from GitHub.
    *   Go to the Git website: [https://git-scm.com/downloads](https://git-scm.com/downloads)
    *   Download the installer for Windows.
    *   Run the downloaded installer. You can usually accept the default options during installation.
    *   **Verification:** To check if Git was installed correctly, open your Command Prompt and type:
        ```bash
        git --version
        ```
        Press Enter. You should see the Git version number printed.

### Step 2: Download the Application Code

Now that you have Python and Git installed, you can download the application code from GitHub.

1.  **Open your terminal:** Open Command Prompt or PowerShell.
2.  **Clone the repository:** Run the following command. This will download all the project files into a new folder named `Whisper-local-app` in your current location.
    ```bash
    git clone https://github.com/AliasUruz/Whisper-local-app.git
    ```

3.  **Navigate to the project directory:** Change your current working directory in the terminal to the folder you just cloned:
    ```bash
    cd Whisper-local-app
    ```
    Your terminal prompt should now show that you are inside the `Whisper-local-app` folder.

### Step 3: Set up a Virtual Environment (Best Practice!)

Setting up a virtual environment is highly recommended. It creates an isolated space for this project's Python libraries, preventing them from interfering with libraries used by other Python projects on your computer.

1.  **Create the virtual environment:** While inside the `Whisper-local-app` directory in your terminal, run:
    ```bash
    python -m venv venv
    ```
    This command creates a folder named `venv` inside your project directory, which contains the virtual environment files.

2.  **Activate the virtual environment:** You need to activate the virtual environment in each terminal session you use for this project.
    *   **On Windows Command Prompt:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On Windows PowerShell:**
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
    *   **On Git Bash or other Unix-like shells (like in VS Code terminal):**
        ```bash
        source venv/bin/activate
        ```
    After activation, your terminal prompt should change to include `(venv)` at the beginning, like `(venv) C:\path\to\Whisper-local-app>`. This indicates that the virtual environment is active.

### Step 4: Install Application Dependencies

With your virtual environment activated, you can now install the libraries the application needs to run. These are listed in the `requirements.txt` file.

1.  **Install dependencies:** Run the following command in your activated terminal:
    ```bash
    pip install -r requirements.txt
    ```
    The `pip` command is Python's package installer. The `-r requirements.txt` part tells pip to install everything listed in that file. This step will download and install all necessary packages, including large ones like `torch` and `transformers`. This might take several minutes depending on your internet speed.

2.  **Optional: Install PyTorch with CUDA (For GPU Acceleration):**
    The `requirements.txt` includes a basic installation of PyTorch. However, if you have a compatible NVIDIA graphics card, you can significantly speed up the transcription process by installing a version of PyTorch that uses your GPU (CUDA).
    *   **How to check if you have CUDA:** Open Command Prompt and type `nvcc --version`. If you see version information, CUDA is installed. Note the version number (e.g., CUDA 11.8, CUDA 12.1). If the command is not found, you likely don't have CUDA installed or it's not in your PATH.
    *   **Get the correct command:** Go to the official PyTorch website's installation section: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select your operating system (Windows), Package (Pip), Language (Python), and importantly, your CUDA version (or select "CPU" if you don't have a compatible GPU).
    *   Copy the provided installation command and run it in your **activated virtual environment**.
    *   **Example (for Windows, Pip, Python, CUDA 11.8):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   This command will download and install the GPU-accelerated version of PyTorch. It's a large download. If you already installed the CPU version via `requirements.txt`, this command will upgrade it.
    *   If you do *not* have a compatible GPU or prefer not to use it, you can skip this step. The application will still work using your CPU, just slower.

### Step 5: Run the Application

You are now ready to run the Whisper Transcription App!

1.  **Start the main script:** In your **activated virtual environment** within the `Whisper-local-app` directory, run:
    ```bash
    python whisper_tkinter.py
    ```
    This command executes the main application file.
2.  **Application Window:** A graphical window should appear. This is the application's main interface.
3.  **System Tray Icon:** The application will likely minimize to your Windows system tray (near the clock). You can usually interact with it by right-clicking the icon.

## Configuration Guide (Using the GUI)

The application's settings are managed through its graphical interface and stored in a file named `config.json` in the project directory. This file is automatically created the first time you run `whisper_tkinter.py`.

To access and change settings:

1.  Run the application (`python whisper_tkinter.py`).
2.  Look for a "Settings" or "Configuration" option within the application window or by right-clicking the system tray icon. Click it to open the settings window.

### Key Configuration Options:

*   **Hotkey:** Change the keyboard shortcut used to start and stop recording. The default is F3. Choose a key combination that doesn't conflict with other applications you use frequently.
*   **Enable Text Correction:** Check this box if you want to use an external AI model (OpenRouter or Gemini) to improve the transcribed text.
*   **Text Correction Service:** If text correction is enabled, select whether you want to use "OpenRouter" or "Google Gemini".
*   **API Key:** This is where you enter the API key for the text correction service you selected. **Keep your API keys private!** They are stored locally in `config.json`, which is ignored by Git.
    *   **How to get an OpenRouter API Key:** Visit [https://openrouter.ai/](https://openrouter.ai/) and sign up or log in. You can generate API keys from your account dashboard.
    *   **How to get a Google Gemini API Key:** Visit [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) and generate a new API key.
*   **Model:** For text correction, you can usually select the specific AI model you want to use from the chosen service (e.g., "deepseek/deepseek-chat-v3-0324:free" for OpenRouter, "gemini-2.0-flash-001" for Gemini). Defaults are usually provided.
*   **Sound Feedback:** Toggle sound notifications for recording start/stop.

Remember to save your changes in the settings window.

## How to Use the App

![Screenshot do Aplicativo](./images/image_1.png)

Once the application is running and configured:

1.  **Ensure the app is running:** Check for its icon in the system tray.
2.  **Open the application where you want to paste text:** This could be Notepad, Word, your web browser, etc.
3.  **Press your Hotkey:** Press the hotkey you configured (default F3). The application starts listening. Speak clearly into your microphone.
4.  **Press the Hotkey Again:** Press the *same* hotkey to stop recording.
5.  **Wait for Transcription:** The application will process the audio. This might take a few moments depending on the length of the recording and whether you are using CPU or GPU for transcription. If text correction is enabled, it will also communicate with the API.
6.  **Text Appears:** The transcribed (and corrected) text will automatically appear in the application window that was active when you stopped recording.

## Troubleshooting Common Issues

### Hotkeys Stop Working on Windows 11

This is a known issue related to the underlying libraries. If your main hotkey (default F3) stops working after the first use, try these solutions:

*   **Press the Reload Hotkey (Default: F4):** The application includes a secondary hotkey specifically to try and fix this. Press F4.
*   **Use the System Tray Menu:** Right-click the application's icon in the system tray and look for an option like "Reload Hotkey Listener" or "Restart Keyboard Hook".
*   **Automatic Reload:** The app attempts to automatically handle this in the background, but manual intervention might sometimes be needed.

### PyTorch Installation Problems

If `pip install -r requirements.txt` fails or the application doesn't run due to PyTorch errors:

*   **Verify Python and Pip:** Ensure Python is correctly installed and added to your PATH (check with `python --version` and `pip --version` in Command Prompt).
*   **Virtual Environment:** Make sure your virtual environment is activated (`(venv)` in your terminal prompt).
*   **CUDA Compatibility:** If you are trying to install the CUDA version, double-check that your NVIDIA driver and CUDA toolkit versions are compatible with the PyTorch version you are trying to install, according to the PyTorch website.
*   **Internet Connection:** Ensure you have a stable internet connection, as PyTorch and other libraries are large downloads.

## License

This project is open-source and licensed under the MIT License. You can find the full legal text in the `LICENSE` file included in this repository. This means you are free to use, copy, modify, and distribute the software.

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please:

1.  Fork the repository on GitHub.
2.  Create a new branch for your changes.
3.  Make your changes and commit them with clear messages.
4.  Push your changes to your fork.
5.  Open a Pull Request from your fork to the original repository's `master` branch.
6.  Describe your changes and why they should be included.

## Support

If you need help, encounter bugs, or have questions:

1.  **Check the Troubleshooting section:** Review the common issues listed above.
2.  **Open a GitHub Issue:** The best way to get support is to open a new issue on the repository's GitHub page: [https://github.com/AliasUruz/Whisper-local-app/issues](https://github.com/AliasUruz/Whisper-local-app/issues). Please provide as much detail as possible about your problem, including your operating system, Python version, steps to reproduce the issue, and any error messages.
