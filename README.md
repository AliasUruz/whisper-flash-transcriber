### Whisper Local App  

Whisper Local App is a lightweight, fully local recording and transcription application powered by OpenAI's Whisper "large" model. All processing happens on your machine, ensuring full privacy with no data sent to the cloud.  

---

### Features  

- **Local Recording & Transcription**  
  Record audio locally, and once you stop, the app automatically transcribes it.  

- **Pause & Resume Recording**  
  Pause and resume the recording without losing previously recorded audio.  

- **Dark Mode Interface**  
  A sleek, always-on-top dark-themed interface.  

- **Automatic Model Updates**  
  The app checks if the "large" model is outdated (more than 7 days old) and updates it automatically.  

- **Real-Time Transcription Progress**  
  A dynamic progress system:  
  - The percentage increases linearly up to 90%.  
  - After that, it gradually rises to 99% while the transcription finalizes.  
  - It only displays 100% once the transcription is complete.  

- **Automatic Clipboard Copy**  
  Once transcription is finished, the text is automatically copied to your clipboard.  

---

### **Installation Guide (Step-by-Step)**  

#### **1. Install Python 3.10** *(Mandatory)*  
The application requires **Python 3.10** to run properly.  

1. **Download Python 3.10**  
   - Go to the official Python website:  
     [https://www.python.org/downloads/release/python-31010/](https://www.python.org/downloads/release/python-31010/)  
   - Download and install Python **3.10.10** (64-bit version is recommended).  
   - During installation, **check the box** that says "Add Python to PATH".  

2. **Verify Python installation**  
   Open **Command Prompt (CMD)** or **PowerShell** and run:  

   ```
   python --version
   ```

   Expected output:  

   ```
   Python 3.10.10
   ```

---

#### **2. Install Git (Optional but Recommended)**  
Git is recommended for downloading and managing the project repository.  

1. Download Git from:  
   [https://git-scm.com/downloads](https://git-scm.com/downloads)  
2. Install it with **default settings**.  
3. Verify installation by running in CMD:  

   ```
   git --version
   ```

---

#### **3. Download the Whisper Local App**  

##### **Option 1: Clone via Git (Recommended)**
If Git is installed, clone the repository by running:  

```
git clone https://github.com/yourusername/Whisper-Local-App.git
```

Then, navigate to the project folder:  

```
cd Whisper-Local-App
```

##### **Option 2: Download Manually**  
1. Go to the repository on GitHub.  
2. Click the **"Code"** button and select **"Download ZIP"**.  
3. Extract the ZIP file to a convenient location.  
4. Open **Command Prompt** and navigate to the extracted folder using:  

   ```
   cd path\to\Whisper-Local-App
   ```

---

#### **4. Install Required Dependencies**  

Run the following command to install all necessary dependencies:  

```
pip install -r requirements.txt
```

**If you encounter errors**, manually install each package:  

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sounddevice numpy whisper wave
```

---

#### **5. Install FFmpeg (Required for Whisper)**  

FFmpeg is required for audio processing. Install it via Chocolatey (Windows):  

1. Open **PowerShell as Administrator**  
2. Run:  

   ```
   choco install ffmpeg -y
   ```

3. After installation, verify with:  

   ```
   ffmpeg -version
   ```

---

### **How to Run the Application**  

#### **Run with CMD (Default)**  
To start the application, open **Command Prompt** and navigate to the project folder:  

```
cd path\to\Whisper-Local-App
python whisper_tkinter.py
```

#### **Run without CMD Window (Recommended for GUI Use)**  
To hide the terminal window, use:  

```
pythonw whisper_tkinter.py
```

This prevents the console from appearing while running the application.

---

### **How to Use the Application**  

1. Click the **"🎤 Start Recording"** button to begin recording.  
2. To **pause recording**, click **"⏸ Pause"**.  
   - Click **"▶ Resume"** to continue recording.  
3. Click **"⏹ Stop Recording"** to end the recording and start transcription.  
4. The **transcription progress** will be displayed.  
5. Once transcription is complete, the text is **automatically copied** to your clipboard.  

---

### **License**  

This project is licensed under the MIT License. See the LICENSE file for details.
