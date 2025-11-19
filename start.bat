@echo off
REM Whisper Flash Transcriber Launcher

REM Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
)

REM Run the application
echo Starting Whisper Flash Transcriber...
python src\main.py

REM Keep window open if it crashes
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%
    pause
)
