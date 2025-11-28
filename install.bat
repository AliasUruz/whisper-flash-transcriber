@echo off
title Whisper Flash Transcriber - Installer
cls

echo ========================================================
echo    Whisper Flash Transcriber - Automated Installer
echo ========================================================
echo.

:: 1. Check for Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found!
    echo Please install Python 3.11 or higher from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)
echo Python found.
echo.

:: 2. Create Virtual Environment
echo [2/4] Setting up Virtual Environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

:: 3. Install Dependencies
echo [3/4] Installing Dependencies (This may take a few minutes)...
echo Upgrading pip...
call .venv\Scripts\python.exe -m pip install --upgrade pip >nul
echo Installing requirements from requirements.txt...
call .venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b
)
echo Dependencies installed successfully.
echo.

:: 4. Create Desktop Shortcut
echo [4/4] Creating Desktop Shortcut...
set "TARGET=%~dp0.venv\Scripts\pythonw.exe"
set "ARGS=%~dp0src\main.py"
set "ICON=%~dp0assets\icon.ico"
set "NAME=Whisper Flash"
set "WSCRIPT=%temp%\CreateShortcut.vbs"

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%WSCRIPT%"
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\%NAME%.lnk" >> "%WSCRIPT%"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%WSCRIPT%"
echo oLink.TargetPath = "%TARGET%" >> "%WSCRIPT%"
echo oLink.Arguments = """%ARGS%""" >> "%WSCRIPT%"
echo oLink.WorkingDirectory = "%~dp0" >> "%WSCRIPT%"
echo oLink.IconLocation = "%ICON%" >> "%WSCRIPT%"
echo oLink.Save >> "%WSCRIPT%"

cscript /nologo "%WSCRIPT%"
del "%WSCRIPT%"
echo Shortcut created on Desktop.
echo.

echo ========================================================
echo    Installation Complete! 🚀
echo ========================================================
echo.
echo You can now start the app using the "Whisper Flash" shortcut on your Desktop.
echo.
pause
