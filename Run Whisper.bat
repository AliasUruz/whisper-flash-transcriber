@echo off
setlocal enableextensions
echo Whisper Flash Transcriber launcher

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_CMD=python"
set "HEADLESS_FLAG="

if /I "%~1"=="--headless" (
    set "HEADLESS_FLAG=--headless"
    shift
) else if /I "%~1"=="headless" (
    set "HEADLESS_FLAG=--headless"
    shift
) else if /I "%~1"=="/headless" (
    set "HEADLESS_FLAG=--headless"
    shift
)

%PYTHON_CMD% src\main.py %HEADLESS_FLAG% %*
endlocal
