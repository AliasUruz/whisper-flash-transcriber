# Whisper Flash Transcriber - Diagnostics
# ---------------------------------------
# Usage:
#   pwsh -File scripts/diagnostics.ps1 [-UseVenv] [-SkipLogTail] [-LogTail N]
#                                      [-SkipPythonImports] [-NoColor] [-NoPause]

param(
    [switch]$UseVenv,
    [switch]$SkipLogTail,
    [int]$LogTail = 80,
    [switch]$SkipPythonImports,
    [switch]$NoColor,
    [switch]$NoPause,
    [string]$ReportPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Issues = New-Object System.Collections.Generic.List[pscustomobject]

function Add-Issue {
    param(
        [string]$Category,
        [string]$Message,
        [string]$Recommendation = ""
    )
    $Issues.Add([pscustomobject]@{
        Category       = $Category
        Message        = $Message
        Recommendation = $Recommendation
    }) | Out-Null
}

function Write-Section {
    param([string]$Title)
    $banner = "=== $Title ==="
    if ($NoColor) { Write-Host "`n$banner" }
    else { Write-Host "`n$banner" -ForegroundColor Cyan }
    if ($script:TranscriptActive) { return }
}

function Write-Status {
    param(
        [string]$Label,
        [bool]$Ok,
        [string]$Message,
        [string]$Category = "",
        [string]$Recommendation = ""
    )
    $prefix = if ($Ok) { "[OK]" } else { "[WARN]" }
    if ($NoColor) { Write-Host ("{0} {1}: {2}" -f $prefix, $Label, $Message) }
    else {
        $color = if ($Ok) { 'Green' } else { 'Yellow' }
        Write-Host ("{0} {1}: {2}" -f $prefix, $Label, $Message) -ForegroundColor $color
    }
    if (-not $Ok -and $Category) {
        Add-Issue -Category $Category -Message ("{0}: {1}" -f $Label, $Message) -Recommendation $Recommendation
    }
}

function Write-Detail {
    param([string]$Message)
    if ($NoColor) { Write-Host "  - $Message" }
    else { Write-Host "  - $Message" -ForegroundColor DarkGray }
}

$script:TranscriptActive = $false
if ($ReportPath) {
    try {
        Start-Transcript -Path (Resolve-Path -Path $ReportPath) -Force -ErrorAction Stop | Out-Null
        $script:TranscriptActive = $true
    }
    catch {
        Write-Status -Label "Transcript" -Ok $false -Message $_.Exception.Message -Category "Reporting" -Recommendation "Verify the target path or run without -ReportPath."
    }
}

$scriptPath = $MyInvocation.MyCommand.Path
if (-not $scriptPath -and $PSScriptRoot) {
    $scriptPath = Join-Path $PSScriptRoot "diagnostics.ps1"
}
if (-not $scriptPath) {
    throw "Unable to determine diagnostics.ps1 location. Run with 'pwsh -File scripts/diagnostics.ps1'."
}

$repoRoot = Split-Path -Path (Split-Path -Path $scriptPath -Parent) -Parent
Set-Location -LiteralPath $repoRoot

# ---------------------------------------------------------------------------
# Operating system
# ---------------------------------------------------------------------------
Write-Section "Operating System"
try {
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Detail ("Edition : {0}" -f $os.Caption)
    Write-Detail ("Version : {0}" -f $os.Version)
    Write-Detail ("Build   : {0}" -f $os.BuildNumber)
    $freeMb = [math]::Round($os.FreePhysicalMemory / 1024, 0)
    $totalMb = [math]::Round($os.TotalVisibleMemorySize / 1024, 0)
    Write-Detail ("RAM free: {0} MB of {1} MB" -f $freeMb, $totalMb)
}
catch {
    Write-Status -Label "Win32_OperatingSystem" -Ok $false -Message $_.Exception.Message -Category "System" -Recommendation "Run PowerShell as administrator and verify WMI."
}

# Check elevation
$isElevated = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$elevationMessage = if ($isElevated) { "Yes" } else { "No" }
Write-Status -Label "PowerShell elevated" -Ok $isElevated -Message $elevationMessage -Category "Permissions" -Recommendation "Some diagnostics (e.g. device enumeration) may require elevated PowerShell."

# Disk usage for repo drive
$repoDrive = ($repoRoot[0])
try {
    $driveInfo = Get-PSDrive -Name $repoDrive -ErrorAction Stop
    $freeGb = [math]::Round($driveInfo.Free/1GB, 2)
    $totalGb = [math]::Round($driveInfo.Used/1GB + $freeGb, 2)
    Write-Detail ("Disk {0}: {1} GB free of {2} GB" -f $repoDrive, $freeGb, $totalGb)
    if ($freeGb -lt 2) {
        Write-Status -Label "Disk space" -Ok $false -Message ("Drive {0}: only {1} GB free" -f $repoDrive, $freeGb) -Category "Storage" -Recommendation "Free up disk space before downloading large models."
    }
}
catch {
    Write-Status -Label "Disk info" -Ok $false -Message $_.Exception.Message -Category "System" -Recommendation "Ensure the drive is accessible."
}

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Python discovery
# ---------------------------------------------------------------------------
Write-Section "Python"
function Resolve-Python {
    param([switch]$PreferVenv)
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if ($PreferVenv -and (Test-Path $venvPython)) { return $venvPython }
    if (Test-Path $venvPython) { return $venvPython }
    try { return (Get-Command python -ErrorAction Stop).Source } catch { return $null }
}

$pythonExe = Resolve-Python -PreferVenv:$UseVenv
if (-not $pythonExe) {
    Write-Status -Label "Python" -Ok $false -Message "Executable not found." -Category "Environment" -Recommendation "Install Python 3.11+ or create .venv before starting the app."
    return
}
Write-Status -Label "Python" -Ok $true -Message $pythonExe
try {
    $pyVersion = (& $pythonExe --version 2>&1).Trim()
    Write-Detail ("Version : {0}" -f $pyVersion)
}
catch {
    Write-Status -Label "Version" -Ok $false -Message $_.Exception.Message -Category "Environment" -Recommendation "Run '$pythonExe --version' manually to inspect the error."
}
try {
    $pyArch = (& $pythonExe -c "import platform; print(platform.architecture()[0])" 2>&1).Trim()
    if ($pyArch) { Write-Detail ("Architecture : {0}" -f $pyArch) }
}
catch {
    Write-Status -Label "Architecture" -Ok $false -Message $_.Exception.Message -Category "Environment" -Recommendation "Verify the Python installation."
}

# ---------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------
Write-Section "Virtual Environment"
$venvPath = Join-Path $repoRoot ".venv"
$venvExists = Test-Path $venvPath
Write-Status -Label ".venv folder" -Ok $venvExists -Message $venvPath -Category "Environment" -Recommendation "Create with 'python -m venv .venv' to isolate dependencies."
if ($venvExists) {
    $venvPython = Join-Path $venvPath "Scripts\python.exe"
    Write-Status -Label "Python in .venv" -Ok (Test-Path $venvPython) -Message $venvPython -Category "Environment" -Recommendation "Recreate the virtualenv if the interpreter is missing."
}
$venvActive = [bool]$env:VIRTUAL_ENV
$venvMessage = if ($env:VIRTUAL_ENV) { $env:VIRTUAL_ENV } else { "<inactive>" }
Write-Status -Label "Virtualenv active" -Ok $venvActive -Message $venvMessage -Category "Environment" -Recommendation "Activate with '.\.venv\Scripts\activate' before running the app."

$pipVersion = $null
Write-Section "pip"
try {
    $pipVersion = (& $pythonExe -m pip --version 2>&1).Trim()
    Write-Status -Label "pip --version" -Ok $true -Message $pipVersion
}
catch {
    Write-Status -Label "pip --version" -Ok $false -Message $_.Exception.Message -Category "Dependencies" -Recommendation "Install pip for this interpreter or repair the Python installation."
}

# ---------------------------------------------------------------------------
# Python modules
# ---------------------------------------------------------------------------
Write-Section "Python Modules"
if ($SkipPythonImports) {
    Write-Detail "Module probe skipped (--SkipPythonImports)."
}
else {
    $moduleProbe = @"
import json
import importlib
modules = ["sounddevice", "keyboard", "ctranslate2"]
result = {}
for name in modules:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        result[name] = {"ok": False, "error": str(exc)}
    else:
        result[name] = {"ok": True, "version": getattr(mod, "__version__", "unknown")}
print(json.dumps(result))
"@
    $probeFile = [System.IO.Path]::GetTempFileName()
    Set-Content -LiteralPath $probeFile -Value $moduleProbe -Encoding UTF8
    $exitCode = 0
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $output = & $pythonExe $probeFile 2>&1
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $prev
        $serialized = ($output -join "`n").Trim()
        $jsonLine = ($serialized -split "`n") | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1
        if ($jsonLine) {
            $parsed = $jsonLine | ConvertFrom-Json -ErrorAction Stop
            foreach ($entry in $parsed.PSObject.Properties) {
                $name = $entry.Name
                $payload = $entry.Value
                if ($payload.ok) {
                    Write-Status -Label $name -Ok $true -Message ("version {0}" -f $payload.version)
                }
                else {
                    Write-Status -Label $name -Ok $false -Message $payload.error -Category "Dependencies" -Recommendation ("Reinstall with 'pip install {0}'" -f $name)
                }
            }
        }
        else {
            Write-Status -Label "Import probe" -Ok $false -Message "Python did not emit JSON." -Category "Dependencies" -Recommendation "Run the probe manually to inspect the traceback."
        }
    }
    catch {
        Write-Status -Label "Import probe" -Ok $false -Message $_.Exception.Message -Category "Dependencies" -Recommendation "Review the traceback above and reinstall packages inside the virtualenv."
    }
    finally {
        Remove-Item -LiteralPath $probeFile -ErrorAction SilentlyContinue
        if ($exitCode -ne 0) {
            Write-Status -Label "Import probe exit code" -Ok $false -Message $exitCode -Category "Dependencies" -Recommendation "Importing critical modules failed. Reinstall dependencies in the active environment."
        }
    }
}

# ---------------------------------------------------------------------------
# Persistence / profile
# ---------------------------------------------------------------------------
Write-Section "Profile & Storage"
$profileRoot = Join-Path $env:USERPROFILE ".cache\whisper_flash_transcriber"
$pathsToCheck = @(
    @{ Label = "Profile root"; Path = $profileRoot; Category = "Storage"; Rec = "Ensure the user has write permission." },
    @{ Label = "config.json"; Path = Join-Path $profileRoot "config.json"; Category = "Config"; Rec = "Launch the application once to generate defaults." },
    @{ Label = "secrets.json"; Path = Join-Path $profileRoot "secrets.json"; Category = "Config"; Rec = "Created automatically when saving API keys. Create an empty file to silence warnings." },
    @{ Label = "hotkey_config.json"; Path = Join-Path $profileRoot "hotkey_config.json"; Category = "Config"; Rec = "Allow bootstrap to regenerate or copy the template hotkey file." },
    @{ Label = "models dir"; Path = Join-Path $profileRoot "models"; Category = "Storage"; Rec = "Ensure this directory exists or adjust storage paths in settings." },
    @{ Label = "recordings dir"; Path = Join-Path $profileRoot "recordings"; Category = "Storage"; Rec = "Confirm write access for temporary audio files." }
)
foreach ($item in $pathsToCheck) {
    $exists = Test-Path -LiteralPath $item.Path
    Write-Status -Label $item.Label -Ok $exists -Message $item.Path -Category $item.Category -Recommendation $item.Rec
}

# ---------------------------------------------------------------------------
# Config inspection
# ---------------------------------------------------------------------------
Write-Section "Config Inspection"
$configPath = Join-Path $profileRoot "config.json"
$configJson = $null
if (Test-Path -LiteralPath $configPath) {
    try {
        $configInfo = Get-Item -LiteralPath $configPath
        Write-Detail ("config.json size : {0} bytes" -f $configInfo.Length)
        Write-Detail ("config.json mtime: {0}" -f $configInfo.LastWriteTime)
        $configJson = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
        Write-Status -Label "config.json" -Ok $true -Message "Valid JSON"
    }
    catch {
        Write-Status -Label "config.json" -Ok $false -Message $_.Exception.Message -Category "Config" -Recommendation "Fix the JSON or delete the file to regenerate defaults."
    }
}
else {
    Write-Status -Label "config.json" -Ok $false -Message "File not found" -Category "Config" -Recommendation "Run the application once to materialize defaults."
}

if ($configJson) {
    $threads = $configJson.advanced.performance.asr_ct2_cpu_threads
    if ($null -eq $threads) {
        Write-Status -Label "asr_ct2_cpu_threads" -Ok $false -Message "Value is null" -Category "Config" -Recommendation "Set an explicit number (e.g. 0) in advanced performance settings."
    }
    $modelId = $configJson.asr_model_id
    if (-not $modelId) {
        Write-Status -Label "asr_model_id" -Ok $false -Message "No model selected" -Category "Config" -Recommendation "Choose a model in the UI before transcribing."
    }
    $recordMode = $configJson.record_mode
    if ($recordMode -and $recordMode -notin @("toggle", "press")) {
        Write-Status -Label "record_mode" -Ok $false -Message ("Unexpected value: {0}" -f $recordMode) -Category "Config" -Recommendation "Use 'toggle' or 'press' for the recording hotkey mode."
    }
}

# ---------------------------------------------------------------------------
# Audio devices
# ---------------------------------------------------------------------------
Write-Section "Audio Input Devices"
try {
    $audioDevices = Get-PnpDevice -Class AudioEndpoint -PresentOnly | Where-Object { $_.FriendlyName -match "Microphone|Mic|Input" }
    if ($audioDevices) {
        foreach ($device in $audioDevices) {
            Write-Status -Label $device.FriendlyName -Ok $true -Message $device.InstanceId
        }
    }
    else {
        Write-Status -Label "AudioEndpoint" -Ok $false -Message "No microphone detected." -Category "Audio" -Recommendation "Connect a microphone or enable access in Windows privacy settings."
    }
}
catch {
    Write-Status -Label "Device enumeration" -Ok $false -Message $_.Exception.Message -Category "Audio" -Recommendation "Run PowerShell as administrator or check audio drivers."
}

# ---------------------------------------------------------------------------
# Log analysis
# ---------------------------------------------------------------------------
$recentLogErrors = @()
if (-not $SkipLogTail) {
    Write-Section "Recent Log Entries"
    $logPath = Join-Path $repoRoot "logs\whisper-flash-transcriber.log"
    if (Test-Path -LiteralPath $logPath) {
        Write-Detail ("File: {0}" -f $logPath)
        try {
            $logTailContent = Get-Content -LiteralPath $logPath -Tail $LogTail
            $logTailContent | ForEach-Object { Write-Host $_ }
            $recentLogErrors = $logTailContent | Where-Object { $_ -match "ERROR|CRITICAL|MemoryError|bootstrap\.failure|bootstrap\.step" }
        }
        catch {
            Write-Status -Label "Log tail" -Ok $false -Message $_.Exception.Message -Category "Logs" -Recommendation "Check file permissions or whether the log is locked."
        }
    }
    else {
        Write-Status -Label "Log file" -Ok $false -Message "No log found at $logPath" -Category "Logs" -Recommendation "Run the application to generate logs or verify the logs directory."
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Section "Summary"
if ($Issues.Count -gt 0) {
    if ($NoColor) { Write-Host "Potential issues detected:" }
    else { Write-Host "Potential issues detected:" -ForegroundColor Magenta }
    $Issues | Sort-Object Category | Format-Table Category, Message, Recommendation -AutoSize
}
else {
    Write-Host "No critical issues detected. Environment looks ready."
}

if ($recentLogErrors.Count -gt 0) {
    Write-Host ""
    if ($NoColor) { Write-Host "Recent log errors:" }
    else { Write-Host "Recent log errors:" -ForegroundColor Magenta }
    $recentLogErrors | ForEach-Object { Write-Host $_ }
    Add-Issue -Category "Logs" -Message "Errors detected in recent log entries" -Recommendation "Inspect logs/whisper-flash-transcriber.log for details."
}

Write-Host ""
Write-Host "Review the findings above. Attach this output when asking for support."

if ($script:TranscriptActive) {
    try { Stop-Transcript | Out-Null }
    catch {
        Write-Status -Label "Transcript" -Ok $false -Message $_.Exception.Message -Category "Reporting" -Recommendation "Check write permissions for the chosen report path."
    }
}

if (-not $NoPause) {
    Write-Host ""
    Write-Host "Press Enter to exit..."
    [void](Read-Host)
}

