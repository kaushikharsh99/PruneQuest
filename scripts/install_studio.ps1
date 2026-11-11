# PruneQuest Studio Bootstrap Installer for Windows (PowerShell)
# Sets up an isolated Python environment and installs PruneQuest Studio.

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "🌱 PruneQuest Studio Bootstrap Installer (Windows)" -ForegroundColor Green
Write-Host "=========================================================="

# 1. Find compatible Python
$PythonCmd = $null
foreach ($cmd in @("python", "py")) {
    try {
        $ver = & $cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
        $major = & $cmd -c 'import sys; print(sys.version_info.major)'
        $minor = & $cmd -c 'import sys; print(sys.version_info.minor)'
        if ([int]$major -eq 3 -and [int]$minor -ge 10) {
            $PythonCmd = $cmd
            Write-Host "✔ Found compatible Python $ver ($cmd)" -ForegroundColor Green
            break
        }
    } catch {
        continue
    }
}

if (-not $PythonCmd) {
    Write-Host "❌ Error: Python 3.10+ is required to install PruneQuest Studio." -ForegroundColor Red
    Write-Host "Please install Python 3.10 or higher from https://python.org" -ForegroundColor Red
    exit 1
}

# 2. Create isolated environment
$InstallDir = Join-Path $HOME ".prunequest\studio-env"
Write-Host "Creating isolated virtual environment at $InstallDir..."
& $PythonCmd -m venv $InstallDir

$VenvPython = Join-Path $InstallDir "Scripts\python.exe"

# 3. Upgrade pip and install prunequest[studio]
Write-Host "Installing PruneQuest Studio package..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
if (Test-Path "pyproject.toml") {
    & $VenvPython -m pip install ".[studio]"
} else {
    & $VenvPython -m pip install "prunequest[studio]"
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "✔ PruneQuest Studio installed successfully!" -ForegroundColor Green
Write-Host "=========================================================="
Write-Host ""
Write-Host "To launch PruneQuest Studio, run:"
Write-Host "    $InstallDir\Scripts\prunequest studio"
Write-Host ""
