# One-command local launch on Windows:  .\run.ps1
# Creates/reuses a .venv, installs deps once, then serves the dashboard and opens it.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv (first run)..." -ForegroundColor Cyan
    py -3.12 -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install --upgrade pip
    & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

$env:MUSIC_OPEN_BROWSER = "1"
Write-Host "Starting Music DNA Analyzer at http://127.0.0.1:8000 ..." -ForegroundColor Green
& .\.venv\Scripts\python.exe -m app
