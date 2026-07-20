$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
& ".\.venv\Scripts\python.exe" "demo\app.py" @args
