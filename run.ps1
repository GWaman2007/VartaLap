$env:PYTHONIOENCODING = "utf-8"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = "$scriptDir\src"

if (Test-Path "$scriptDir\.venv\Scripts\python.exe") {
    & "$scriptDir\.venv\Scripts\python.exe" -m vartalap.main @args
} else {
    uv run python -m vartalap.main @args
}
