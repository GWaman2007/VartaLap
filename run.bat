@echo off
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0src
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m vartalap.main %*
) else (
    uv run python -m vartalap.main %*
)
