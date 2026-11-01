@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Setup required. See README.md.
  exit /b 1
)
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
