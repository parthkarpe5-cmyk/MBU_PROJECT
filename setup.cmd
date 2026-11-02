@echo off
setlocal
cd /d "%~dp0"
python -c "import sys,tkinter; assert sys.version_info >= (3,10), 'Python 3.10+ required'"
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if errorlevel 1 exit /b 1
python -m pip --python .venv install -r requirements.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" setup_model.py
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" setup_expression.py
if errorlevel 1 exit /b 1
echo Setup complete. Run run.cmd to start DriveGuard AI.
