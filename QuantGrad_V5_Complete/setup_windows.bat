@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo Python was not found on PATH. Install Python 3.12+ from python.org or the Microsoft Store, then reopen VS Code.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

echo Upgrading pip and installing QuantGrad backend requirements...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r ".\backend\requirements.txt"
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Run run_backend.bat in one terminal and run_frontend.bat in another.
