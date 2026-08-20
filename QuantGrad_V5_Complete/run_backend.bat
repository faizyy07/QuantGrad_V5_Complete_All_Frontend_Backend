@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)

"%PYTHON%" -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
  echo Python dependencies are not installed. Run setup_windows.bat first.
  exit /b 1
)

cd backend
echo Starting QuantGrad Python model API at http://127.0.0.1:8000 ...
"%PYTHON%" -m uvicorn server:app --host 127.0.0.1 --port 8000
