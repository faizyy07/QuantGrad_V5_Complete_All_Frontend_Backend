@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run setup_windows.bat first.
  exit /b 1
)

cd backend
echo Starting QuantGrad quick training. New artifacts will be saved automatically in backend\artifacts\.
echo This uses BTCUSDT hourly data, 1,500 bars, and the trainer's quick 5-epoch setting.
"%~dp0.venv\Scripts\python.exe" trainer_v3.py --quick --bars 1500
if errorlevel 1 (
  echo.
  echo Training did not finish. Read the error above; no incomplete model should be used.
  exit /b 1
)

echo.
echo Training completed. Start run_backend.bat, then run_frontend.bat, and open http://localhost:3000.
