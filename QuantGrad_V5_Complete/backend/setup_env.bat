@echo off
REM setup_env.bat — One-time environment setup for QuantGrad v4 on Windows
REM Run: setup_env.bat

echo ===================================================
echo  QuantGrad v4 ^- One-Time Environment Setup
echo ===================================================

IF NOT EXIST venv (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) ELSE (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
pip install --upgrade pip --quiet

echo [4/4] Installing packages...
pip install -r requirements.txt

echo.
echo ===================================================
echo  Setup complete!
echo  To activate each session: venv\Scripts\activate.bat
echo.
echo  Then run:
echo    python verify.py
echo    python macro_fetcher.py
echo    python trainer_v3.py --quick
echo    python server.py
echo  Open: http://localhost:8000
echo ===================================================
