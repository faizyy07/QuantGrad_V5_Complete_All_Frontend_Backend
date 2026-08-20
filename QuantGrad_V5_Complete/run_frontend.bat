@echo off
setlocal
cd /d "%~dp0frontend"
echo Installing or verifying V5 frontend dependencies...
npx pnpm@10.4.1 install --frozen-lockfile
if errorlevel 1 (
  echo.
  echo Frontend dependency installation failed. Check the error above.
  exit /b 1
)

echo.
echo Starting QuantGrad V5 at http://localhost:3000 ...
npx pnpm@10.4.1 dev
