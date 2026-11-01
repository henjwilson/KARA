@echo off
setlocal
cd /d "%~dp0.."
echo WARNING: REAL PAROL6 ROBOT MOTION ENABLED.
echo Verify the workspace is clear, calibration is correct, and E-STOP is accessible.
echo Every motion command still requires a specific second spoken confirmation.
choice /M "Enable real robot motion"
if errorlevel 2 exit /b 1
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m voice --enable-motion %*
) else (
  py -3.10 -m voice --enable-motion %*
)
pause
