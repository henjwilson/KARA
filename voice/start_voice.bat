@echo off
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m voice %*
) else (
  py -3.10 -m voice %*
)
pause
