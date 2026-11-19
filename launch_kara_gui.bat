@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else (
    set "PYTHON=py -3"
)
%PYTHON% "launch_kara_gui.py"
if errorlevel 1 (
    echo.
    echo Could not launch KARA Qt. Run install_kara_gui.bat to install PySide6.
    echo Also check that the required PAROL6 dependencies are installed.
    pause
)
