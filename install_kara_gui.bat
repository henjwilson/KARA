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
%PYTHON% -m pip install -r requirements-kara-gui.txt
if errorlevel 1 (
    echo.
    echo PySide6 installation failed. Check that you have Internet access and Python 3.12+.
    pause
    exit /b 1
)
echo.
echo PySide6 installation complete. Double-click launch_kara_gui.bat.
pause
