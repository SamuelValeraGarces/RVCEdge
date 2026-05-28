@echo off
set SD_ENABLE_ASIO=1
set PYTHONPATH=%~dp0

set PYTHON=E:\ComfyUI_windows_portable_nvidia_cu128\ComfyUI_windows_portable\python_embeded\python.exe

if not exist "%PYTHON%" (
    echo ComfyUI Python not found at expected path.
    echo Edit run.bat to point to your Python.
    pause
    exit /b 1
)

echo Starting RVCEdge...
"%PYTHON%" ui/app.py

if errorlevel 1 (
    echo.
    echo [ERROR] RVCEdge crashed. Check output above.
    pause
)
