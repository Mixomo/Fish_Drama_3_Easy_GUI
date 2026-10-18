@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Fish Audio Drama 3 Easy GUI

set "PY_EXE=%CD%\.venv\Scripts\python.exe"
set "DRAMA3_AUDIO_CPP_DIR=%CD%\.runtime\audio.cpp"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "CUDA_MODULE_LOADING=LAZY"
set "HF_HOME=%CD%\.runtime\cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "GRADIO_TEMP_DIR=%CD%\.runtime\gradio-temp"
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
set "PATH=%CD%\.runtime\vulkan\bin;%CD%\.runtime\audio.cpp\build\windows-cuda-release\bin;%PATH%"

if not exist "%PY_EXE%" (
  echo [ERROR] Local environment not found. Run install.bat first.
  pause
  exit /b 1
)

echo.
echo [INFO] Drama 3 inference uses the CUDA/Vulkan audio.cpp runtime or the Vulkan speech.cpp runtime.
echo [INFO] CPU-only synthesis is not available.
echo.

"%PY_EXE%" app.py
if errorlevel 1 pause
