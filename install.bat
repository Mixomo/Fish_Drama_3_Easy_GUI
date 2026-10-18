@echo off
setlocal
cd /d "%~dp0"
title Fish Audio Drama 3 Easy GUI - Installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
  echo.
  echo [ERROR] Installation failed.
  pause
  exit /b 1
)
pause
