@echo off
rem smash-speed GUI launcher (double-click this file)
rem Starts a local server on this PC and opens the app in your web browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo [smash-speed] .venv was not found. Please follow the setup steps in README.md first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m smash_speed.webapp
