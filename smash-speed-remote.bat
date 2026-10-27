@echo off
rem smash-speed: start the app so that your phone / other PCs can use it through Tailscale.
rem Only devices on your own Tailscale network can connect, and they need the passphrase.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo [smash-speed] .venv was not found. Please follow the setup steps in README.md first.
  pause
  exit /b 1
)
if not exist "outputs\remote_access.json" (
  echo [smash-speed] First, choose a passphrase for logging in from other devices.
  ".venv\Scripts\python.exe" -m smash_speed.webapp --set-passphrase
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
start "" ".venv\Scripts\pythonw.exe" -m smash_speed.webapp --remote --no-browser
echo [smash-speed] Started. On this PC open http://127.0.0.1:8765/
where tailscale >nul 2>nul
if %errorlevel%==0 (
  for /f "usebackq tokens=*" %%i in (`tailscale ip -4`) do echo [smash-speed] On your phone / other PC open http://%%i:8765/
) else (
  echo [smash-speed] Tailscale was not found. Install it to use the app from your phone / other PC.
)
echo [smash-speed] To stop it, use the quit button at the bottom left of the app on this PC.
pause
