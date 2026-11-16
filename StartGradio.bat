@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0StartGradio.ps1" %*
set "launcher_exit=%errorlevel%"
if not "%launcher_exit%"=="0" (
    echo.
    echo CXR Evidence could not start. See the error above.
    pause
)
exit /b %launcher_exit%
