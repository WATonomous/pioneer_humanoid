@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Configure this in Windows Command Prompt or PowerShell, not only in WSL.
if not defined ISAACLAB_BAT (
    echo Set ISAACLAB_BAT to your native Windows Isaac Lab launcher.
    echo Example: set "ISAACLAB_BAT=C:\path\to\IsaacLab\isaaclab.bat"
    exit /b 2
)

rem Normalize before changing cwd, so relative installation paths also work.
for %%I in ("%ISAACLAB_BAT%") do set "WATO_ISAACLAB_BAT=%%~fI"
if not exist "%WATO_ISAACLAB_BAT%" (
    echo Isaac Lab launcher not found: "%WATO_ISAACLAB_BAT%"
    exit /b 2
)

rem pushd temporarily maps a UNC/WSL checkout to a Windows drive for USD.
pushd "%~dp0..\..\..\..\.."
if errorlevel 1 (
    echo Could not access the repository from Windows.
    exit /b 1
)

set "WATO_REPO=%CD%"
set "WATO_VIEWER=%WATO_REPO%\src\simulation\humanoid_rl\humanoid_rl\scripts\play_torchscript.py"
if not exist "%WATO_VIEWER%" (
    echo Viewer script not found: "%WATO_VIEWER%"
    popd
    exit /b 2
)

set "PYTHONPATH=%WATO_REPO%\src\simulation\humanoid_rl;%WATO_REPO%\src\simulation\humanoid_rl_tasks;%WATO_REPO%\src\pioneer_humanoid;%PYTHONPATH%"
set "PYTHONUNBUFFERED=1"

rem All arguments belong to the viewer; relative paths are repository-relative.
call "%WATO_ISAACLAB_BAT%" -p -u "%WATO_VIEWER%" %*
set "WATO_RESULT=%ERRORLEVEL%"
if not "%WATO_RESULT%"=="0" echo Viewer exited with error code %WATO_RESULT%.
popd
rem Batch termination also ends setlocal and restores the caller's environment.
exit /b %WATO_RESULT%
