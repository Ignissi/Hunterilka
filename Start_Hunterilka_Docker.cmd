@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist "runtime\python.exe" goto portable
if exist ".venv\Scripts\python.exe" goto venv
python "scripts\start_docker.py" %*
goto done
:portable
"runtime\python.exe" -u "scripts\start_docker.py" %*
goto done
:venv
".venv\Scripts\python.exe" -u "scripts\start_docker.py" %*
:done
if not errorlevel 1 exit /b 0
pause
exit /b 1
