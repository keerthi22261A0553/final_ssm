@echo off
rem Continue the experiment after a shutdown. Finished distances are
rem skipped; a half-trained distance resumes after its last saved epoch.
cd /d "%~dp0"
set PY=C:\Users\rds-l002\Lightweight_UIB_SSM\venv\Scripts\python.exe

"%PY%" -u scripts\run_all.py
if errorlevel 1 (
    echo run_all.py stopped with an error - see the messages above.
    pause
    exit /b 1
)

"%PY%" -u scripts\run_optional.py
pause
