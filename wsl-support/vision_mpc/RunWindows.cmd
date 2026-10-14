@echo off
setlocal
pushd "%~dp0..\.."
set "TEAM_PY=python"
if exist ".venv\Scripts\python.exe" set "TEAM_PY=%CD%\.venv\Scripts\python.exe"
"%TEAM_PY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,10) else 1)"
if errorlevel 1 (
  echo Python 3.10 is required. Activate the project Conda environment or create .venv first.
  popd
  pause
  exit /b 1
)
"%TEAM_PY%" wsl-support\vision_mpc\run.py --mode demo --show --video --realtime
popd
pause
