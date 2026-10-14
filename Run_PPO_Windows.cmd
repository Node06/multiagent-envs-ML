@echo off
setlocal
pushd "%~dp0"
set "TEAM_PY=python"
if exist ".venv\Scripts\python.exe" set "TEAM_PY=%CD%\.venv\Scripts\python.exe"
"%TEAM_PY%" team_run.py ppo --show
popd
pause
