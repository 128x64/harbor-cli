@echo off
rem The whole command is one parenthesised block, so cmd reads it completely before Python starts.
rem That keeps the self-updater free to replace this file while Harbor CLI is running.
setlocal EnableDelayedExpansion
(
  "%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
  exit /b !errorlevel!
)
