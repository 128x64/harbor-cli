@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
echo [1/5] Creating virtual environment...
%PY% -m venv .venv || goto :err
call ".venv\Scripts\activate.bat"
echo [2/5] Installing requirements...
python -m pip install --upgrade pip || goto :err
python -m pip install -r requirements.txt || goto :err
echo [3/5] Installing Harbor CLI...
python -m pip install -e . || goto :err
echo [4/5] Optional: run model.gguf in-process (skip if you use llama-server.exe)...
python -m pip install -r requirements-local.txt || echo   (llama-cpp-python skipped - use llama-server.exe instead)
echo [5/5] Optional extras (browser tools, OCR)...
python -m pip install -r requirements-extra.txt || echo   (optional extras skipped)
echo.
echo Done.
echo   NVIDIA NIM : setx NVIDIA_API_KEY "nvapi-..."  (free key: https://build.nvidia.com), then open a NEW terminal
echo   Local model: put model.gguf next to main.py (plus llama-server.exe from github.com/ggml-org/llama.cpp/releases)
echo Harbor CLI checks GitHub for updates when it starts (turn off: set "auto_update": "off" in %%USERPROFILE%%\.harbor\config.json).
echo Run from your project folder:  "%~dp0harbor.bat"
exit /b 0
:err
echo Install failed - see the messages above.
exit /b 1
