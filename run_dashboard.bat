@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

where python >nul 2>&1
if %errorlevel% neq 0 (
    where py >nul 2>&1
    if %errorlevel% neq 0 (
        echo [error] Python 3 is required but was not found in PATH.
        exit /b 1
    )
    set "PY_EXE=py -3"
) else (
    set "PY_EXE=python"
)

if not exist ".venv" (
    echo [setup] Creating virtual environment (.venv)...
    %PY_EXE% -m venv .venv
    if %errorlevel% neq 0 (
        echo [error] Failed to create virtual environment.
        exit /b 1
    )
)

call ".venv\Scripts\activate"
if %errorlevel% neq 0 (
    echo [error] Unable to activate virtual environment.
    exit /b 1
)

echo [setup] Upgrading pip and installing dashboard dependencies...
python -m pip install --upgrade pip
python -m pip install flask

echo [setup] Installing UNO BNN training requirements...
python -m pip install numpy tqdm pyro-ppl
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo [ready] Launching UNO BNN dashboard at http://127.0.0.1:8000
echo [info] Press CTRL+C to stop the dashboard and training backend.
python dashboard_server.py %*

popd
endlocal
