@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

rem Fast path: once the environment has been verified, never run pip/checks again.
if exist ".runtime_ok" if exist "%PY%" goto start_app

if not exist "%PY%" (
    echo Creating Python 3.10 virtual environment...
    py -3.10 -m venv .venv
    if errorlevel 1 goto setup_error
)

set "PY=.venv\Scripts\python.exe"

call :ensure_pytorch
if errorlevel 1 goto setup_error

call :ensure_runtime
if errorlevel 1 goto setup_error

"%PY%" check_env.py >nul 2>&1
if errorlevel 1 goto setup_error

echo Environment verified successfully.
type nul > .runtime_ok

echo.

:start_app
echo Starting ReverseTagger...
echo Browser will open automatically.
echo.
"%PY%" app.py
if errorlevel 1 (
    echo.
    echo ReverseTagger stopped with code %errorlevel%.
    pause
)
exit /b %errorlevel%

:ensure_pytorch
"%PY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    echo PyTorch CUDA environment is already working.
    exit /b 0
)

echo PyTorch CUDA is unavailable. Installing the official PyTorch 2.14 CUDA 13.0 build...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install --upgrade --force-reinstall "torch==2.14.0" "torchvision==0.29.0" --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 exit /b 1

echo Verifying CUDA...
"%PY%" -c "import torch; print('PyTorch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
exit /b %errorlevel%

:ensure_runtime
"%PY%" check_env.py >nul 2>&1
if not errorlevel 1 exit /b 0

echo Checking/repairing ReverseTagger dependencies (first run only)...
"%PY%" -m pip install --upgrade -r requirements.txt
if errorlevel 1 exit /b 1
exit /b 0

:setup_error
echo.
echo Failed to repair/install the pinned environment.
echo If this is unexpected, delete .venv and .runtime_ok, then run run.bat again.
pause
exit /b 1
