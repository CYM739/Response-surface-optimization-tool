@echo off
SETLOCAL

REM ================================================================
REM  AI-PRS Analysis Tool — Launcher
REM ================================================================
REM
REM  Uses system Python instead of a venv so that heavy ML packages
REM  (torch ~2.5 GB, torch_geometric, rdkit) already installed
REM  system-wide are available without re-downloading into the venv.
REM
REM  To switch back to a venv: change PYTHON_EXE to point to
REM  venv\Scripts\python.exe and install the ML packages there:
REM    venv\Scripts\python.exe -m pip install torch torchvision torch_geometric rdkit
REM    venv\Scripts\python.exe -m pip install -r requirements.txt
REM ================================================================

REM --- SYSTEM Python (avoids ~3 GB venv download) -----------------
SET "PYTHON_EXE=C:\Users\Maxma\AppData\Local\Programs\Python\Python310\python.exe"

REM Verify Python is reachable
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found at %PYTHON_EXE%
    pause
    exit /b 1
)

ECHO ================================================================
ECHO  AI-PRS Analysis Tool
ECHO ================================================================
ECHO.

ECHO Installing / updating required packages...
%PYTHON_EXE% -m pip install -q -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] pip install failed.
    pause
    exit /b 1
)

ECHO.
ECHO Starting Application...
ECHO ================================================================

%PYTHON_EXE% -m streamlit run "%~dp0src\app.py"

ENDLOCAL
pause
