@echo off
SETLOCAL

REM Define the path to your Python executable
SET PYTHON_EXE="%~dp0venv\Scripts\python.exe"

ECHO ================================================================
ECHO  AI-PRS Analysis Tool (Education Edition) - Starting...
ECHO ================================================================
ECHO.

REM Run the Streamlit app for the Education Edition
%PYTHON_EXE% -m streamlit run "%~dp0src\app_edu.py"

ENDLOCAL
pause