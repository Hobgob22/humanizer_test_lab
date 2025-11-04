@echo off
REM dev.bat - Start development environment with hot reload
REM Starts both API server and Streamlit UI

echo ========================================
echo Humanizer Test-Bench Development Mode
echo ========================================
echo.
echo Starting API server on http://localhost:8000
echo Starting Streamlit UI on http://localhost:8501
echo.
echo Both servers have hot reload enabled
echo Press Ctrl+C to stop
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please create it first: python -m venv .venv
    pause
    exit /b 1
)

REM Start API server in background
start "API Server" cmd /k ".venv\Scripts\python.exe start_api.py"

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start Streamlit UI
echo Starting Streamlit UI...
.venv\Scripts\python.exe -m streamlit run src/ui.py --server.runOnSave true --server.fileWatcherType auto
