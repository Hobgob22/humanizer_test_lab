@echo off
REM dev.bat - Start development server with hot reload
echo Starting Humanizer Test-Bench in development mode...
echo.
echo Server will be available at http://localhost:8501
echo Press Ctrl+C to stop
echo.
docker-compose --profile dev up