@echo off
REM scripts.bat - Windows convenience script runner

if "%1"=="" goto :help

if "%1"=="dev" goto :dev
if "%1"=="prod" goto :prod
if "%1"=="build" goto :build
if "%1"=="logs" goto :logs
if "%1"=="shell" goto :shell
if "%1"=="stop" goto :stop
if "%1"=="clean" goto :clean
if "%1"=="backup" goto :backup

:help
echo Humanizer Test-Bench - Windows Commands
echo.
echo Usage: scripts.bat [command]
echo.
echo Commands:
echo   dev     - Start development server with hot reload
echo   prod    - Start production server
echo   build   - Build Docker images
echo   logs    - View container logs
echo   shell   - Open shell in container
echo   stop    - Stop all containers
echo   clean   - Stop and remove all containers
echo   backup  - Create backup of data
echo.
goto :eof

:dev
echo Starting development server with hot reload...
docker-compose --profile dev up
goto :eof

:prod
echo Starting production server...
docker-compose --profile prod up -d
echo Server running at http://localhost:8501
goto :eof

:build
echo Building Docker images...
docker-compose build
goto :eof

:logs
echo Showing logs (press Ctrl+C to exit)...
docker-compose logs -f
goto :eof

:shell
echo Opening shell in development container...
docker-compose exec humanizer-dev /bin/bash
goto :eof

:stop
echo Stopping containers...
docker-compose stop
goto :eof

:clean
echo Stopping and removing containers...
docker-compose down -v
goto :eof

:backup
echo Creating backup...
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)
docker-compose exec humanizer-dev tar -czf backups/backup-%mydate%-%mytime%.tar.gz data/ results/ cache/ logs/
echo Backup created!
goto :eof