@echo off
REM Disaster Recovery Operations for Humanizer Test Bench

if "%1"=="" (
    echo.
    echo 🚨 Disaster Recovery Operations
    echo.
    echo Usage:
    echo   disaster_recovery.bat export         - Export all runs to JSON
    echo   disaster_recovery.bat list-backups   - List available backups
    echo   disaster_recovery.bat restore ^<file^> - Restore from backup
    echo   disaster_recovery.bat health-check   - Check system health
    echo.
    goto :eof
)

if "%1"=="export" (
    echo 📦 Exporting all runs...
    python scripts/disaster_recovery.py export
    goto :eof
)

if "%1"=="list-backups" (
    echo 📋 Listing available backups...
    python scripts/disaster_recovery.py list-backups
    goto :eof
)

if "%1"=="restore" (
    if "%2"=="" (
        echo ❌ Missing backup filename
        echo Usage: disaster_recovery.bat restore ^<backup_filename^>
        goto :eof
    )
    echo 🔄 Restoring from backup: %2
    python scripts/disaster_recovery.py restore %2
    goto :eof
)

if "%1"=="health-check" (
    echo 🔍 Running system health check...
    python scripts/disaster_recovery.py health-check
    goto :eof
)

echo ❌ Unknown command: %1
echo.
echo Available commands: export, list-backups, restore, health-check 