#!/usr/bin/env python3
"""
Disaster Recovery Operations for Humanizer Test Bench

Usage:
    python scripts/disaster_recovery.py export
    python scripts/disaster_recovery.py list-backups
    python scripts/disaster_recovery.py restore <backup_filename>
    python scripts/disaster_recovery.py health-check
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.results_db import export_all_runs, list_json_backups, restore_from_json_backup, list_runs
except ImportError:
    # Try alternative import paths
    try:
        from src.results_db import export_all_runs, list_json_backups, restore_from_json_backup, list_runs
    except ImportError:
        # If still failing, add current directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.results_db import export_all_runs, list_json_backups, restore_from_json_backup, list_runs
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_command():
    """Export all runs to a timestamped JSON file."""
    try:
        export_path = export_all_runs()
        print(f"✅ Export completed: {export_path}")
        print(f"📁 File size: {export_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

def list_backups_command():
    """List all available JSON backups."""
    try:
        backups = list_json_backups()
        if not backups:
            print("ℹ️  No JSON backups found")
            return
        
        print(f"📦 Found {len(backups)} JSON backups:")
        print("=" * 80)
        
        for backup in backups[:10]:  # Show last 10
            print(f"Run: {backup['run_name']}")
            print(f"  📁 File: {backup['filename']}")
            print(f"  📅 Created: {backup['backup_created']}")
            print(f"  📊 Models: {', '.join(backup['models']) if backup['models'] else 'None'}")
            print(f"  📂 Folders: {', '.join(backup['folders']) if backup['folders'] else 'None'}")
            print()
            
        if len(backups) > 10:
            print(f"... and {len(backups) - 10} more backups")
            
    except Exception as e:
        print(f"❌ Failed to list backups: {e}")
        sys.exit(1)

def restore_command(backup_filename: str):
    """Restore a run from JSON backup."""
    try:
        backups = list_json_backups()
        backup_file = None
        
        for backup in backups:
            if backup['filename'] == backup_filename:
                backup_file = backup['filepath']
                break
        
        if not backup_file:
            print(f"❌ Backup file '{backup_filename}' not found")
            print("Available backups:")
            for backup in backups[:5]:
                print(f"  - {backup['filename']}")
            sys.exit(1)
        
        restored_name = restore_from_json_backup(backup_file)
        print(f"✅ Successfully restored run: {restored_name}")
        
    except Exception as e:
        print(f"❌ Restore failed: {e}")
        sys.exit(1)

def health_check_command():
    """Check the health of the storage system."""
    print("🔍 Storage System Health Check")
    print("=" * 40)
    
    # Check database connectivity
    try:
        runs = list_runs()
        print(f"✅ Database: Connected ({len(runs)} runs)")
    except Exception as e:
        print(f"❌ Database: Failed ({e})")
    
    # Check backup directory
    try:
        backups = list_json_backups()
        print(f"✅ JSON Backups: {len(backups)} available")
    except Exception as e:
        print(f"❌ JSON Backups: Failed ({e})")
    
    # Check environment variables
    turso_url = os.getenv("TURSO_DATABASE_URL")
    turso_token = os.getenv("TURSO_AUTH_TOKEN")
    
    if turso_url and turso_token:
        print("✅ Turso: Environment variables present")
    else:
        print("⚠️  Turso: Environment variables missing (using local SQLite)")
    
    # Check write permissions
    try:
        try:
            from src.paths import RESULTS
        except ImportError:
            from src.paths import RESULTS
        test_file = RESULTS / "write_test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print("✅ File System: Write permissions OK")
    except Exception as e:
        print(f"❌ File System: Write permission failed ({e})")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "export":
        export_command()
    elif command == "list-backups":
        list_backups_command()
    elif command == "restore":
        if len(sys.argv) < 3:
            print("❌ Missing backup filename")
            print("Usage: python scripts/disaster_recovery.py restore <backup_filename>")
            sys.exit(1)
        restore_command(sys.argv[2])
    elif command == "health-check":
        health_check_command()
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main() 