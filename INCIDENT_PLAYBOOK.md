# 🚨 Incident Response Playbook

**Quick reference for storage failures and data recovery**

## When Runs Are Not Saving

### 1. Check the Obvious First (30 seconds)
```bash
# Run health check
.\disaster_recovery.bat health-check

# Check environment variables in logs
# Look for: ✅ Turso: Environment variables present
```

### 2. Database Issues? Your Data is Safe! (1 minute)
- **Every run is automatically backed up as JSON** in `./results/pending/`
- **SQLite fallback works** even when Turso is down
- **Check the logs** for these patterns:
  - `✓ Successfully saved run` = Good
  - `❌ Turso unavailable` + `✓ Successfully saved run to local database` = OK, using fallback
  - `💾 Run is preserved in JSON backup` = Database failed, but data is safe

## Quick Recovery Commands

### Export Everything (Weekly Backup)
```bash
.\disaster_recovery.bat export
# Creates: results/full_export_YYYYMMDD_HHMMSS.json
```

### List Available Backups
```bash
.\disaster_recovery.bat list-backups
# Shows all JSON backups with run names and dates
```

### Restore from Backup
```bash
.\disaster_recovery.bat restore run_name_timestamp.json
# Restores a specific run from JSON backup
```

## Storage System Architecture

```
🎯 Primary: Turso Database (Production)
     ↓ (on failure)
🔄 Fallback: Local SQLite 
     ↓ (on failure)
💾 Disaster Recovery: JSON Backups (Always created)
```

## Log Patterns to Watch

### Good Signs ✅
- `✓ Successfully saved run 'xyz' to Turso database`
- `✓ Successfully saved run 'xyz' to local database`
- `JSON backup saved: results/pending/xyz_timestamp.json`

### Warning Signs ⚠️
- `⚠️ Turso API error on attempt 1/3`
- `🔄 Saving run 'xyz' to local SQLite as fallback`

### Red Flags 🚨
- `❌ Failed to save to local database`
- `💾 Run 'xyz' is preserved in JSON backup at: results/pending`

## Quick Fixes

### If Turso is Down
**Action**: Nothing required! System automatically falls back to local SQLite.
**Data Loss**: None
**Impact**: Runs saved locally, will sync when Turso is back

### If Both Databases Fail
**Action**: Check `./results/pending/` for JSON backups
**Data Loss**: None (JSON backups are always created first)
**Recovery**: Use `disaster_recovery.bat restore <filename>`

### If Write Permissions are Wrong
```bash
# Check write permissions
.\disaster_recovery.bat health-check

# Fix permissions (if needed)
chmod -R 755 ./results/
```

## Recovery Scenarios

### Scenario 1: "Turso API is returning 502 errors"
- ✅ System automatically uses local SQLite
- ✅ JSON backups still being created
- ✅ No data loss, no action needed

### Scenario 2: "Local SQLite says read-only database"
- ✅ JSON backups are being created (check logs)
- 🔧 Fix permissions: `chmod 755 ./results/`
- 🔄 Restore from JSON: `.\disaster_recovery.bat restore <file>`

### Scenario 3: "Everything is broken"
- ✅ Check `./results/pending/` for JSON backups
- 📊 List available backups: `.\disaster_recovery.bat list-backups`
- 🔄 Restore individually: `.\disaster_recovery.bat restore <file>`

## Monitoring Commands

### Daily Health Check
```bash
.\disaster_recovery.bat health-check
```

### Weekly Export
```bash
.\disaster_recovery.bat export
# Push the export file to S3/Git for off-site backup
```

### Emergency Recovery
```bash
# 1. List what's available
.\disaster_recovery.bat list-backups

# 2. Restore the most recent run
.\disaster_recovery.bat restore <most_recent_backup.json>

# 3. Check system health
.\disaster_recovery.bat health-check
```

## Contact Points

- **Logs Location**: Check Streamlit app logs for detailed error messages
- **Backup Location**: `./results/pending/` (automatically preserved between restarts)
- **Full Exports**: `./results/full_export_*.json` (manual exports)

## Prevention

- Run `.\disaster_recovery.bat health-check` weekly
- Run `.\disaster_recovery.bat export` weekly and save the file elsewhere
- Monitor logs for `❌` patterns
- Ensure `./results/pending/` directory exists and is writable 