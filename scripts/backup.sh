#!/bin/bash
# scripts/backup.sh - Automated backup script for Humanizer Test-Bench
#
# Usage: ./scripts/backup.sh [local|s3|both]
# 
# Cron example (daily at 2 AM):
# 0 2 * * * /opt/humanizer-testbench/scripts/backup.sh both >> /var/log/humanizer-backup.log 2>&1

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/opt/humanizer-testbench/backups}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
APP_DIR="${APP_DIR:-/opt/humanizer-testbench}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-humanizer-testbench}"

# Directories to backup
BACKUP_DIRS=(
    "data"
    "results"
    "cache"
    "logs"
)

# Timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_NAME="humanizer-backup-${TIMESTAMP}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create backup directory
create_backup() {
    log "Starting backup: ${BACKUP_NAME}"
    
    # Create backup directory
    mkdir -p "${BACKUP_DIR}"
    
    # Create temporary directory
    TMP_DIR=$(mktemp -d)
    trap "rm -rf ${TMP_DIR}" EXIT
    
    # Copy directories
    for dir in "${BACKUP_DIRS[@]}"; do
        if [ -d "${APP_DIR}/${dir}" ]; then
            log "Backing up ${dir}..."
            cp -r "${APP_DIR}/${dir}" "${TMP_DIR}/"
        else
            warning "Directory ${dir} not found, skipping"
        fi
    done
    
    # Add metadata
    cat > "${TMP_DIR}/backup-info.txt" << EOF
Backup Name: ${BACKUP_NAME}
Backup Date: $(date)
Hostname: $(hostname)
User: $(whoami)
App Directory: ${APP_DIR}
Directories Backed Up: ${BACKUP_DIRS[@]}
EOF
    
    # Create compressed archive
    log "Creating compressed archive..."
    cd "${TMP_DIR}"
    tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" .
    cd - > /dev/null
    
    # Get backup size
    BACKUP_SIZE=$(du -h "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
    success "Local backup created: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})"
}

# Upload to S3
upload_to_s3() {
    if [ -z "${S3_BUCKET}" ]; then
        warning "S3_BUCKET not set, skipping S3 upload"
        return 1
    fi
    
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install it for S3 uploads."
    fi
    
    log "Uploading to S3: s3://${S3_BUCKET}/${S3_PREFIX}/"
    
    aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
        "s3://${S3_BUCKET}/${S3_PREFIX}/${BACKUP_NAME}.tar.gz" \
        --storage-class STANDARD_IA
    
    if [ $? -eq 0 ]; then
        success "Backup uploaded to S3"
    else
        error "S3 upload failed"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (retention: ${BACKUP_RETENTION_DAYS} days)"
    
    # Local cleanup
    find "${BACKUP_DIR}" -name "humanizer-backup-*.tar.gz" -type f -mtime +${BACKUP_RETENTION_DAYS} -delete
    
    # S3 cleanup (if configured)
    if [ ! -z "${S3_BUCKET}" ] && command -v aws &> /dev/null; then
        log "Cleaning up old S3 backups..."
        
        # List and delete old S3 objects
        CUTOFF_DATE=$(date -d "${BACKUP_RETENTION_DAYS} days ago" +%Y-%m-%d)
        
        aws s3api list-objects-v2 \
            --bucket "${S3_BUCKET}" \
            --prefix "${S3_PREFIX}/" \
            --query "Contents[?LastModified<='${CUTOFF_DATE}'].Key" \
            --output text | \
        while read -r key; do
            if [ ! -z "$key" ]; then
                log "Deleting old S3 backup: $key"
                aws s3 rm "s3://${S3_BUCKET}/${key}"
            fi
        done
    fi
}

# Verify backup
verify_backup() {
    log "Verifying backup..."
    
    # Check if file exists
    if [ ! -f "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" ]; then
        error "Backup file not found"
    fi
    
    # Test archive integrity
    if tar -tzf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" > /dev/null 2>&1; then
        success "Backup integrity verified"
    else
        error "Backup integrity check failed"
    fi
}

# Database backup (if using external database in future)
backup_database() {
    # SQLite databases are included in file backup
    # This function is placeholder for future PostgreSQL/MySQL support
    log "Database backup included in file backup"
}

# Main execution
main() {
    local backup_type="${1:-local}"
    
    log "Starting Humanizer Test-Bench backup (type: ${backup_type})"
    
    # Create backup
    create_backup
    
    # Verify backup
    verify_backup
    
    # Backup database
    backup_database
    
    # Upload to S3 if requested
    if [ "${backup_type}" == "s3" ] || [ "${backup_type}" == "both" ]; then
        upload_to_s3
    fi
    
    # Cleanup old backups
    cleanup_old_backups
    
    log "Backup completed successfully"
}

# Run main function
main "$@"