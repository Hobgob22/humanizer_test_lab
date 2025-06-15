#!/bin/bash
# scripts/health_check.sh - Health check script for monitoring

set -e

# Configuration
HEALTH_URL="${HEALTH_URL:-http://localhost:8501/_stcore/health}"
TIMEOUT="${TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-5}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check health
check_health() {
    local retry=0
    
    while [ $retry -lt $MAX_RETRIES ]; do
        echo -n "Checking health (attempt $((retry + 1))/$MAX_RETRIES)... "
        
        if curl -sf --max-time $TIMEOUT "$HEALTH_URL" > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            return 0
        else
            echo -e "${RED}FAILED${NC}"
            
            if [ $retry -lt $((MAX_RETRIES - 1)) ]; then
                echo "Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            fi
        fi
        
        retry=$((retry + 1))
    done
    
    return 1
}

# Function to check disk space
check_disk_space() {
    echo -n "Checking disk space... "
    
    # Get disk usage percentage (remove % sign)
    USAGE=$(df -h /app 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ -z "$USAGE" ]; then
        echo -e "${YELLOW}UNKNOWN${NC}"
        return 1
    elif [ "$USAGE" -gt 90 ]; then
        echo -e "${RED}CRITICAL (${USAGE}% used)${NC}"
        return 1
    elif [ "$USAGE" -gt 80 ]; then
        echo -e "${YELLOW}WARNING (${USAGE}% used)${NC}"
        return 0
    else
        echo -e "${GREEN}OK (${USAGE}% used)${NC}"
        return 0
    fi
}

# Function to check memory usage
check_memory() {
    echo -n "Checking memory usage... "
    
    if command -v free >/dev/null 2>&1; then
        # Get memory usage percentage
        USAGE=$(free | awk 'NR==2 {printf "%.0f", $3/$2 * 100}')
        
        if [ "$USAGE" -gt 90 ]; then
            echo -e "${RED}CRITICAL (${USAGE}% used)${NC}"
            return 1
        elif [ "$USAGE" -gt 80 ]; then
            echo -e "${YELLOW}WARNING (${USAGE}% used)${NC}"
            return 0
        else
            echo -e "${GREEN}OK (${USAGE}% used)${NC}"
            return 0
        fi
    else
        echo -e "${YELLOW}SKIPPED (free command not available)${NC}"
        return 0
    fi
}

# Function to check API keys
check_api_keys() {
    echo -n "Checking API keys... "
    
    local missing=0
    local keys=""
    
    # Check required API keys
    for key in OPENAI_API_KEY GPTZERO_API_KEY SAPLING_API_KEY; do
        if [ -z "${!key}" ]; then
            keys="$keys $key"
            missing=$((missing + 1))
        fi
    done
    
    if [ $missing -eq 0 ]; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}MISSING:$keys${NC}"
        return 1
    fi
}

# Function to check container status
check_container() {
    echo -n "Checking container status... "
    
    if docker ps --format '{{.Names}}' | grep -q "humanizer-testbench"; then
        echo -e "${GREEN}RUNNING${NC}"
        return 0
    else
        echo -e "${RED}NOT RUNNING${NC}"
        return 1
    fi
}

# Main health check
main() {
    echo "======================================"
    echo "Humanizer Test-Bench Health Check"
    echo "======================================"
    echo "Time: $(date)"
    echo ""
    
    local exit_code=0
    
    # Run all checks
    check_health || exit_code=1
    check_container || exit_code=1
    check_disk_space || exit_code=1
    check_memory || exit_code=1
    
    # Check API keys only if .env exists
    if [ -f .env ]; then
        source .env
        check_api_keys || exit_code=1
    fi
    
    echo ""
    echo "======================================"
    
    if [ $exit_code -eq 0 ]; then
        echo -e "Overall Status: ${GREEN}HEALTHY${NC}"
    else
        echo -e "Overall Status: ${RED}UNHEALTHY${NC}"
    fi
    
    echo "======================================"
    
    exit $exit_code
}

# Run main function
main