#!/usr/bin/env python3
"""
Start the FastAPI backend server with hot reload.
"""

import uvicorn
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Configure logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting Humanizer Test-Bench API on {host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print("Hot reload enabled - changes will auto-restart the server")
    print("Press Ctrl+C to stop")
    
    # Configure reload directories - watch src/ directory for changes
    reload_dirs = [str(ROOT / "src")]
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        reload_dirs=reload_dirs,
        reload_includes=["*.py"],
        log_level="info"
    )

