#!/usr/bin/env python3
"""
Development script to start both API server and Streamlit UI with hot reload.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_api():
    """Start API server with hot reload."""
    print("🚀 Starting API server on http://localhost:8000")
    print("   Hot reload enabled - watching src/ directory")
    
    # Use the start_api.py script
    subprocess.run([
        sys.executable,
        str(Path(__file__).parent / "start_api.py")
    ])

def start_streamlit():
    """Start Streamlit UI with hot reload."""
    print("🚀 Starting Streamlit UI on http://localhost:8501")
    print("   Hot reload enabled - watching src/ directory")
    
    subprocess.run([
        sys.executable,
        "-m", "streamlit", "run",
        "src/ui.py",
        "--server.runOnSave=true",
        "--server.fileWatcherType=auto"
    ])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start development servers")
    parser.add_argument("--api-only", action="store_true", help="Start only API server")
    parser.add_argument("--ui-only", action="store_true", help="Start only Streamlit UI")
    
    args = parser.parse_args()
    
    if args.api_only:
        start_api()
    elif args.ui_only:
        start_streamlit()
    else:
        print("=" * 60)
        print("Humanizer Test-Bench Development Environment")
        print("=" * 60)
        print()
        print("Starting both servers...")
        print("  - API Server: http://localhost:8000")
        print("  - Streamlit UI: http://localhost:8501")
        print()
        print("Both servers have hot reload enabled!")
        print("Press Ctrl+C to stop")
        print()
        
        # For Windows, start them in separate windows
        if os.name == 'nt':
            import webbrowser
            import threading
            
            # Start API in background thread
            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()
            
            # Wait a moment for API to start
            time.sleep(3)
            
            # Start Streamlit (will block)
            try:
                start_streamlit()
            except KeyboardInterrupt:
                print("\n\nShutting down servers...")
        else:
            # For Unix-like systems, use subprocess
            import multiprocessing
            
            api_process = multiprocessing.Process(target=start_api)
            ui_process = multiprocessing.Process(target=start_streamlit)
            
            api_process.start()
            time.sleep(3)
            ui_process.start()
            
            try:
                api_process.join()
                ui_process.join()
            except KeyboardInterrupt:
                print("\n\nShutting down servers...")
                api_process.terminate()
                ui_process.terminate()

