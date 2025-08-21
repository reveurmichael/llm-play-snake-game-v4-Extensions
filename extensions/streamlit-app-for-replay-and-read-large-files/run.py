#!/usr/bin/env python3
"""
Launcher script for Streamlit Replay and File Reader App
========================================================

Simple launcher that starts the Streamlit app with proper configuration.

Usage:
    python run.py
    python run.py --port 8502
    python run.py --host 0.0.0.0 --port 8503
"""

import sys
import subprocess
from pathlib import Path
import argparse

def main():
    """Launch the Streamlit app with optional configuration."""
    parser = argparse.ArgumentParser(
        description="Launch Streamlit Replay and File Reader App"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to bind to (default: 8501)"
    )
    parser.add_argument(
        "--browser", 
        action="store_true", 
        help="Open browser automatically"
    )
    
    args = parser.parse_args()
    
    # Get app.py path
    app_path = Path(__file__).parent / "app.py"
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.address", args.host,
        "--server.port", str(args.port)
    ]
    
    if not args.browser:
        cmd.extend(["--server.headless", "true"])
    
    print(f"🚀 Starting Streamlit app at http://{args.host}:{args.port}")
    print(f"📁 App location: {app_path}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⚠️ App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()