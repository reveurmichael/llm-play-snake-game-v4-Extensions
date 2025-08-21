#!/usr/bin/env python3
"""
Supervised Learning App Launcher
===============================

Simple launcher for the supervised learning Streamlit app with proper
path configuration and dependency checking.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the supervised learning Streamlit app."""
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit available")
    except ImportError:
        print("❌ Streamlit not available. Install with: pip install streamlit")
        return
    
    # Check agents
    try:
        from agents import agent_factory
        agents = agent_factory.list_available_agents()
        print(f"✅ Agents available: {list(agents.keys())}")
    except ImportError as e:
        print(f"❌ Agents not available: {e}")
        return
    
    # Get the app.py path
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"❌ App file not found: {app_path}")
        return
    
    print(f"🚀 Launching Supervised Learning App...")
    print(f"📁 App path: {app_path}")
    print(f"🌐 Open your browser to the URL shown below")
    print("-" * 60)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", "8502",  # Use different port from main app
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main()