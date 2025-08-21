"""
Streamlit App for Replay and Large File Reading
===============================================

Comprehensive Streamlit application for replaying Snake Game AI sessions
and reading large log files with elegant navigation and display.

Key Features:
- Game replay using Task0 infrastructure (PyGame and Web modes)
- Large file reader supporting JSON, CSV, JSONL files up to 10GB+
- Efficient pagination and navigation for large files
- Clean, user-friendly Streamlit interface

Usage:
    streamlit run app.py
"""

__version__ = "1.0"
__author__ = "Snake Game AI Project"

# This extension is a pure Streamlit application
# No game manager or logic classes needed - just the app interface
__all__ = []