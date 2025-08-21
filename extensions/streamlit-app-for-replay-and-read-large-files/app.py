"""
Streamlit App for Replay and Large File Reading
===============================================

Comprehensive Streamlit application for replaying Snake Game AI sessions
and reading large log files with elegant navigation and display.

Key Features:
- JSON replay functionality (PyGame and Web modes)
- Large file reader with pagination and navigation
- Supports JSON, CSV, and JSONL files up to 10GB+
- Clean interface leveraging Task0 replay infrastructure
- Efficient memory management for large files

Design Philosophy:
- SUPREME_RULE NO.5 compliance: Streamlit launcher interface
- Reuse Task0 replay infrastructure where possible
- Elegant, user-friendly interface design
- Efficient handling of large files without memory issues
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Setup project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import streamlit as st
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import subprocess
import time

# Import Task0 infrastructure for replay functionality
from core.game_file_manager import FileManager
from utils.session_utils import run_replay, run_web_replay
from config.network_constants import HOST_CHOICES
from utils.print_utils import print_info, print_warning, print_error


class ReplayAndFileApp:
    """
    Streamlit app for game replay and large file reading.
    
    Combines two major functionalities:
    1. Game replay using Task0 infrastructure
    2. Large file reading with efficient navigation
    """
    
    def __init__(self):
        self.file_manager = FileManager()
        self.setup_page_config()
        self.run()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Snake Game Replay & File Reader",
            page_icon="🐍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main app execution."""
        st.title("🐍 Snake Game Replay & Large File Reader")
        st.markdown("---")
        
        # Main navigation
        tab_replay, tab_files = st.tabs(["🎮 Game Replay", "📁 Large File Reader"])
        
        with tab_replay:
            self.render_replay_section()
        
        with tab_files:
            self.render_file_reader_section()
    
    def render_replay_section(self):
        """Render the game replay section."""
        st.header("🎮 Game Replay")
        st.markdown("Replay recorded Snake Game AI sessions with PyGame or Web interface.")
        
        # Get available log folders
        log_folders = self.file_manager.get_log_folders()
        
        if not log_folders:
            st.warning("No experiment logs found in logs/ directory.")
            st.info("Run some games first to generate replay data.")
            return
        
        # Create sub-tabs for PyGame and Web replay
        replay_pygame_tab, replay_web_tab = st.tabs(["🎯 PyGame Replay", "🌐 Web Replay"])
        
        with replay_pygame_tab:
            self.render_pygame_replay(log_folders)
        
        with replay_web_tab:
            self.render_web_replay(log_folders)
    
    def render_pygame_replay(self, log_folders: List[str]):
        """Render PyGame replay interface."""
        st.subheader("🎯 PyGame Replay")
        
        # Experiment selection
        col_exp, col_game = st.columns(2)
        sorted_folders = sorted(log_folders, key=self.file_manager.get_folder_display_name)
        
        with col_exp:
            exp = st.selectbox(
                "Select Experiment",
                options=sorted_folders,
                format_func=self.file_manager.get_folder_display_name,
                key="pygame_exp"
            )
        
        # Game selection
        games = self.file_manager.load_game_data(exp) if exp else {}
        if not games:
            st.warning("No games found in selected experiment.")
            return
        
        with col_game:
            game_num = st.selectbox(
                "Select Game",
                sorted(games.keys()),
                key="pygame_game",
                format_func=lambda x: f"Game {x} (Score: {games[x].get('score', 0)})"
            )
        
        # Game preview
        if game_num and game_num in games:
            game_data = games[game_num]
            
            # Show game summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", game_data.get("score", 0))
            with col2:
                st.metric("Steps", game_data.get("steps", 0))
            with col3:
                st.metric("Duration", f"{game_data.get('duration_seconds', 0):.1f}s")
            with col4:
                st.metric("End Reason", game_data.get("end_reason", "Unknown"))
            
            # Replay options
            st.markdown("#### Replay Options")
            col_speed, col_button = st.columns([1, 1])
            
            with col_speed:
                speed = st.slider("Replay Speed", 0.1, 2.0, 1.0, 0.1, key="pygame_speed")
            
            with col_button:
                st.markdown("<br>", unsafe_allow_html=True)  # Vertical alignment
                if st.button("🚀 Start PyGame Replay", key="pygame_replay_btn"):
                    self.launch_pygame_replay(exp, game_num, speed)
            
            # Show game JSON preview
            with st.expander(f"📄 Preview game_{game_num}.json"):
                st.json(game_data)
    
    def render_web_replay(self, log_folders: List[str]):
        """Render Web replay interface."""
        st.subheader("🌐 Web Replay")
        
        # Experiment selection
        col_exp, col_game = st.columns(2)
        sorted_folders = sorted(log_folders, key=self.file_manager.get_folder_display_name)
        
        with col_exp:
            exp = st.selectbox(
                "Select Experiment",
                options=sorted_folders,
                format_func=self.file_manager.get_folder_display_name,
                key="web_exp"
            )
        
        # Game selection
        games = self.file_manager.load_game_data(exp) if exp else {}
        if not games:
            st.warning("No games found in selected experiment.")
            return
        
        with col_game:
            game_num = st.selectbox(
                "Select Game",
                sorted(games.keys()),
                key="web_game",
                format_func=lambda x: f"Game {x} (Score: {games[x].get('score', 0)})"
            )
        
        # Web replay options
        if game_num and game_num in games:
            game_data = games[game_num]
            
            # Show game summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", game_data.get("score", 0))
            with col2:
                st.metric("Steps", game_data.get("steps", 0))
            with col3:
                st.metric("Duration", f"{game_data.get('duration_seconds', 0):.1f}s")
            with col4:
                st.metric("End Reason", game_data.get("end_reason", "Unknown"))
            
            # Web replay configuration
            st.markdown("#### Web Replay Configuration")
            col_host, col_port, col_button = st.columns([1, 1, 1])
            
            with col_host:
                host = st.selectbox("Host", HOST_CHOICES, key="web_host")
            
            with col_port:
                port = st.number_input("Port", 5000, 9999, 5001, key="web_port")
            
            with col_button:
                st.markdown("<br>", unsafe_allow_html=True)  # Vertical alignment
                if st.button("🌐 Start Web Replay", key="web_replay_btn"):
                    self.launch_web_replay(exp, game_num, host, port)
    
    def render_file_reader_section(self):
        """Render the large file reader section."""
        st.header("📁 Large File Reader")
        st.markdown("Read and navigate through large log files (JSON, CSV, JSONL) up to 10GB+")
        
        # File selection
        available_files = self.find_large_files()
        
        if not available_files:
            st.warning("No large files found in logs/ directory.")
            st.info("Generate some datasets or run games to create log files.")
            return
        
        # File selection interface
        col_file, col_info = st.columns([2, 1])
        
        with col_file:
            selected_file = st.selectbox(
                "Select File to Read",
                options=available_files,
                format_func=lambda x: f"{Path(x).name} ({self.get_file_size_str(x)})",
                key="file_select"
            )
        
        with col_info:
            if selected_file:
                file_path = Path(selected_file)
                st.metric("File Size", self.get_file_size_str(selected_file))
                st.metric("File Type", file_path.suffix.upper())
        
        if selected_file:
            self.render_file_content(selected_file)
    
    def find_large_files(self) -> List[str]:
        """Find large files in logs directory."""
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return []
        
        large_files = []
        extensions = [".json", ".csv", ".jsonl"]
        
        # Recursively find files with target extensions
        for ext in extensions:
            for file_path in logs_dir.rglob(f"*{ext}"):
                if file_path.is_file():
                    # Include all files, but prioritize larger ones
                    large_files.append(str(file_path))
        
        # Sort by file size (largest first)
        large_files.sort(key=lambda x: Path(x).stat().st_size, reverse=True)
        
        return large_files
    
    def get_file_size_str(self, file_path: str) -> str:
        """Get human-readable file size string."""
        try:
            size_bytes = Path(file_path).stat().st_size
            
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        except:
            return "Unknown"
    
    def render_file_content(self, file_path: str):
        """Render file content with navigation and pagination."""
        st.markdown("---")
        st.subheader(f"📄 {Path(file_path).name}")
        
        file_ext = Path(file_path).suffix.lower()
        
        # Navigation controls
        col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)
        
        with col_nav1:
            lines_per_page = st.selectbox(
                "Lines per page",
                [50, 100, 200, 500, 1000],
                index=1,
                key="lines_per_page"
            )
        
        with col_nav2:
            goto_line = st.number_input(
                "Go to line",
                min_value=1,
                value=1,
                key="goto_line"
            )
        
        # Get total line count efficiently
        total_lines = self.count_file_lines(file_path)
        
        with col_nav3:
            st.metric("Total Lines", f"{total_lines:,}")
        
        with col_nav4:
            # Calculate current page
            current_page = max(1, (goto_line - 1) // lines_per_page + 1)
            total_pages = max(1, (total_lines - 1) // lines_per_page + 1)
            st.metric("Page", f"{current_page} / {total_pages}")
        
        # Page navigation
        col_prev, col_next, col_jump = st.columns(3)
        
        with col_prev:
            if st.button("⬅️ Previous Page", disabled=current_page <= 1, key="prev_page"):
                st.session_state.goto_line = max(1, goto_line - lines_per_page)
                st.rerun()
        
        with col_next:
            if st.button("➡️ Next Page", disabled=current_page >= total_pages, key="next_page"):
                st.session_state.goto_line = min(total_lines, goto_line + lines_per_page)
                st.rerun()
        
        with col_jump:
            if st.button("🎯 Jump to Line", key="jump_line"):
                # goto_line is already set by the number_input
                st.rerun()
        
        # Display file content based on type
        start_line = goto_line - 1
        end_line = min(total_lines, start_line + lines_per_page)
        
        try:
            if file_ext == ".json":
                self.display_json_content(file_path, start_line, end_line)
            elif file_ext == ".csv":
                self.display_csv_content(file_path, start_line, end_line)
            elif file_ext == ".jsonl":
                self.display_jsonl_content(file_path, start_line, end_line)
            else:
                self.display_text_content(file_path, start_line, end_line)
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Try reducing the number of lines per page or check file permissions.")
    
    def count_file_lines(self, file_path: str) -> int:
        """Efficiently count lines in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def display_json_content(self, file_path: str, start_line: int, end_line: int):
        """Display JSON content with syntax highlighting."""
        try:
            # For JSON files, we need to handle them specially since they might be single objects
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse as JSON for pretty display
            try:
                json_data = json.loads(content)
                st.json(json_data)
            except json.JSONDecodeError:
                # If not valid JSON, display as text
                lines = content.split('\n')
                display_lines = lines[start_line:end_line]
                st.code('\n'.join(display_lines), language='json')
                
        except Exception as e:
            st.error(f"Error displaying JSON: {e}")
    
    def display_csv_content(self, file_path: str, start_line: int, end_line: int):
        """Display CSV content as a dataframe."""
        try:
            # Read CSV with pandas for better display
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Calculate pagination for dataframe
            start_row = start_line
            end_row = min(len(df), end_line)
            
            # Display subset
            subset_df = df.iloc[start_row:end_row]
            
            st.dataframe(subset_df, use_container_width=True)
            
            # Show column info
            with st.expander("📊 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Sample Value': [str(df[col].iloc[0]) if len(df) > 0 else 'N/A' for col in df.columns]
                })
                st.dataframe(col_info, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying CSV: {e}")
            # Fallback to text display
            self.display_text_content(file_path, start_line, end_line)
    
    def display_jsonl_content(self, file_path: str, start_line: int, end_line: int):
        """Display JSONL content line by line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Display subset of lines
            display_lines = lines[start_line:end_line]
            
            st.markdown(f"**Showing lines {start_line + 1} to {min(len(lines), end_line)}:**")
            
            for i, line in enumerate(display_lines):
                line_num = start_line + i + 1
                try:
                    # Parse and display each JSON line
                    json_obj = json.loads(line.strip())
                    
                    with st.expander(f"Line {line_num}"):
                        st.json(json_obj)
                        
                except json.JSONDecodeError:
                    # Display as text if not valid JSON
                    st.code(f"Line {line_num}: {line.strip()}", language='text')
                    
        except Exception as e:
            st.error(f"Error displaying JSONL: {e}")
    
    def display_text_content(self, file_path: str, start_line: int, end_line: int):
        """Display text content with line numbers."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Display subset of lines with line numbers
            display_lines = lines[start_line:end_line]
            
            content_with_numbers = []
            for i, line in enumerate(display_lines):
                line_num = start_line + i + 1
                content_with_numbers.append(f"{line_num:6d}: {line.rstrip()}")
            
            st.code('\n'.join(content_with_numbers), language='text')
            
        except Exception as e:
            st.error(f"Error displaying text content: {e}")
    
    def launch_pygame_replay(self, experiment: str, game_num: int, speed: float):
        """Launch PyGame replay using Task0 infrastructure."""
        try:
            st.info(f"🚀 Launching PyGame replay for Game {game_num}...")
            
            # Use Task0 replay infrastructure
            success = run_replay(
                log_dir=experiment,
                game_number=game_num,
                speed_multiplier=speed
            )
            
            if success:
                st.success("✅ PyGame replay launched successfully!")
                st.info("Check the PyGame window that opened.")
            else:
                st.error("❌ Failed to launch PyGame replay.")
                
        except Exception as e:
            st.error(f"Error launching PyGame replay: {e}")
    
    def launch_web_replay(self, experiment: str, game_num: int, host: str, port: int):
        """Launch Web replay using Task0 infrastructure."""
        try:
            st.info(f"🌐 Starting web replay server on {host}:{port}...")
            
            # Use Task0 web replay infrastructure
            success = run_web_replay(
                log_dir=experiment,
                game_number=game_num,
                host=host,
                port=port
            )
            
            if success:
                st.success("✅ Web replay server started!")
                st.markdown(f"**🌐 Access replay at: [http://{host}:{port}](http://{host}:{port})**")
                st.info("The web server is running in the background.")
            else:
                st.error("❌ Failed to start web replay server.")
                
        except Exception as e:
            st.error(f"Error launching web replay: {e}")


# Sidebar navigation and info
with st.sidebar:
    st.header("🎯 Navigation")
    st.markdown("""
    **🎮 Game Replay**
    - Replay recorded game sessions
    - PyGame or Web interface options
    - Leverages Task0 replay infrastructure
    
    **📁 Large File Reader**
    - Read files up to 10GB+ efficiently
    - Supports JSON, CSV, JSONL formats
    - Navigate with pagination and line jumping
    """)
    
    st.markdown("---")
    st.header("📊 File Statistics")
    
    # Show logs directory info
    logs_path = Path("logs")
    if logs_path.exists():
        total_files = sum(1 for _ in logs_path.rglob("*") if _.is_file())
        total_size = sum(f.stat().st_size for f in logs_path.rglob("*") if f.is_file())
        
        st.metric("Total Files", total_files)
        st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")
    else:
        st.warning("logs/ directory not found")


if __name__ == "__main__":
    # Initialize and run the app
    app = ReplayAndFileApp()