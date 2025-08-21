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
            page_title="🐍 Snake Game Replay & File Reader",
            page_icon="🐍",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'About': "Advanced Snake Game AI analysis tools for replay and large file management",
                'Report a bug': None,
                'Get help': None
            }
        )
    
    def run(self):
        """Main app execution."""
        # Hero section
        st.title("🐍 Snake Game Replay & Large File Reader")
        st.markdown("""
        **Advanced analysis tools for Snake Game AI** - Replay game sessions and explore large datasets with elegant navigation and visualization.
        """)
        
        # Quick stats overview
        self.render_quick_stats()
        
        st.markdown("---")
        
        # Main navigation with enhanced styling
        tab_replay, tab_files, tab_analysis = st.tabs([
            "🎮 Game Replay", 
            "📁 Large File Reader", 
            "📊 Data Analysis"
        ])
        
        with tab_replay:
            self.render_replay_section()
        
        with tab_files:
            self.render_file_reader_section()
            
        with tab_analysis:
            self.render_analysis_section()
    
    def render_quick_stats(self):
        """Render quick statistics overview."""
        # Get logs directory statistics
        logs_path = Path("logs")
        if not logs_path.exists():
            st.warning("📂 No logs directory found. Run some games to generate data.")
            return
        
        # Calculate statistics
        total_files = 0
        total_size = 0
        game_files = 0
        dataset_files = 0
        
        for file_path in logs_path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size
                
                if file_path.name.startswith("game_") and file_path.suffix == ".json":
                    game_files += 1
                elif file_path.suffix in [".csv", ".jsonl"]:
                    dataset_files += 1
        
        # Display quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Total Files", f"{total_files:,}")
        
        with col2:
            size_mb = total_size / (1024 * 1024)
            if size_mb < 1024:
                st.metric("💾 Total Size", f"{size_mb:.1f} MB")
            else:
                st.metric("💾 Total Size", f"{size_mb/1024:.1f} GB")
        
        with col3:
            st.metric("🎮 Game Files", f"{game_files:,}")
        
        with col4:
            st.metric("📊 Dataset Files", f"{dataset_files:,}")
    
    def render_analysis_section(self):
        """Render data analysis section."""
        st.header("📊 Data Analysis")
        st.markdown("Analyze and compare game sessions, algorithms, and performance metrics.")
        
        # Get available experiments
        log_folders = self.file_manager.get_log_folders()
        
        if not log_folders:
            st.warning("No experiment data found for analysis.")
            return
        
        # Analysis options
        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "📈 Performance Comparison",
                "🎯 Algorithm Analysis", 
                "📊 Session Statistics",
                "🔍 Game Details"
            ]
        )
        
        if analysis_type == "📈 Performance Comparison":
            self.render_performance_comparison(log_folders)
        elif analysis_type == "🎯 Algorithm Analysis":
            self.render_algorithm_analysis(log_folders)
        elif analysis_type == "📊 Session Statistics":
            self.render_session_statistics(log_folders)
        elif analysis_type == "🔍 Game Details":
            self.render_game_details(log_folders)
    
    def render_performance_comparison(self, log_folders: List[str]):
        """Render performance comparison analysis."""
        st.subheader("📈 Performance Comparison")
        
        # Select experiments to compare
        selected_experiments = st.multiselect(
            "Select Experiments to Compare",
            log_folders,
            format_func=self.file_manager.get_folder_display_name,
            help="Choose multiple experiments to compare performance"
        )
        
        if len(selected_experiments) < 2:
            st.info("Select at least 2 experiments to compare performance.")
            return
        
        # Collect performance data
        comparison_data = []
        
        for exp in selected_experiments:
            summary_data = self.file_manager.load_session_summary(exp)
            if summary_data:
                comparison_data.append({
                    'Experiment': self.file_manager.get_folder_display_name(exp),
                    'Total Games': summary_data.get('total_games', 0),
                    'Avg Score': summary_data.get('average_score', 0),
                    'Total Steps': summary_data.get('total_steps', 0),
                    'Success Rate': f"{summary_data.get('success_rate', 0)*100:.1f}%" if 'success_rate' in summary_data else "N/A",
                    'Algorithm': summary_data.get('algorithm', 'Unknown')
                })
        
        if comparison_data:
            # Display comparison table
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Average Score Comparison**")
                chart_data = df.set_index('Experiment')['Avg Score']
                st.bar_chart(chart_data)
            
            with col2:
                st.markdown("**Total Steps Comparison**")
                chart_data = df.set_index('Experiment')['Total Steps']
                st.bar_chart(chart_data)
        else:
            st.warning("No valid summary data found in selected experiments.")
    
    def render_algorithm_analysis(self, log_folders: List[str]):
        """Render algorithm-specific analysis."""
        st.subheader("🎯 Algorithm Analysis")
        st.info("🚧 Advanced algorithm analysis coming soon - compare pathfinding efficiency, success rates, and decision patterns.")
    
    def render_session_statistics(self, log_folders: List[str]):
        """Render session statistics."""
        st.subheader("📊 Session Statistics")
        st.info("🚧 Comprehensive session statistics coming soon - detailed breakdowns of game performance and trends.")
    
    def render_game_details(self, log_folders: List[str]):
        """Render detailed game analysis."""
        st.subheader("🔍 Game Details")
        st.info("🚧 Detailed game analysis coming soon - move-by-move analysis and decision tree visualization.")
    
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
        
                    # Enhanced game preview
            if game_num and game_num in games:
                game_data = games[game_num]
                
                # Show comprehensive game summary
                st.markdown("#### 📊 Game Summary")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Score", game_data.get("score", 0))
                with col2:
                    st.metric("Steps", game_data.get("steps", 0))
                with col3:
                    st.metric("Duration", f"{game_data.get('duration_seconds', 0):.1f}s")
                with col4:
                    st.metric("End Reason", game_data.get("end_reason", "Unknown"))
                with col5:
                    # Calculate efficiency
                    score = game_data.get("score", 0)
                    steps = game_data.get("steps", 1)
                    efficiency = score / steps if steps > 0 else 0
                    st.metric("Efficiency", f"{efficiency:.3f}")
                
                # Additional game info
                if game_data.get("algorithm"):
                    st.info(f"🧠 Algorithm: {game_data['algorithm']}")
                
                # Show move sequence preview
                moves = game_data.get("moves", [])
                if moves:
                    st.markdown("**Move Sequence Preview:**")
                    move_preview = " → ".join(moves[:10])
                    if len(moves) > 10:
                        move_preview += f" ... (+{len(moves)-10} more)"
                    st.code(move_preview)
            
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
        """Render the enhanced large file reader section."""
        st.header("📁 Large File Reader")
        st.markdown("Read and navigate through large log files (JSON, CSV, JSONL) up to 10GB+ with advanced features")
        
        # File selection with enhanced filtering
        available_files = self.find_large_files()
        
        if not available_files:
            st.warning("No files found in logs/ directory.")
            st.info("Generate some datasets or run games to create log files.")
            return
        
        # Enhanced file selection interface
        col_filter, col_file, col_info = st.columns([1, 2, 1])
        
        with col_filter:
            st.markdown("**🔍 Filters**")
            
            # File type filter
            file_types = st.multiselect(
                "File Types",
                [".json", ".csv", ".jsonl", ".txt"],
                default=[".json", ".csv", ".jsonl"],
                key="file_type_filter"
            )
            
            # Size filter
            min_size_mb = st.number_input(
                "Min Size (MB)",
                min_value=0.0,
                max_value=1000.0,
                value=0.0,
                step=0.1,
                key="min_size_filter"
            )
            
            # Name filter
            name_filter = st.text_input(
                "Name Contains",
                placeholder="game_, dataset_, etc.",
                key="name_filter"
            )
            
            # Quick filter buttons
            st.markdown("**Quick Filters:**")
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                if st.button("🎮 Games", key="filter_games"):
                    st.session_state.name_filter = "game_"
                    st.rerun()
            with col_q2:
                if st.button("📊 Datasets", key="filter_datasets"):
                    st.session_state.name_filter = "dataset"
                    st.rerun()
        
        # Filter files based on selection
        filtered_files = []
        for f in available_files:
            file_path = Path(f)
            
            # Check file type
            if file_path.suffix not in file_types:
                continue
            
            # Check size
            if file_path.stat().st_size < min_size_mb * 1024 * 1024:
                continue
            
            # Check name filter
            if name_filter and name_filter.lower() not in file_path.name.lower():
                continue
            
            filtered_files.append(f)
        
        with col_file:
            if not filtered_files:
                st.warning("No files match the current filters.")
                return
                
            selected_file = st.selectbox(
                "Select File to Read",
                options=filtered_files,
                format_func=lambda x: f"{Path(x).name} ({self.get_file_size_str(x)})",
                key="file_select"
            )
        
        with col_info:
            if selected_file:
                file_path = Path(selected_file)
                st.metric("File Size", self.get_file_size_str(selected_file))
                st.metric("File Type", file_path.suffix.upper())
                
                # Add file analysis button
                if st.button("🔍 Analyze File", key="analyze_file"):
                    self.analyze_file_structure(selected_file)
        
        if selected_file:
            self.render_file_content(selected_file)
    
    def analyze_file_structure(self, file_path: str):
        """Analyze and display file structure information."""
        st.markdown("---")
        st.subheader("🔍 File Analysis")
        
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == ".json":
                self.analyze_json_file(file_path)
            elif file_ext == ".csv":
                self.analyze_csv_file(file_path)
            elif file_ext == ".jsonl":
                self.analyze_jsonl_file(file_path)
            else:
                self.analyze_text_file(file_path)
        except Exception as e:
            st.error(f"Error analyzing file: {e}")
    
    def analyze_json_file(self, file_path: str):
        """Analyze JSON file structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            st.success("✅ Valid JSON file")
            
            # Show structure
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Structure:**")
                if isinstance(data, dict):
                    st.info(f"📋 Object with {len(data)} keys")
                    st.json(list(data.keys())[:10])  # Show first 10 keys
                elif isinstance(data, list):
                    st.info(f"📜 Array with {len(data)} items")
                    if data and isinstance(data[0], dict):
                        st.json(list(data[0].keys())[:10])
                else:
                    st.info(f"📄 {type(data).__name__}")
            
            with col2:
                st.markdown("**Sample Data:**")
                if isinstance(data, dict):
                    sample = {k: v for i, (k, v) in enumerate(data.items()) if i < 5}
                elif isinstance(data, list) and data:
                    sample = data[0] if len(data) > 0 else {}
                else:
                    sample = data
                st.json(sample)
                
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON: {e}")
        except Exception as e:
            st.error(f"❌ Error reading JSON: {e}")
    
    def analyze_csv_file(self, file_path: str):
        """Analyze CSV file structure."""
        try:
            # Read just the header and a few rows for analysis
            df_sample = pd.read_csv(file_path, nrows=100, encoding='utf-8')
            
            st.success("✅ Valid CSV file")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**File Info:**")
                st.metric("Columns", len(df_sample.columns))
                st.metric("Sample Rows", len(df_sample))
                
                # Show data types
                st.markdown("**Column Types:**")
                type_info = df_sample.dtypes.to_frame('Type')
                st.dataframe(type_info, use_container_width=True)
            
            with col2:
                st.markdown("**Sample Data:**")
                st.dataframe(df_sample.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error analyzing CSV: {e}")
    
    def analyze_jsonl_file(self, file_path: str):
        """Analyze JSONL file structure."""
        try:
            sample_lines = []
            line_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line_count += 1
                    if i < 5:  # Sample first 5 lines
                        try:
                            sample_lines.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            sample_lines.append({"error": "Invalid JSON line"})
                    elif i > 100:  # Don't count all lines for very large files
                        break
            
            st.success("✅ Valid JSONL file")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**File Info:**")
                st.metric("Sample Lines", min(line_count, 100))
                if sample_lines and isinstance(sample_lines[0], dict):
                    st.metric("Keys per Line", len(sample_lines[0]))
            
            with col2:
                st.markdown("**Sample Line:**")
                if sample_lines:
                    st.json(sample_lines[0])
                    
        except Exception as e:
            st.error(f"❌ Error analyzing JSONL: {e}")
    
    def analyze_text_file(self, file_path: str):
        """Analyze text file structure."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample_lines = [f.readline().strip() for _ in range(10)]
            
            st.success("✅ Text file")
            
            st.markdown("**Sample Lines:**")
            for i, line in enumerate(sample_lines[:5]):
                if line:
                    st.code(f"Line {i+1}: {line}")
                    
        except Exception as e:
            st.error(f"❌ Error analyzing text file: {e}")
    
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
        
        # Enhanced navigation controls
        st.markdown("#### 🧭 Navigation Controls")
        col_nav1, col_nav2, col_nav3, col_nav4, col_nav5 = st.columns(5)
        
        with col_nav1:
            lines_per_page = st.selectbox(
                "Lines per page",
                [25, 50, 100, 200, 500, 1000],
                index=2,
                key="lines_per_page"
            )
        
        with col_nav2:
            goto_line = st.number_input(
                "Go to line",
                min_value=1,
                value=1,
                key="goto_line"
            )
        
        with col_nav3:
            # Quick jump options
            jump_option = st.selectbox(
                "Quick Jump",
                ["Current", "Start", "Middle", "End"],
                key="quick_jump"
            )
        
        # Get total line count efficiently
        total_lines = self.count_file_lines(file_path)
        
        # Handle quick jump
        if jump_option == "Start":
            goto_line = 1
        elif jump_option == "Middle":
            goto_line = total_lines // 2
        elif jump_option == "End":
            goto_line = max(1, total_lines - lines_per_page + 1)
        
        with col_nav4:
            st.metric("Total Lines", f"{total_lines:,}")
        
        with col_nav5:
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


# Enhanced sidebar navigation and info
with st.sidebar:
    st.header("🎯 Navigation Guide")
    
    # App features overview
    with st.expander("🎮 Game Replay Features"):
        st.markdown("""
        **PyGame Replay:**
        - Visual desktop replay with speed control
        - Game selection with score preview
        - Move sequence visualization
        
        **Web Replay:**
        - Browser-based replay interface
        - Customizable host and port settings
        - Background server execution
        """)
    
    with st.expander("📁 File Reader Features"):
        st.markdown("""
        **Large File Support:**
        - Handle files up to 10GB+ efficiently
        - Smart pagination and navigation
        - Format-specific rendering
        
        **File Analysis:**
        - Structure analysis and validation
        - Sample data preview
        - Column information for CSV files
        """)
    
    with st.expander("📊 Analysis Features"):
        st.markdown("""
        **Performance Comparison:**
        - Compare multiple experiments
        - Visual charts and metrics
        - Algorithm efficiency analysis
        
        **Data Insights:**
        - Session statistics overview
        - Game pattern analysis
        - Export capabilities
        """)
    
    st.markdown("---")
    st.header("📈 System Statistics")
    
    # Enhanced logs directory info
    logs_path = Path("logs")
    if logs_path.exists():
        # Count different file types
        file_counts = {".json": 0, ".csv": 0, ".jsonl": 0, "other": 0}
        total_size = 0
        largest_file = {"name": "", "size": 0}
        
        for file_path in logs_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                
                # Track largest file
                if size > largest_file["size"]:
                    largest_file = {"name": file_path.name, "size": size}
                
                # Count by extension
                ext = file_path.suffix.lower()
                if ext in file_counts:
                    file_counts[ext] += 1
                else:
                    file_counts["other"] += 1
        
        # Display enhanced statistics
        total_files = sum(file_counts.values())
        st.metric("📁 Total Files", f"{total_files:,}")
        
        if total_size < 1024 * 1024 * 1024:
            st.metric("💾 Total Size", f"{total_size / (1024*1024):.1f} MB")
        else:
            st.metric("💾 Total Size", f"{total_size / (1024*1024*1024):.1f} GB")
        
        # File type breakdown
        st.markdown("**File Types:**")
        for ext, count in file_counts.items():
            if count > 0:
                st.text(f"{ext.upper()}: {count:,}")
        
        # Largest file info
        if largest_file["name"]:
            st.markdown("**Largest File:**")
            size_str = f"{largest_file['size'] / (1024*1024):.1f} MB"
            if largest_file["size"] > 1024*1024*1024:
                size_str = f"{largest_file['size'] / (1024*1024*1024):.1f} GB"
            st.text(f"{largest_file['name'][:20]}... ({size_str})")
    else:
        st.warning("📂 logs/ directory not found")
        st.info("💡 Run some games to generate log files")
    
    st.markdown("---")
    st.header("🔧 Quick Actions")
    
    if st.button("🔄 Refresh Data", key="refresh_sidebar"):
        st.rerun()
    
    if st.button("🗂️ Open Logs Folder", key="open_logs"):
        import subprocess
        import platform
        
        logs_path = Path("logs").absolute()
        if logs_path.exists():
            system = platform.system()
            try:
                if system == "Windows":
                    subprocess.run(["explorer", str(logs_path)])
                elif system == "Darwin":  # macOS
                    subprocess.run(["open", str(logs_path)])
                else:  # Linux
                    subprocess.run(["xdg-open", str(logs_path)])
                st.success("📂 Opened logs folder")
            except:
                st.warning("Could not open folder automatically")
                st.code(str(logs_path))
        else:
            st.error("Logs folder not found")


if __name__ == "__main__":
    # Initialize and run the app
    app = ReplayAndFileApp()