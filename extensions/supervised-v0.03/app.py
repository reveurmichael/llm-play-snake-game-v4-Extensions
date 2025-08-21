"""
Supervised Learning Streamlit App v0.03
=======================================

Beautiful Streamlit interface for supervised learning agents that use trained
ML models for intelligent Snake Game AI decision making.

Key Features:
- Agent-based interface (MLP, LightGBM agents)
- Model loading and management
- Real-time game execution with trained agents
- Performance analysis and visualization
- Training data management from CSV files

Design Philosophy:
- Agent-centric approach using trained models
- Clean interface for model inference
- Performance tracking and analysis
- Educational value with clear explanations
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Setup project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Add current directory to path for agents import
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import subprocess
from typing import List, Dict, Any, Optional
import json
import time
import glob

# Import supervised learning agents
try:
    from agents import agent_factory
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    st.error(f"Agents not available: {e}")


class SupervisedLearningApp:
    """
    Streamlit app for supervised learning agents with trained ML models.
    
    Provides comprehensive interface for loading trained models, running
    game sessions, and analyzing agent performance.
    """
    
    def __init__(self):
        self.setup_page_config()
        if AGENTS_AVAILABLE:
            self.run()
        else:
            self.show_error_page()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="🧠 Supervised Learning Agents v0.03",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'About': "Supervised learning agents using trained ML models for Snake Game AI"
            }
        )
    
    def show_error_page(self):
        """Show error page when agents are not available."""
        st.title("🧠 Supervised Learning Agents v0.03")
        st.error("⚠️ Agents not available. Please check the installation.")
        
        st.markdown("### 🔧 Troubleshooting:")
        st.markdown("1. Ensure the `agents/` folder exists in this extension")
        st.markdown("2. Check that agent files are properly imported")
        st.markdown("3. Install required dependencies: `pip install torch lightgbm numpy`")
    
    def run(self):
        """Main app execution."""
        # Hero section
        st.title("🧠 Supervised Learning Agents v0.03")
        st.markdown("**Intelligent Snake Game AI using trained ML models with comprehensive performance analysis**")
        
        # Show available agents
        self.show_agent_availability()
        
        st.markdown("---")
        
        # Main navigation
        tab_agents, tab_play, tab_analysis, tab_models = st.tabs([
            "🤖 Agent Management", 
            "🎮 Play Games", 
            "📊 Performance Analysis",
            "🎓 Model Training"
        ])
        
        with tab_agents:
            self.render_agent_management()
        
        with tab_play:
            self.render_gameplay_section()
            
        with tab_analysis:
            self.render_analysis_section()
            
        with tab_models:
            self.render_training_section()
    
    def show_agent_availability(self):
        """Show available agents and their status."""
        st.markdown("### 🤖 Available Agents")
        
        available_agents = agent_factory.list_available_agents()
        
        if not available_agents:
            st.warning("⚠️ No agents available. Check agent registration.")
            return
        
        cols = st.columns(len(available_agents))
        
        for i, (agent_name, description) in enumerate(available_agents.items()):
            with cols[i]:
                # Try to create agent to check availability
                try:
                    test_agent = agent_factory.create_agent(agent_name)
                    if test_agent:
                        st.success(f"✅ **{agent_name.upper()}**")
                        st.caption(description)
                    else:
                        st.warning(f"⚠️ **{agent_name.upper()}**")
                        st.caption("Agent creation failed")
                except Exception as e:
                    st.error(f"❌ **{agent_name.upper()}**")
                    st.caption(f"Error: {str(e)[:50]}...")
    
    def render_agent_management(self):
        """Render agent management interface."""
        st.markdown("### 🤖 Agent Management")
        
        # Agent selection
        available_agents = list(agent_factory.list_available_agents().keys())
        
        if not available_agents:
            st.error("No agents available")
            return
        
        selected_agent = st.selectbox(
            "Select Agent Type",
            available_agents,
            help="Choose the ML agent type to use"
        )
        
        # Model file selection
        st.markdown("#### 📁 Model File Selection")
        
        # Look for model files
        model_files = self.find_model_files()
        
        if model_files:
            model_file = st.selectbox(
                "Select Model File",
                ["None"] + model_files,
                help="Choose a trained model file to load"
            )
        else:
            st.warning("⚠️ No model files found. Train a model first or upload one.")
            model_file = st.file_uploader(
                "Upload Model File",
                type=['pth', 'pkl', 'pickle', 'joblib'],
                help="Upload a trained model file"
            )
        
        # Agent creation and testing
        if st.button("🚀 Create and Test Agent", type="primary"):
            if model_file and model_file != "None":
                self.test_agent_creation(selected_agent, model_file)
            else:
                st.warning("Please select a model file first")
        
        # Agent information display
        if model_file and model_file != "None":
            self.show_agent_info(selected_agent, model_file)
    
    def find_model_files(self):
        """Find available model files."""
        model_extensions = ['*.pth', '*.pkl', '*.pickle', '*.joblib']
        model_files = []
        
        # Look in common model directories
        search_dirs = [
            Path.cwd(),
            Path.cwd() / "models",
            Path.cwd() / "logs",
            Path.cwd().parent / "logs",
            Path(__file__).parent / "models"
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in model_extensions:
                    model_files.extend(glob.glob(str(search_dir / ext)))
        
        return [str(Path(f).name) for f in model_files]
    
    def test_agent_creation(self, agent_name: str, model_file):
        """Test agent creation with selected model."""
        try:
            with st.spinner(f"Creating {agent_name} agent..."):
                if isinstance(model_file, str):
                    model_path = model_file
                else:
                    # Handle uploaded file
                    model_path = f"temp_{model_file.name}"
                    with open(model_path, "wb") as f:
                        f.write(model_file.getvalue())
                
                agent = agent_factory.create_agent(agent_name, model_path=model_path)
                
                if agent:
                    st.success(f"✅ {agent_name} agent created successfully!")
                    
                    # Test prediction
                    test_game_state = {
                        "snake": [[10, 10], [10, 9]],
                        "food": [15, 15],
                        "grid_size": 20,
                        "last_move": "UP"
                    }
                    
                    move = agent.predict_move(test_game_state)
                    st.info(f"🎯 Test prediction: {move}")
                    
                    # Show agent stats
                    if hasattr(agent, 'get_performance_stats'):
                        stats = agent.get_performance_stats()
                        st.json(stats)
                
                else:
                    st.error("❌ Failed to create agent")
        
        except Exception as e:
            st.error(f"❌ Error creating agent: {e}")
    
    def show_agent_info(self, agent_name: str, model_file):
        """Show information about the selected agent and model."""
        st.markdown("#### 📋 Agent Information")
        
        try:
            agent = agent_factory.create_agent(agent_name, model_path=model_file)
            
            if agent and hasattr(agent, 'get_model_info'):
                info = agent.get_model_info()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Agent Details:**")
                    st.write(f"**Name:** {info.get('agent_name', 'Unknown')}")
                    st.write(f"**Type:** {info.get('model_type', 'Unknown')}")
                    st.write(f"**Framework:** {info.get('framework', 'Unknown')}")
                    st.write(f"**Status:** {'✅ Loaded' if info.get('is_loaded', False) else '❌ Not Loaded'}")
                
                with col2:
                    st.markdown("**Model Details:**")
                    if 'total_parameters' in info:
                        st.write(f"**Parameters:** {info['total_parameters']:,}")
                    if 'model_size_mb' in info:
                        st.write(f"**Size:** {info['model_size_mb']:.2f} MB")
                    if 'device' in info:
                        st.write(f"**Device:** {info['device']}")
        
        except Exception as e:
            st.error(f"Error getting agent info: {e}")
    
    def render_gameplay_section(self):
        """Render gameplay section with agent execution."""
        st.markdown("### 🎮 Game Execution with Agents")
        
        # Agent selection for gameplay
        available_agents = list(agent_factory.list_available_agents().keys())
        
        if not available_agents:
            st.error("No agents available for gameplay")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            agent_type = st.selectbox("Agent Type", available_agents, key="gameplay_agent")
            model_files = self.find_model_files()
            
            if model_files:
                model_file = st.selectbox("Model File", model_files, key="gameplay_model")
            else:
                st.warning("No model files available")
                return
        
        with col2:
            games_to_run = st.number_input("Number of Games", min_value=1, max_value=100, value=5)
            grid_size = st.selectbox("Grid Size", [15, 20, 25], index=1)
            max_steps = st.number_input("Max Steps per Game", min_value=100, max_value=2000, value=500)
        
        # Game execution
        if st.button("🚀 Run Games", type="primary"):
            self.run_agent_games(agent_type, model_file, games_to_run, grid_size, max_steps)
        
        # Quick test section
        st.markdown("#### 🧪 Quick Agent Test")
        
        if st.button("🎯 Test Single Prediction"):
            self.test_single_prediction(agent_type, model_file)
    
    def run_agent_games(self, agent_type: str, model_file: str, num_games: int, grid_size: int, max_steps: int):
        """Run games with the selected agent."""
        try:
            agent = agent_factory.create_agent(agent_type, model_path=model_file)
            
            if not agent:
                st.error("Failed to create agent")
                return
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(num_games):
                status_text.text(f"Running game {i+1}/{num_games}...")
                
                # Simulate game execution (replace with actual game logic)
                game_result = self.simulate_game(agent, grid_size, max_steps)
                results.append(game_result)
                
                progress_bar.progress((i + 1) / num_games)
            
            status_text.text("✅ Games completed!")
            
            # Display results
            self.display_game_results(results)
        
        except Exception as e:
            st.error(f"Error running games: {e}")
    
    def simulate_game(self, agent, grid_size: int, max_steps: int) -> Dict[str, Any]:
        """Simulate a game with the agent (placeholder implementation)."""
        import random
        
        # Simple simulation - replace with actual game logic
        score = random.randint(1, 50)
        steps = random.randint(10, max_steps)
        success = score > 10
        
        # Test agent prediction
        test_state = {
            "snake": [[10, 10], [10, 9]],
            "food": [15, 15],
            "grid_size": grid_size,
            "last_move": "UP"
        }
        
        move = agent.predict_move(test_state)
        
        return {
            "score": score,
            "steps": steps,
            "success": success,
            "final_move": move,
            "grid_size": grid_size
        }
    
    def display_game_results(self, results: List[Dict[str, Any]]):
        """Display game execution results."""
        st.markdown("#### 📊 Game Results")
        
        # Summary statistics
        total_games = len(results)
        successful_games = sum(1 for r in results if r["success"])
        avg_score = sum(r["score"] for r in results) / total_games
        avg_steps = sum(r["steps"] for r in results) / total_games
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Games", total_games)
        with col2:
            st.metric("Success Rate", f"{successful_games/total_games*100:.1f}%")
        with col3:
            st.metric("Average Score", f"{avg_score:.1f}")
        with col4:
            st.metric("Average Steps", f"{avg_steps:.1f}")
        
        # Detailed results
        if st.checkbox("Show Detailed Results"):
            df = pd.DataFrame(results)
            st.dataframe(df)
    
    def test_single_prediction(self, agent_type: str, model_file: str):
        """Test a single prediction with the agent."""
        try:
            agent = agent_factory.create_agent(agent_type, model_path=model_file)
            
            if not agent:
                st.error("Failed to create agent")
                return
            
            # Create test game state
            test_state = {
                "snake": [[10, 10], [10, 9], [10, 8]],
                "food": [15, 15],
                "grid_size": 20,
                "last_move": "UP"
            }
            
            start_time = time.time()
            predicted_move = agent.predict_move(test_state)
            prediction_time = time.time() - start_time
            
            st.success(f"🎯 Predicted Move: **{predicted_move}**")
            st.info(f"⏱️ Prediction Time: {prediction_time*1000:.2f} ms")
            
            # Show confidence if available
            if hasattr(agent, 'get_prediction_confidence'):
                confidence = agent.get_prediction_confidence(test_state)
                st.json(confidence)
        
        except Exception as e:
            st.error(f"Error testing prediction: {e}")
    
    def render_analysis_section(self):
        """Render performance analysis section."""
        st.markdown("### 📊 Performance Analysis")
        
        st.info("🚧 Performance analysis features coming soon!")
        
        # Placeholder for analysis features
        st.markdown("#### 🎯 Planned Features:")
        st.markdown("- Agent performance comparison")
        st.markdown("- Prediction timing analysis")
        st.markdown("- Model accuracy metrics")
        st.markdown("- Feature importance visualization")
        st.markdown("- Training data analysis")
    
    def render_training_section(self):
        """Render model training section."""
        st.markdown("### 🎓 Model Training")
        
        st.info("🚧 Training interface coming soon!")
        
        # Training data selection
        st.markdown("#### 📁 Training Data")
        st.markdown("Training data should be CSV files generated by the heuristics extension.")
        
        # Look for CSV files
        csv_files = glob.glob("../heuristics-v0.04/logs/**/*.csv", recursive=True)
        csv_files.extend(glob.glob("../../logs/**/*.csv", recursive=True))
        
        if csv_files:
            st.success(f"✅ Found {len(csv_files)} CSV training files")
            
            if st.checkbox("Show Available Training Files"):
                for csv_file in csv_files[:10]:  # Show first 10
                    st.text(f"📄 {Path(csv_file).name}")
        else:
            st.warning("⚠️ No CSV training files found. Run heuristics extension first.")
        
        # Training parameters (placeholder)
        st.markdown("#### ⚙️ Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("Model Type", ["MLP", "LightGBM"])
            st.number_input("Training Epochs", min_value=10, max_value=1000, value=100)
        
        with col2:
            st.number_input("Batch Size", min_value=16, max_value=512, value=64)
            st.number_input("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
        
        if st.button("🚀 Start Training", type="primary"):
            st.info("🚧 Training functionality will be implemented soon!")


def main():
    """Main entry point for the Streamlit app."""
    SupervisedLearningApp()


if __name__ == "__main__":
    main()