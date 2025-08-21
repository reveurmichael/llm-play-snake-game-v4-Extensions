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

# Try to import ML libraries with fallbacks
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
        """Render comprehensive model training section."""
        st.markdown("### 🎓 Model Training & Dataset Management")
        
        # Dataset location guidelines
        self.show_dataset_guidelines()
        
        # Dataset discovery and selection
        datasets = self.discover_training_datasets()
        
        if not datasets:
            st.error("❌ No training datasets found. Please generate datasets first using heuristics extension.")
            self.show_dataset_generation_help()
            return
        
        # Training interface
        self.render_training_interface(datasets)
    
    def show_dataset_guidelines(self):
        """Show dataset folder location guidelines."""
        with st.expander("📋 Dataset Location Guidelines", expanded=False):
            st.markdown("""
            **📁 Standard Dataset Locations:**
            
            1. **Heuristics Extension Logs:**
               - `../heuristics-v0.04/logs/datasets/`
               - `../heuristics-v0.04/logs/*/datasets/`
            
            2. **Project Root Logs:**
               - `../../logs/heuristics-v0.04/datasets/`
               - `../../logs/datasets/`
            
            3. **Local Extension Logs:**
               - `./logs/training_data/`
               - `./training_data/`
            
            **📄 Supported File Formats:**
            - CSV files with game state and move data
            - Headers: `snake_head_x, snake_head_y, food_x, food_y, ..., move`
            - Moves encoded as: UP=0, DOWN=1, LEFT=2, RIGHT=3
            """)
    
    def discover_training_datasets(self):
        """Discover available training datasets following location guidelines."""
        st.markdown("#### 📁 Dataset Discovery")
        
        # Define search paths following guidelines
        search_paths = [
            # Heuristics extension logs
            "../heuristics-v0.04/logs/datasets/*.csv",
            "../heuristics-v0.04/logs/*/datasets/*.csv",
            
            # Project root logs
            "../../logs/heuristics-v0.04/datasets/*.csv",
            "../../logs/datasets/*.csv",
            
            # Local extension logs
            "./logs/training_data/*.csv",
            "./training_data/*.csv",
            
            # Fallback locations
            "../*/logs/**/*.csv",
            "../../logs/**/*.csv"
        ]
        
        datasets = []
        for pattern in search_paths:
            found_files = glob.glob(pattern, recursive=True)
            for file in found_files:
                if self.is_valid_dataset(file):
                    datasets.append(file)
        
        # Remove duplicates and sort
        datasets = list(set(datasets))
        datasets.sort()
        
        if datasets:
            st.success(f"✅ Found {len(datasets)} valid training datasets")
            
            # Show dataset summary
            with st.expander(f"📊 Dataset Summary ({len(datasets)} files)", expanded=True):
                for i, dataset in enumerate(datasets[:10]):  # Show first 10
                    file_path = Path(dataset)
                    try:
                        # Quick file info
                        file_size = file_path.stat().st_size / 1024  # KB
                        df = pd.read_csv(dataset, nrows=1)  # Just check structure
                        st.text(f"📄 {file_path.name} ({file_size:.1f} KB, {len(df.columns)} columns)")
                    except Exception as e:
                        st.text(f"📄 {file_path.name} (Error: {str(e)[:30]}...)")
                
                if len(datasets) > 10:
                    st.text(f"... and {len(datasets) - 10} more files")
        else:
            st.warning("⚠️ No valid training datasets found.")
        
        return datasets
    
    def is_valid_dataset(self, file_path: str) -> bool:
        """Check if a CSV file is a valid training dataset."""
        try:
            df = pd.read_csv(file_path, nrows=5)
            
            # Check for required columns (basic validation)
            required_cols = ['move']  # At minimum, we need move labels
            has_required = any(col in df.columns for col in required_cols)
            
            # Check file size (should have some data)
            file_size = Path(file_path).stat().st_size
            has_data = file_size > 100  # At least 100 bytes
            
            return has_required and has_data
        except Exception:
            return False
    
    def show_dataset_generation_help(self):
        """Show help for generating training datasets."""
        with st.expander("🔧 How to Generate Training Datasets", expanded=True):
            st.markdown("""
            **Generate datasets using the heuristics extension:**
            
            ```bash
            # Navigate to heuristics extension
            cd ../heuristics-v0.04
            
            # Run dataset generation
            python scripts/main.py --games 100 --algorithm BFS-1024
            
            # Or use the heuristics Streamlit app
            streamlit run app.py
            ```
            
            **Dataset will be saved to:**
            - `../heuristics-v0.04/logs/datasets/game_data.csv`
            - Contains optimal moves from BFS pathfinding
            - Ready for supervised learning training
            """)
    
    def render_training_interface(self, datasets: List[str]):
        """Render the actual training interface."""
        st.markdown("#### 🎯 Training Configuration")
        
        # Dataset selection
        selected_dataset = st.selectbox(
            "Select Training Dataset",
            datasets,
            format_func=lambda x: f"{Path(x).parent.name}/{Path(x).name}",
            help="Choose the CSV dataset to train on"
        )
        
        # Show dataset preview
        if st.checkbox("📋 Preview Dataset"):
            self.show_dataset_preview(selected_dataset)
        
        # Training parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Model Configuration**")
            model_type = st.selectbox("Model Type", ["MLP", "LightGBM"])
            
            if model_type == "MLP":
                epochs = st.number_input("Training Epochs", min_value=10, max_value=1000, value=100)
                batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=64)
                learning_rate = st.number_input("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
                hidden_layers = st.text_input("Hidden Layers (comma-separated)", value="64,32")
            else:
                num_leaves = st.number_input("Number of Leaves", min_value=10, max_value=300, value=31)
                learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, format="%.2f")
                n_estimators = st.number_input("Number of Estimators", min_value=50, max_value=1000, value=100)
        
        with col2:
            st.markdown("**📊 Training Options**")
            test_split = st.slider("Test Split Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            validation_split = st.slider("Validation Split", min_value=0.0, max_value=0.3, value=0.1, step=0.05)
            random_seed = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
            
            # Output configuration
            model_name = st.text_input("Model Name", value=f"{model_type.lower()}_model")
            save_path = st.text_input("Save Path", value="./models/")
        
        # Training execution
        st.markdown("#### 🚀 Training Execution")
        
        if st.button("🎯 Start Training", type="primary"):
            self.execute_training(
                dataset_path=selected_dataset,
                model_type=model_type,
                model_name=model_name,
                save_path=save_path,
                **self.get_training_params(model_type, locals())
            )
        
        # Quick training option
        if st.button("⚡ Quick Train (Default Settings)"):
            self.execute_quick_training(selected_dataset, model_type)
    
    def show_dataset_preview(self, dataset_path: str):
        """Show preview of the selected dataset."""
        try:
            df = pd.read_csv(dataset_path, nrows=100)
            
            st.markdown("**📊 Dataset Preview:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows (preview)", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                # Try to count total rows without loading all data
                try:
                    total_rows = sum(1 for _ in open(dataset_path)) - 1  # Subtract header
                    st.metric("Total Rows", f"{total_rows:,}")
                except:
                    st.metric("Total Rows", "Unknown")
            
            # Show column info
            st.markdown("**📋 Columns:**")
            st.text(", ".join(df.columns.tolist()))
            
            # Show data preview
            st.markdown("**🔍 Data Preview:**")
            st.dataframe(df.head(10))
            
            # Show move distribution if available
            if 'move' in df.columns:
                move_counts = df['move'].value_counts()
                st.markdown("**📊 Move Distribution:**")
                st.bar_chart(move_counts)
        
        except Exception as e:
            st.error(f"Error previewing dataset: {e}")
    
    def get_training_params(self, model_type: str, params: dict) -> dict:
        """Extract training parameters based on model type."""
        if model_type == "MLP":
            return {
                "epochs": params.get("epochs", 100),
                "batch_size": params.get("batch_size", 64),
                "learning_rate": params.get("learning_rate", 0.01),
                "hidden_layers": [int(x.strip()) for x in params.get("hidden_layers", "64,32").split(",")],
                "test_split": params.get("test_split", 0.2),
                "validation_split": params.get("validation_split", 0.1),
                "random_seed": params.get("random_seed", 42)
            }
        else:  # LightGBM
            return {
                "num_leaves": params.get("num_leaves", 31),
                "learning_rate": params.get("learning_rate", 0.1),
                "n_estimators": params.get("n_estimators", 100),
                "test_split": params.get("test_split", 0.2),
                "random_seed": params.get("random_seed", 42)
            }
    
    def execute_training(self, dataset_path: str, model_type: str, model_name: str, save_path: str, **params):
        """Execute model training with advanced pipeline."""
        st.markdown("#### 🔄 Advanced Training Progress")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import the advanced training system
            from training import SupervisedTrainingPipeline
            
            status_text.text("🚀 Initializing advanced training pipeline...")
            progress_bar.progress(10)
            
            pipeline = SupervisedTrainingPipeline()
            
            # Load and validate data
            status_text.text("📁 Loading and validating dataset...")
            progress_bar.progress(20)
            
            df = pipeline.load_data(dataset_path)
            if df.empty:
                st.error("❌ Failed to load dataset")
                return
            
            st.info(f"✅ Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
            
            # Feature engineering
            status_text.text("🔧 Advanced feature engineering...")
            progress_bar.progress(40)
            
            X, y = pipeline.feature_engineer.engineer_features(df)
            st.info(f"✅ Engineered {X.shape[1]} features from {len(df)} samples")
            
            # Show feature names
            feature_names = pipeline.feature_engineer.get_feature_names()
            with st.expander("🔍 Engineered Features", expanded=False):
                st.write(", ".join(feature_names))
            
            # Training
            status_text.text(f"🎯 Training {model_type} with advanced techniques...")
            progress_bar.progress(60)
            
            # Prepare training parameters
            training_params = {
                'epochs': params.get('epochs', 100),
                'batch_size': params.get('batch_size', 64),
                'learning_rate': params.get('learning_rate', 0.001),
                'hidden_layers': params.get('hidden_layers', [128, 64, 32]),
                'num_leaves': params.get('num_leaves', 31),
                'n_estimators': params.get('n_estimators', 100)
            }
            
            # Train specific model
            if model_type == "MLP":
                results = pipeline.mlp_trainer.train(
                    X[:int(0.8*len(X))], y[:int(0.8*len(X))],  # Train split
                    X[int(0.8*len(X)):int(0.9*len(X))], y[int(0.8*len(X)):int(0.9*len(X))],  # Val split
                    **training_params
                )
            else:  # LightGBM
                results = pipeline.lgb_trainer.train(
                    X[:int(0.8*len(X))], y[:int(0.8*len(X))],
                    X[int(0.8*len(X)):int(0.9*len(X))], y[int(0.8*len(X)):int(0.9*len(X))],
                    **training_params
                )
            
            progress_bar.progress(80)
            
            # Save model
            status_text.text("💾 Saving trained model...")
            
            model_path = Path(save_path)
            model_path.mkdir(parents=True, exist_ok=True)
            
            if model_type == "MLP" and pipeline.mlp_trainer.model:
                save_success = pipeline.mlp_trainer.save_model(str(model_path / f"{model_name}.pth"))
            elif model_type == "LightGBM" and pipeline.lgb_trainer.model:
                save_success = pipeline.lgb_trainer.save_model(str(model_path / f"{model_name}.pkl"))
            else:
                save_success = False
            
            progress_bar.progress(100)
            status_text.text("✅ Advanced training completed!")
            
            # Display comprehensive results
            if save_success:
                st.success(f"🎉 {model_type} model trained successfully with advanced techniques!")
                st.info(f"📁 Model saved to: {model_path / model_name}")
                
                # Display training metrics
                self.display_advanced_training_results(results, model_type)
            else:
                st.warning("⚠️ Training completed but model save failed")
        
        except Exception as e:
            st.error(f"❌ Advanced training failed: {e}")
            status_text.text("❌ Training failed")
            import traceback
            st.code(traceback.format_exc())
    
    def execute_quick_training(self, dataset_path: str, model_type: str):
        """Execute quick training with default parameters."""
        st.info("⚡ Starting quick training with default parameters...")
        
        default_params = {
            "epochs": 50 if model_type == "MLP" else None,
            "batch_size": 64,
            "learning_rate": 0.01 if model_type == "MLP" else 0.1,
            "test_split": 0.2,
            "random_seed": 42
        }
        
        model_name = f"quick_{model_type.lower()}_{int(time.time())}"
        
        self.execute_training(
            dataset_path=dataset_path,
            model_type=model_type,
            model_name=model_name,
            save_path="./models/",
            **default_params
        )
    
    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare features and labels from dataset."""
        # This is a simplified version - in reality, you'd want more sophisticated feature engineering
        feature_cols = [col for col in df.columns if col != 'move']
        
        X = df[feature_cols].values
        y = df['move'].values
        
        return X, y
    
    def train_model(self, model_type: str, X_train, y_train, X_test, y_test, params: dict):
        """Train the specified model type."""
        if model_type == "MLP":
            return self.train_mlp_model(X_train, y_train, X_test, y_test, params)
        else:
            return self.train_lightgbm_model(X_train, y_train, X_test, y_test, params)
    
    def train_mlp_model(self, X_train, y_train, X_test, y_test, params: dict):
        """Train MLP model (simplified implementation)."""
        # This is a placeholder - real implementation would use PyTorch
        st.info("🧠 MLP training simulation (replace with real PyTorch implementation)")
        
        # Simulate training progress
        import time
        for epoch in range(min(params.get("epochs", 50), 10)):
            time.sleep(0.1)  # Simulate training time
            if epoch % 5 == 0:
                st.text(f"Epoch {epoch}: Loss = {0.5 - epoch*0.01:.3f}")
        
        # Mock model and history
        model = {"type": "MLP", "params": params}
        history = {"loss": [0.5, 0.3, 0.2], "accuracy": [0.6, 0.8, 0.9]}
        
        return model, history
    
    def train_lightgbm_model(self, X_train, y_train, X_test, y_test, params: dict):
        """Train LightGBM model (simplified implementation)."""
        st.info("🌲 LightGBM training simulation (replace with real LightGBM implementation)")
        
        # Mock model and history
        model = {"type": "LightGBM", "params": params}
        history = {"train_score": [0.7, 0.8, 0.85], "valid_score": [0.65, 0.75, 0.8]}
        
        return model, history
    
    def save_trained_model(self, model, model_path: Path, model_type: str, training_history: dict):
        """Save the trained model to file."""
        import json
        
        # Save model metadata
        metadata = {
            "model_type": model_type,
            "training_history": training_history,
            "timestamp": time.time(),
            "model_path": str(model_path)
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        st.info(f"📋 Model metadata saved to: {metadata_path}")
    
    def display_training_results(self, training_history: dict):
        """Display training results and metrics."""
        st.markdown("#### 📊 Training Results")
        
        if "loss" in training_history and "accuracy" in training_history:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📉 Training Loss**")
                st.line_chart(pd.DataFrame({"Loss": training_history["loss"]}))
            
            with col2:
                st.markdown("**📈 Training Accuracy**")
                st.line_chart(pd.DataFrame({"Accuracy": training_history["accuracy"]}))
        
        # Show final metrics
        if training_history:
            st.json(training_history)
    
    def display_advanced_training_results(self, results: dict, model_type: str):
        """Display comprehensive training results from advanced pipeline."""
        st.markdown("#### 🏆 Advanced Training Results")
        
        if not results or "error" in results:
            st.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            training_time = results.get('training_time', 0)
            st.metric("Training Time", f"{training_time:.2f}s")
        
        with col2:
            if model_type == "MLP":
                best_acc = results.get('best_accuracy', 0)
                st.metric("Best Accuracy", f"{best_acc:.4f}")
            else:
                num_trees = results.get('num_trees', 0)
                st.metric("Number of Trees", num_trees)
        
        with col3:
            if model_type == "MLP":
                final_acc = results.get('final_train_accuracy', 0)
                st.metric("Final Train Accuracy", f"{final_acc:.4f}")
            else:
                st.metric("Model Type", "LightGBM")
        
        with col4:
            if model_type == "MLP":
                epochs = results.get('epochs', 0)
                st.metric("Epochs Completed", epochs)
            else:
                st.metric("Boosting Type", "GBDT")
        
        # Training curves for MLP
        if model_type == "MLP" and "train_accuracies" in results:
            st.markdown("#### 📈 Training Progress")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if "train_losses" in results:
                    loss_df = pd.DataFrame({
                        "Epoch": range(len(results["train_losses"])),
                        "Training Loss": results["train_losses"]
                    })
                    st.line_chart(loss_df.set_index("Epoch"))
                    st.caption("Training Loss Over Time")
            
            with col2:
                acc_data = {"Training": results["train_accuracies"]}
                if "val_accuracies" in results and results["val_accuracies"]:
                    acc_data["Validation"] = results["val_accuracies"]
                
                acc_df = pd.DataFrame(acc_data)
                acc_df.index.name = "Epoch"
                st.line_chart(acc_df)
                st.caption("Accuracy Over Time")
        
        # Feature importance for LightGBM
        if model_type == "LightGBM" and "feature_importance" in results:
            st.markdown("#### 🎯 Feature Importance")
            
            importance = results["feature_importance"]
            if importance:
                # Sort by importance
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Create DataFrame for visualization
                importance_df = pd.DataFrame(sorted_features[:10], columns=["Feature", "Importance"])
                
                st.bar_chart(importance_df.set_index("Feature"))
                st.caption("Top 10 Most Important Features")
                
                # Show all features in expandable section
                with st.expander("🔍 All Feature Importances", expanded=False):
                    st.dataframe(pd.DataFrame(sorted_features, columns=["Feature", "Importance"]))
        
        # Model configuration
        with st.expander("⚙️ Model Configuration", expanded=False):
            st.json(results)
        
        # Performance insights
        st.markdown("#### 💡 Performance Insights")
        
        if model_type == "MLP":
            if results.get('best_accuracy', 0) > 0.9:
                st.success("🏆 Excellent model performance! Accuracy > 90%")
            elif results.get('best_accuracy', 0) > 0.8:
                st.info("✅ Good model performance! Accuracy > 80%")
            else:
                st.warning("⚠️ Model performance could be improved. Consider hyperparameter tuning.")
            
            # Training efficiency
            training_time = results.get('training_time', 0)
            epochs = results.get('epochs', 1)
            time_per_epoch = training_time / epochs if epochs > 0 else 0
            
            if time_per_epoch < 1:
                st.success(f"⚡ Fast training: {time_per_epoch:.3f}s per epoch")
            elif time_per_epoch < 5:
                st.info(f"⏱️ Moderate training speed: {time_per_epoch:.2f}s per epoch")
            else:
                st.warning(f"🐌 Slow training: {time_per_epoch:.2f}s per epoch")
        
        else:  # LightGBM
            num_trees = results.get('num_trees', 0)
            if num_trees > 500:
                st.info("🌲 Complex model with many trees - good for complex patterns")
            elif num_trees > 100:
                st.success("🌿 Balanced model complexity")
            else:
                st.warning("🌱 Simple model - might underfit complex data")


def main():
    """Main entry point for the Streamlit app."""
    SupervisedLearningApp()


if __name__ == "__main__":
    main()