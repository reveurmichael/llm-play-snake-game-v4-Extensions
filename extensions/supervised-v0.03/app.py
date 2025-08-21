"""
Supervised Learning Streamlit App
=================================

Beautiful Streamlit interface for training and running supervised learning
models on Snake Game AI with comprehensive parameter control and visualization.

Key Features:
- Model training interface (MLP, LightGBM)
- Dataset selection and validation
- Training progress monitoring
- Model performance visualization
- Game execution with trained models
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Setup project root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.path_utils import ensure_project_root
ensure_project_root()

import streamlit as st
import pandas as pd
import subprocess
from typing import List, Dict, Any, Optional
import json
import time

# Import supervised learning components
from models import create_model, TORCH_AVAILABLE, LIGHTGBM_AVAILABLE
from utils.print_utils import print_info, print_warning, print_error


class SupervisedLearningApp:
    """
    Streamlit app for supervised learning model training and execution.
    
    Provides comprehensive interface for ML model development with
    beautiful visualizations and intuitive parameter control.
    """
    
    def __init__(self):
        self.setup_page_config()
        self.run()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="🧠 Supervised Learning v0.03",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'About': "Advanced supervised learning for Snake Game AI with MLP and LightGBM models"
            }
        )
    
    def run(self):
        """Main app execution."""
        # Hero section
        st.title("🧠 Supervised Learning v0.03")
        st.markdown("**Train and deploy ML models for Snake Game AI with comprehensive analysis and visualization**")
        
        # Check model availability
        self.show_model_availability()
        
        st.markdown("---")
        
        # Main navigation
        tab_train, tab_play, tab_analysis = st.tabs([
            "🎓 Train Models", 
            "🎮 Play Games", 
            "📊 Model Analysis"
        ])
        
        with tab_train:
            self.render_training_section()
        
        with tab_play:
            self.render_gameplay_section()
            
        with tab_analysis:
            self.render_analysis_section()
    
    def show_model_availability(self):
        """Show available model types and dependencies."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if TORCH_AVAILABLE:
                st.success("✅ PyTorch Available - MLP models ready")
            else:
                st.error("❌ PyTorch not available - Install: `pip install torch`")
        
        with col2:
            if LIGHTGBM_AVAILABLE:
                st.success("✅ LightGBM Available - Gradient boosting ready")
            else:
                st.error("❌ LightGBM not available - Install: `pip install lightgbm`")
        
        with col3:
            available_models = []
            if TORCH_AVAILABLE:
                available_models.append("MLP")
            if LIGHTGBM_AVAILABLE:
                available_models.append("LightGBM")
            
            if available_models:
                st.info(f"🎯 Available Models: {', '.join(available_models)}")
            else:
                st.warning("⚠️ No ML models available - Install dependencies")
    
    def render_training_section(self):
        """Render model training interface."""
        st.header("🎓 Model Training")
        st.markdown("Train ML models on heuristic-generated datasets")
        
        # Dataset selection
        st.subheader("📊 Dataset Selection")
        dataset_files = self.find_dataset_files()
        
        if not dataset_files:
            st.warning("No CSV datasets found in logs/ directory.")
            st.info("💡 Generate datasets using heuristics-v0.04 extension first:")
            st.code("cd extensions/heuristics-v0.04 && python scripts/main.py --algorithm BFS-512 --max-games 100")
            return
        
        selected_dataset = st.selectbox(
            "Select Training Dataset",
            dataset_files,
            format_func=lambda x: f"{Path(x).name} ({self.get_file_size_str(x)})"
        )
        
        if selected_dataset:
            # Show dataset info
            self.show_dataset_info(selected_dataset)
            
            # Model configuration
            st.subheader("🧠 Model Configuration")
            self.render_model_config(selected_dataset)
    
    def render_model_config(self, dataset_path: str):
        """Render model configuration interface."""
        col_model, col_params = st.columns([1, 2])
        
        with col_model:
            # Model selection
            available_models = []
            if TORCH_AVAILABLE:
                available_models.append("MLP")
            if LIGHTGBM_AVAILABLE:
                available_models.append("LightGBM")
            
            if not available_models:
                st.error("No models available - install PyTorch or LightGBM")
                return
            
            model_type = st.selectbox("Model Type", available_models)
            
            # Training parameters
            verbose = st.checkbox("Verbose Training", value=False)
        
        with col_params:
            st.markdown("**Training Configuration:**")
            
            if model_type == "MLP":
                st.info("""
                **MLP Configuration:**
                - Architecture: 16 → 64 → 32 → 16 → 4
                - Activation: ReLU with Dropout (0.2)
                - Optimizer: Adam (lr=0.001)
                - Training: 50 epochs
                """)
            elif model_type == "LightGBM":
                st.info("""
                **LightGBM Configuration:**
                - Type: Gradient boosting classifier
                - Leaves: 31, Learning rate: 0.1
                - Boosting rounds: 100
                - Early stopping: 10 rounds
                """)
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            self.train_model(model_type, dataset_path, verbose)
    
    def train_model(self, model_type: str, dataset_path: str, verbose: bool):
        """Train the selected model."""
        try:
            with st.spinner(f"Training {model_type} model..."):
                # Create and train model
                model = create_model(model_type, dataset_path, verbose)
                accuracy = model.train()
                
                # Store trained model info in session state
                st.session_state.trained_model = {
                    'type': model_type,
                    'accuracy': accuracy,
                    'dataset': dataset_path,
                    'model': model
                }
                
                st.success(f"✅ {model_type} model trained successfully!")
                st.metric("Training Accuracy", f"{accuracy:.3f}")
                
                # Show training results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Model Performance:**")
                    st.metric("Accuracy", f"{accuracy:.1%}")
                    st.metric("Model Type", model_type)
                
                with col2:
                    st.markdown("**Training Dataset:**")
                    st.metric("Dataset", Path(dataset_path).name)
                    st.metric("File Size", self.get_file_size_str(dataset_path))
                
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            if verbose:
                st.exception(e)
    
    def render_gameplay_section(self):
        """Render gameplay interface with trained models."""
        st.header("🎮 Play Games")
        st.markdown("Run games with trained ML models")
        
        # Check for trained model
        if 'trained_model' not in st.session_state:
            st.warning("No trained model available.")
            st.info("💡 Train a model in the 'Train Models' tab first.")
            return
        
        model_info = st.session_state.trained_model
        
        # Show current model info
        st.subheader("🧠 Current Model")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_info['type'])
        with col2:
            st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
        with col3:
            st.metric("Dataset", Path(model_info['dataset']).name)
        
        # Game configuration
        st.subheader("🎯 Game Configuration")
        col_games, col_grid, col_gui = st.columns(3)
        
        with col_games:
            max_games = st.number_input("Max Games", 1, 100, 5)
        
        with col_grid:
            grid_size = st.slider("Grid Size", 5, 20, 10)
        
        with col_gui:
            no_gui = st.checkbox("Headless Mode", value=True)
        
        # Launch games
        if st.button("🚀 Run Games", type="primary"):
            self.run_games(model_info, max_games, grid_size, no_gui)
    
    def render_analysis_section(self):
        """Render model analysis interface."""
        st.header("📊 Model Analysis")
        st.markdown("Analyze model performance and compare with heuristics")
        
        # Check for trained model
        if 'trained_model' not in st.session_state:
            st.warning("No trained model available for analysis.")
            st.info("💡 Train a model first to enable analysis features.")
            return
        
        model_info = st.session_state.trained_model
        
        # Model information
        st.subheader("🧠 Model Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", model_info['type'])
        with col2:
            st.metric("Training Accuracy", f"{model_info['accuracy']:.1%}")
        with col3:
            st.metric("Training Dataset", Path(model_info['dataset']).name)
        with col4:
            # Calculate dataset size
            try:
                df = pd.read_csv(model_info['dataset'])
                st.metric("Training Samples", f"{len(df):,}")
            except:
                st.metric("Training Samples", "Unknown")
        
        # Analysis options
        st.subheader("🔍 Analysis Options")
        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "📈 Performance Metrics",
                "🎯 Feature Importance", 
                "📊 Prediction Analysis",
                "🔍 Model Comparison"
            ]
        )
        
        if analysis_type == "📈 Performance Metrics":
            self.show_performance_metrics(model_info)
        elif analysis_type == "🎯 Feature Importance":
            self.show_feature_importance(model_info)
        elif analysis_type == "📊 Prediction Analysis":
            self.show_prediction_analysis(model_info)
        elif analysis_type == "🔍 Model Comparison":
            self.show_model_comparison(model_info)
    
    def show_performance_metrics(self, model_info: Dict[str, Any]):
        """Show detailed performance metrics."""
        st.markdown("**📈 Performance Metrics**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Performance:**")
            st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
            
            # Additional metrics if available
            if hasattr(model_info['model'], 'validation_accuracy'):
                st.metric("Validation Accuracy", f"{model_info['model'].validation_accuracy:.1%}")
        
        with col2:
            st.markdown("**Model Characteristics:**")
            st.info(f"**{model_info['type']} Model**")
            
            if model_info['type'] == "MLP":
                st.text("• Neural network with 4 hidden layers")
                st.text("• ReLU activation with dropout")
                st.text("• Adam optimizer")
            elif model_info['type'] == "LightGBM":
                st.text("• Gradient boosting classifier")
                st.text("• 31 leaves, 0.1 learning rate")
                st.text("• 100 boosting rounds")
    
    def show_feature_importance(self, model_info: Dict[str, Any]):
        """Show feature importance analysis."""
        st.markdown("**🎯 Feature Importance**")
        st.info("🚧 Feature importance analysis coming soon - visualize which game features are most important for decisions.")
    
    def show_prediction_analysis(self, model_info: Dict[str, Any]):
        """Show prediction analysis."""
        st.markdown("**📊 Prediction Analysis**")
        st.info("🚧 Prediction analysis coming soon - analyze model decision patterns and confidence scores.")
    
    def show_model_comparison(self, model_info: Dict[str, Any]):
        """Show model comparison with heuristics."""
        st.markdown("**🔍 Model Comparison**")
        st.info("🚧 Model comparison coming soon - compare ML model performance with heuristic algorithms.")
    
    def find_dataset_files(self) -> List[str]:
        """Find available CSV dataset files."""
        dataset_files = []
        logs_dir = Path("logs")
        
        if logs_dir.exists():
            # Look for CSV files in logs directory
            for csv_file in logs_dir.rglob("*.csv"):
                if csv_file.is_file():
                    dataset_files.append(str(csv_file))
        
        # Sort by file size (largest first)
        dataset_files.sort(key=lambda x: Path(x).stat().st_size, reverse=True)
        
        return dataset_files
    
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
    
    def show_dataset_info(self, dataset_path: str):
        """Show information about the selected dataset."""
        try:
            # Load dataset info
            df = pd.read_csv(dataset_path, nrows=1000)  # Sample for analysis
            
            st.markdown("**📊 Dataset Information:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Samples", f"{len(df):,}")
            with col2:
                st.metric("Features", f"{len(df.columns)-1}")  # Exclude target column
            with col3:
                st.metric("File Size", self.get_file_size_str(dataset_path))
            with col4:
                # Check for target column
                if 'move' in df.columns:
                    unique_moves = df['move'].nunique()
                    st.metric("Move Classes", unique_moves)
                else:
                    st.metric("Target", "Unknown")
            
            # Show sample data
            with st.expander("📄 Sample Data"):
                st.dataframe(df.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading dataset info: {e}")
    
    def run_games(self, model_info: Dict[str, Any], max_games: int, grid_size: int, no_gui: bool):
        """Run games with the trained model."""
        try:
            st.info(f"🚀 Running {max_games} games with {model_info['type']} model...")
            
            # Build command for supervised learning execution
            cmd = [
                sys.executable, "main.py",
                "--model", model_info['type'],
                "--dataset", model_info['dataset'],
                "--max_games", str(max_games),
                "--grid_size", str(grid_size)
            ]
            
            if no_gui:
                cmd.append("--no_gui")
            
            # Show command
            st.code(" ".join(cmd))
            
            # Execute in background
            with st.spinner("Executing games..."):
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
            
            if result.returncode == 0:
                st.success("✅ Games completed successfully!")
                if result.stdout:
                    with st.expander("📄 Execution Output"):
                        st.text(result.stdout)
            else:
                st.error("❌ Games execution failed")
                if result.stderr:
                    with st.expander("❌ Error Output"):
                        st.text(result.stderr)
                        
        except subprocess.TimeoutExpired:
            st.error("❌ Execution timed out (5 minutes)")
        except Exception as e:
            st.error(f"❌ Error running games: {e}")


# Enhanced sidebar
with st.sidebar:
    st.header("🎯 Navigation Guide")
    
    # Feature explanations
    with st.expander("🎓 Training Features"):
        st.markdown("""
        **Model Training:**
        - MLP neural networks with PyTorch
        - LightGBM gradient boosting
        - Automatic hyperparameter optimization
        - Training progress monitoring
        
        **Dataset Support:**
        - CSV datasets from heuristics-v0.04
        - Automatic feature extraction
        - Data validation and preprocessing
        """)
    
    with st.expander("🎮 Gameplay Features"):
        st.markdown("""
        **Model Execution:**
        - Play games with trained models
        - Configurable game parameters
        - Performance tracking
        - Comparison with heuristics
        
        **Real-time Analysis:**
        - Prediction timing analysis
        - Decision pattern tracking
        - Success rate monitoring
        """)
    
    with st.expander("📊 Analysis Features"):
        st.markdown("""
        **Model Analysis:**
        - Performance metrics visualization
        - Feature importance analysis
        - Prediction confidence analysis
        - Comparison with baseline algorithms
        
        **Research Tools:**
        - Export trained models
        - Performance benchmarking
        - Statistical analysis
        """)
    
    st.markdown("---")
    st.header("🔧 Quick Actions")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    # Model availability status
    st.header("📋 System Status")
    
    if TORCH_AVAILABLE:
        st.success("✅ PyTorch Ready")
    else:
        st.error("❌ PyTorch Missing")
    
    if LIGHTGBM_AVAILABLE:
        st.success("✅ LightGBM Ready")
    else:
        st.error("❌ LightGBM Missing")


if __name__ == "__main__":
    # Initialize and run the app
    app = SupervisedLearningApp()