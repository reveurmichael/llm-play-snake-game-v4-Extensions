"""
Supervised Learning Game Manager v0.03
======================================

Streamlined game manager for supervised learning that uses trained ML models
through agent pattern for intelligent Snake game decision making.

Key Features:
- Agent-based architecture using trained models
- JSON output generation (no JSONL needed)
- CSV-based training data (from logs folder)
- No explanations - direct move predictions
- Performance tracking and analysis

Design Philosophy:
- Template Method Pattern: Inherits from BaseGameManager
- Factory Pattern: Uses agent factory for model instantiation
- Strategy Pattern: Pluggable ML agents (MLP, LightGBM)
- Single Responsibility: Focused on supervised learning only
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path for absolute imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Standard library imports
import json
import time
from typing import Dict, Any, Optional, List

# Simple logging - KISS principle, avoid utils dependency
def print_info(msg): print(f"[INFO] {msg}")
def print_warning(msg): print(f"[WARNING] {msg}")
def print_success(msg): print(f"[SUCCESS] {msg}")
def print_error(msg): print(f"[ERROR] {msg}")
from core.game_manager import BaseGameManager
from extensions.common import EXTENSIONS_LOGS_DIR
from config.game_constants import END_REASON_MAP

# Import supervised-specific components
from .game_logic import SupervisedGameLogic
from .agents import agent_factory


class SupervisedGameManager(BaseGameManager):
    """
    Supervised learning session manager using ML agents.
    
    Demonstrates clean inheritance from BaseGameManager with minimal
    extension-specific code focused on supervised learning.
    
    Design Patterns:
    - Template Method: Inherits base session management structure  
    - Factory Pattern: Uses agent factory for model instantiation
    - Strategy Pattern: Pluggable ML agents (MLP, LightGBM)
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Initialize base game manager
        super().__init__(config)
        
        # Supervised learning specific configuration
        self.agent_name = config.get("agent", "mlp")
        self.model_path = config.get("model_path")
        self.current_agent = None
        
        # Performance tracking
        self.prediction_times: List[float] = []
        self.model_accuracy: float = 0.0
        self.total_predictions: int = 0
        
        # Load agent
        self._load_agent()
    
    def _load_agent(self):
        """Load the specified ML agent."""
        try:
            self.current_agent = agent_factory.create_agent(
                self.agent_name,
                model_path=self.model_path
            )
            
            if self.current_agent and self.current_agent.is_loaded:
                print_success(f"✅ Loaded {self.agent_name} agent successfully")
            else:
                print_warning(f"⚠️ Agent {self.agent_name} loaded but model not available")
                
        except Exception as e:
            print_error(f"Failed to load agent {self.agent_name}: {e}")
            self.current_agent = None
    
    def _create_game_logic(self) -> SupervisedGameLogic:
        """Create supervised learning game logic."""
        return SupervisedGameLogic(
            config=self.config,
            agent=self.current_agent
        )
    
    def _get_next_move(self, game_state: Dict[str, Any]) -> str:
        """Get next move from ML agent."""
        if not self.current_agent:
            print_warning("No agent available, using fallback move")
            return "UP"
        
        start_time = time.time()
        move = self.current_agent.predict_move(game_state)
        prediction_time = time.time() - start_time
        
        # Track performance
        self.prediction_times.append(prediction_time)
        self.total_predictions += 1
        
        return move
    
    def _add_task_specific_game_data(self, game_data_dict: Dict[str, Any]) -> None:
        """Add supervised learning specific data to game data."""
        if self.current_agent:
            # Add agent performance stats
            agent_stats = self.current_agent.get_performance_stats()
            game_data_dict["agent_stats"] = agent_stats
            
            # Add model information
            if hasattr(self.current_agent, 'get_model_info'):
                game_data_dict["model_info"] = self.current_agent.get_model_info()
            
            # Add prediction timing
            if self.prediction_times:
                game_data_dict["prediction_timing"] = {
                    "average_prediction_time": sum(self.prediction_times) / len(self.prediction_times),
                    "total_predictions": len(self.prediction_times),
                    "min_prediction_time": min(self.prediction_times),
                    "max_prediction_time": max(self.prediction_times)
                }
    
    def _display_task_specific_results(self, game_data_dict: Dict[str, Any]) -> None:
        """Display supervised learning specific results."""
        if not self.current_agent:
            return
        
        agent_stats = self.current_agent.get_performance_stats()
        
        print_info("🧠 Supervised Learning Results:")
        print_info(f"   Agent: {agent_stats['agent_name']}")
        print_info(f"   Predictions: {agent_stats['predictions_made']}")
        print_info(f"   Avg Prediction Time: {agent_stats['average_prediction_time']:.4f}s")
        print_info(f"   Predictions/sec: {agent_stats['predictions_per_second']:.2f}")
        
        # Display model-specific information
        if hasattr(self.current_agent, 'get_feature_importance'):
            try:
                importance = self.current_agent.get_feature_importance()
                if importance:
                    print_info("   Top Features:")
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    for feature, score in sorted_features[:5]:
                        print_info(f"     {feature}: {score:.3f}")
            except Exception as e:
                print_warning(f"Could not display feature importance: {e}")
    
    def _add_task_specific_summary_data(self, summary_data: Dict[str, Any]) -> None:
        """Add supervised learning data to session summary."""
        if self.current_agent:
            # Add agent performance summary
            agent_stats = self.current_agent.get_performance_stats()
            summary_data["agent_performance"] = agent_stats
            
            # Add model information
            if hasattr(self.current_agent, 'get_model_info'):
                summary_data["model_info"] = self.current_agent.get_model_info()
            
            # Add session-level statistics
            if self.prediction_times:
                summary_data["session_prediction_stats"] = {
                    "total_predictions": len(self.prediction_times),
                    "average_prediction_time": sum(self.prediction_times) / len(self.prediction_times),
                    "total_prediction_time": sum(self.prediction_times),
                    "predictions_per_second": len(self.prediction_times) / sum(self.prediction_times) if self.prediction_times else 0
                }
    
    def _display_task_specific_summary(self, summary_data: Dict[str, Any]) -> None:
        """Display supervised learning session summary."""
        print_info("🧠 Supervised Learning Session Summary:")
        
        if "agent_performance" in summary_data:
            agent_perf = summary_data["agent_performance"]
            print_info(f"   Agent: {agent_perf.get('agent_name', 'Unknown')}")
            print_info(f"   Total Predictions: {agent_perf.get('predictions_made', 0)}")
            print_info(f"   Average Prediction Time: {agent_perf.get('average_prediction_time', 0):.4f}s")
            print_info(f"   Predictions per Second: {agent_perf.get('predictions_per_second', 0):.2f}")
        
        if "model_info" in summary_data:
            model_info = summary_data["model_info"]
            print_info(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
            print_info(f"   Framework: {model_info.get('framework', 'Unknown')}")
            
            if "total_parameters" in model_info:
                params = model_info["total_parameters"]
                print_info(f"   Parameters: {params:,}")
    
    def _create_extension_subdirectories(self, log_dir: Path) -> None:
        """Create supervised learning specific subdirectories."""
        # Create subdirectories for different agents
        agents_dir = log_dir / "agents"
        agents_dir.mkdir(exist_ok=True)
        
        # Create subdirectory for current agent
        if self.current_agent:
            agent_dir = agents_dir / self.current_agent.agent_name.lower()
            agent_dir.mkdir(exist_ok=True)
        
        # Create models directory for storing trained models
        models_dir = log_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Create training data directory
        training_dir = log_dir / "training_data"
        training_dir.mkdir(exist_ok=True)
    
    def get_available_agents(self) -> Dict[str, str]:
        """Get list of available agents."""
        return agent_factory.list_available_agents()
    
    def switch_agent(self, agent_name: str, model_path: Optional[str] = None) -> bool:
        """Switch to a different agent."""
        try:
            new_agent = agent_factory.create_agent(agent_name, model_path=model_path)
            
            if new_agent:
                self.current_agent = new_agent
                self.agent_name = agent_name
                self.model_path = model_path
                
                # Update game logic with new agent
                if hasattr(self, 'game_logic') and self.game_logic:
                    self.game_logic.agent = new_agent
                
                print_success(f"✅ Switched to {agent_name} agent")
                return True
            else:
                print_error(f"Failed to create {agent_name} agent")
                return False
                
        except Exception as e:
            print_error(f"Error switching to {agent_name}: {e}")
            return False
    
    def export_agent_performance(self, output_path: str) -> bool:
        """Export agent performance data to JSON."""
        if not self.current_agent:
            print_warning("No agent available for performance export")
            return False
        
        try:
            performance_data = {
                "agent_info": self.current_agent.get_model_info() if hasattr(self.current_agent, 'get_model_info') else {},
                "performance_stats": self.current_agent.get_performance_stats(),
                "session_stats": {
                    "total_predictions": len(self.prediction_times),
                    "prediction_times": self.prediction_times,
                    "model_accuracy": self.model_accuracy
                },
                "export_timestamp": time.time()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
            
            print_success(f"📊 Agent performance exported to {output_path}")
            return True
            
        except Exception as e:
            print_error(f"Failed to export agent performance: {e}")
            return False