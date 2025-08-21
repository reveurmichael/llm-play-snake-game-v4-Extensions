"""
Supervised Learning Game Logic
=============================

Game logic for supervised learning models that predict moves based on
trained ML models (MLP, LightGBM).

Design Philosophy:
- Extends BaseGameLogic with model-based move prediction
- Clean separation between game mechanics and ML model
- Supports multiple model types through strategy pattern
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from typing import Optional, Tuple, Dict, Any
from core.game_logic import BaseGameLogic
from core.game_data import BaseGameData
from utils.print_utils import print_info, print_warning
from config.game_constants import END_REASON_MAP


class SupervisedGameData(BaseGameData):
    """
    Game data for supervised learning with model-specific tracking.
    
    Extends BaseGameData with supervised learning specific metrics.
    """

    def __init__(self):
        super().__init__()
        self.model_predictions = []
        self.prediction_confidences = []
        self.model_type = "Unknown"
        print_info("[SupervisedGameData] Initialized supervised learning data tracking")


class SupervisedGameLogic(BaseGameLogic):
    """
    Game logic for supervised learning models.
    
    Extends BaseGameLogic with model-based move prediction capabilities.
    Uses the strategy pattern for different model types.
    
    Design Patterns:
    - Strategy Pattern: Pluggable ML models
    - Template Method: Inherits base game mechanics
    """

    # Use supervised learning game data
    GAME_DATA_CLS = SupervisedGameData

    def __init__(self, grid_size: int = 10, use_gui: bool = True):
        super().__init__(grid_size, use_gui)
        self.model = None
        self.model_type = "Unknown"
        print_info(f"[SupervisedGameLogic] Initialized with {grid_size}x{grid_size} grid")

    def set_model(self, model) -> None:
        """Set the ML model for move prediction.
        
        Args:
            model: Trained ML model with predict method
        """
        self.model = model
        self.model_type = getattr(model, 'model_type', 'Unknown')
        if hasattr(self.game_state, 'model_type'):
            self.game_state.model_type = self.model_type
        print_info(f"[SupervisedGameLogic] Model set: {self.model_type}")

    def get_next_planned_move(self) -> str:
        """Get next move from ML model prediction.
        
        Returns:
            Predicted move as string (UP, DOWN, LEFT, RIGHT)
        """
        if not self.model:
            print_warning("[SupervisedGameLogic] No model available for prediction")
            return "NO_PATH_FOUND"

        try:
            # Get current game state
            game_state = self.get_state_snapshot()
            
            # Get prediction from model
            move = self.model.predict(game_state)
            
            # Store prediction in game data
            if hasattr(self.game_state, 'model_predictions'):
                self.game_state.model_predictions.append(move)
            
            return move

        except Exception as e:
            print_warning(f"[SupervisedGameLogic] Model prediction failed: {e}")
            return "NO_PATH_FOUND"

    def get_next_planned_move_with_explanation(self, game_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Get next move with explanation from ML model.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Tuple of (move, explanation_dict)
        """
        if not self.model:
            explanation = {
                "reasoning": "No model available for prediction",
                "confidence": 0.0,
                "model_type": "None"
            }
            return "NO_PATH_FOUND", explanation

        try:
            # Get prediction from model
            move = self.model.predict(game_state)
            
            # Get confidence if model supports it
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                try:
                    confidence = self.model.predict_proba(game_state)
                except:
                    confidence = 0.0
            
            explanation = {
                "reasoning": f"{self.model_type} model prediction",
                "confidence": confidence,
                "model_type": self.model_type,
                "predicted_move": move
            }
            
            # Store prediction and confidence
            if hasattr(self.game_state, 'model_predictions'):
                self.game_state.model_predictions.append(move)
            if hasattr(self.game_state, 'prediction_confidences'):
                self.game_state.prediction_confidences.append(confidence)
            
            return move, explanation

        except Exception as e:
            explanation = {
                "reasoning": f"Model prediction failed: {str(e)}",
                "confidence": 0.0,
                "model_type": self.model_type,
                "error": str(e)
            }
            return "NO_PATH_FOUND", explanation