"""
LightGBM Agent for Supervised Learning
=====================================

LightGBM gradient boosting agent for intelligent Snake game decision making.

Key Features:
- LightGBM model inference
- Feature extraction and preprocessing
- Move prediction with feature importance
- Performance tracking and optimization
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import time
import pickle
from typing import Dict, Any, Optional
import numpy as np
from .base_agent import BaseSupervisedAgent
from utils.print_utils import print_info, print_warning, print_error

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print_warning("[LightGBM Agent] LightGBM not available")


class LightGBMAgent(BaseSupervisedAgent):
    """LightGBM agent for supervised learning."""
    
    description = "LightGBM gradient boosting agent with feature importance analysis"
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)
        self.move_mapping = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    @property
    def agent_name(self) -> str:
        return "LightGBM"
    
    def load_model(self, model_path: str) -> bool:
        """Load trained LightGBM model from file."""
        if not LIGHTGBM_AVAILABLE:
            print_error("LightGBM not available, cannot load model")
            return False
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                print_error(f"Model file not found: {model_path}")
                return False
            
            # Load model (LightGBM supports direct loading)
            if model_path.suffix == '.txt':
                # Text format
                self.model = lgb.Booster(model_file=str(model_path))
            elif model_path.suffix in ['.pkl', '.pickle']:
                # Pickle format
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        self.model = model_data['model']
                        self.feature_names = model_data.get('feature_names')
                    else:
                        self.model = model_data
            else:
                # Try loading as booster
                self.model = lgb.Booster(model_file=str(model_path))
            
            self.is_loaded = True
            self.model_path = str(model_path)
            
            print_info(f"✅ Loaded LightGBM model from {model_path}")
            return True
            
        except Exception as e:
            print_error(f"Failed to load LightGBM model: {e}")
            return False
    
    def predict_move(self, game_state: Dict[str, Any]) -> str:
        """Predict next move using LightGBM model."""
        if not self.is_loaded or self.model is None:
            print_warning("Model not loaded, returning random move")
            return "UP"
        
        start_time = time.time()
        
        try:
            # Extract features
            features = self.extract_features(game_state)
            
            # Make prediction
            prediction = self.model.predict([features], num_iteration=self.model.best_iteration)
            
            # Get predicted class
            if len(prediction.shape) > 1:
                # Multi-class prediction
                predicted_class = np.argmax(prediction[0])
            else:
                # Binary or regression - convert to class
                predicted_class = int(np.round(prediction[0])) % 4
            
            # Map to move
            predicted_move = self.move_mapping.get(predicted_class, "UP")
            
            # Update stats
            self.prediction_count += 1
            self.total_prediction_time += time.time() - start_time
            
            return predicted_move
            
        except Exception as e:
            print_warning(f"LightGBM prediction failed: {e}")
            self.prediction_count += 1
            self.total_prediction_time += time.time() - start_time
            return "UP"
    
    def get_prediction_probabilities(self, game_state: Dict[str, Any]) -> Dict[str, float]:
        """Get prediction probabilities for all moves."""
        if not self.is_loaded or self.model is None:
            return {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}
        
        try:
            # Extract features
            features = self.extract_features(game_state)
            
            # Get prediction
            prediction = self.model.predict([features], num_iteration=self.model.best_iteration)
            
            # Convert to probabilities
            if len(prediction.shape) > 1:
                # Multi-class - use softmax
                probabilities = self._softmax(prediction[0])
            else:
                # Single prediction - distribute based on confidence
                confidence = abs(prediction[0])
                probabilities = np.ones(4) * (1 - confidence) / 3
                best_class = int(np.round(prediction[0])) % 4
                probabilities[best_class] = confidence
            
            # Map to moves
            prob_dict = {}
            for i, move in self.move_mapping.items():
                prob_dict[move] = float(probabilities[i])
            
            return prob_dict
            
        except Exception as e:
            print_warning(f"Probability calculation failed: {e}")
            return {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax function."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the model."""
        if not self.is_loaded or self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importance(importance_type='gain')
            
            # Create feature names if not available
            if self.feature_names is None:
                self.feature_names = [
                    "head_x", "head_y", "food_x", "food_y",
                    "food_dist_x", "food_dist_y",
                    "wall_up", "wall_down", "wall_left", "wall_right",
                    "body_up", "body_down", "body_left", "body_right",
                    "snake_length",
                    "dir_up", "dir_down", "dir_left", "dir_right", "bias"
                ]
            
            # Map importance to feature names
            importance_dict = {}
            for i, imp in enumerate(importance):
                if i < len(self.feature_names):
                    importance_dict[self.feature_names[i]] = float(imp)
            
            return importance_dict
            
        except Exception as e:
            print_warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "agent_name": self.agent_name,
            "model_type": "LightGBM Gradient Boosting",
            "framework": "LightGBM",
            "is_loaded": self.is_loaded,
            "model_path": self.model_path
        }
        
        if self.model is not None:
            try:
                info.update({
                    "num_trees": self.model.num_trees(),
                    "num_features": self.model.num_feature(),
                    "objective": getattr(self.model, 'objective', 'unknown'),
                    "best_iteration": getattr(self.model, 'best_iteration', -1)
                })
            except Exception as e:
                print_warning(f"Could not extract model info: {e}")
        
        return info